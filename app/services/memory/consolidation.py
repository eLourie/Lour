"""
app/services/memory/consolidation.py

Consolidation: the background process that turns raw session activity into
durable long-term memory (ADR-012).

On a schedule (APScheduler, in-process), for every session with a live working
window it:

  1. Reads the short-term window + rolling summary.
  2. Extracts a handful of salient, self-contained facts via the LLM.
  3. Deduplicates each fact against existing long-term memories (cosine ≥
     ``dedup_threshold`` → skip).
  4. Scores importance (LLM-as-judge) and writes facts clearing
     ``min_importance`` to long-term, recording a ``consolidation`` episodic event.

APScheduler (not Celery/arq) is the deliberate default for a single-user
instance: consolidation is a *schedule*, not a high-throughput queue. ``arq``
is the documented scale-up path, not the baseline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.core.telemetry import traced

if TYPE_CHECKING:
    from app.core.config import MemorySettings
    from app.services.llm.structured import StructuredOutputService
    from app.services.memory.episodic import EpisodicMemory
    from app.services.memory.long_term import LongTermMemory
    from app.services.memory.scoring import ImportanceScorer
    from app.services.memory.short_term import ShortTermMemory

logger = get_logger(__name__)

_EXTRACT_SYSTEM = (
    "You extract durable, self-contained facts worth remembering long-term from "
    "a conversation. Each fact must stand alone without the conversation for "
    "context (resolve pronouns, name the subject). Ignore small talk and "
    "transient details. Return an empty list if nothing is worth keeping."
)


class ExtractedFacts(BaseModel):
    """Facts distilled from a session transcript."""

    facts: list[str] = Field(default_factory=list)


class ConsolidationReport(BaseModel):
    """Outcome of consolidating one session."""

    session_id: str
    extracted: int = 0
    written: int = 0
    skipped_duplicate: int = 0
    skipped_low_importance: int = 0


class ConsolidationService:
    """Extracts, dedups and persists facts from sessions on a schedule."""

    def __init__(
        self,
        *,
        short_term: ShortTermMemory,
        long_term: LongTermMemory,
        episodic: EpisodicMemory,
        scorer: ImportanceScorer,
        structured: StructuredOutputService,
        settings: MemorySettings,
    ) -> None:
        self._short_term = short_term
        self._long_term = long_term
        self._episodic = episodic
        self._scorer = scorer
        self._structured = structured
        self._settings = settings
        self._scheduler = AsyncIOScheduler()

    # Scheduler lifecycle

    def start(self) -> None:
        """Register the interval job and start the scheduler (no-op if disabled)."""
        if not self._settings.consolidation_enabled:
            logger.info("consolidation_disabled")
            return
        self._scheduler.add_job(
            self.run_once,
            trigger="interval",
            seconds=self._settings.consolidation_interval_s,
            id="memory_consolidation",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )
        self._scheduler.start()
        logger.info(
            "consolidation_started", interval_s=self._settings.consolidation_interval_s
        )

    async def shutdown(self) -> None:
        """Stop the scheduler if it is running."""
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("consolidation_stopped")

    # Work

    async def run_once(self) -> list[ConsolidationReport]:
        """Consolidate every session that currently has a working window."""
        reports: list[ConsolidationReport] = []
        async for session_id in self._short_term.active_sessions():
            try:
                reports.append(await self.consolidate_session(session_id))
            except Exception:
                # One bad session must never take down the scheduled run.
                logger.exception("consolidation_session_failed", session_id=session_id)
        logger.info("consolidation_run_complete", sessions=len(reports))
        return reports

    @traced("memory_consolidate_session")
    async def consolidate_session(self, session_id: str) -> ConsolidationReport:
        """Extract → dedup → score → persist facts for a single session."""
        transcript = await self._build_transcript(session_id)
        report = ConsolidationReport(session_id=session_id)
        if not transcript.strip():
            return report

        extracted = await self._structured.complete(
            [
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user", "content": f"Conversation:\n{transcript}"},
            ],
            schema=ExtractedFacts,
        )
        facts = [f.strip() for f in extracted.facts if f.strip()]
        report.extracted = len(facts)

        for fact in facts:
            if await self._long_term.nearest_cosine(fact) >= self._settings.dedup_threshold:
                report.skipped_duplicate += 1
                continue
            importance = await self._scorer.score(fact)
            if importance < self._settings.min_importance:
                report.skipped_low_importance += 1
                continue
            await self._long_term.write(
                fact,
                importance=importance,
                session_id=session_id,
                metadata={"origin": "consolidation"},
            )
            report.written += 1

        if report.written:
            await self._episodic.record(
                session_id,
                event_type="consolidation",
                content=f"Consolidated {report.written} fact(s) into long-term memory.",
                metadata=report.model_dump(),
            )
        logger.info(
            "consolidation_session_done",
            session_id=session_id,
            extracted=report.extracted,
            written=report.written,
            duplicates=report.skipped_duplicate,
            low_importance=report.skipped_low_importance,
        )
        return report

    async def _build_transcript(self, session_id: str) -> str:
        """Assemble the rolling summary + verbatim window into one transcript."""
        parts: list[str] = []
        summary = await self._short_term.get_summary(session_id)
        if summary:
            parts.append(f"Earlier summary: {summary}")
        window = await self._short_term.get_window(session_id)
        parts.extend(f"{turn.role}: {turn.content}" for turn in window)
        return "\n".join(parts)
