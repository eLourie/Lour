"""
app/core/policy.py

Unified Policy declaration + PolicyResolver.

Design (ADR-011):
  - Policy is declared in ONE place (here + skill YAML).
  - Enforcement happens in TWO places:
      * BudgetEnforcer   — in the Orchestration layer (graph node)
      * ToolGate         — in the Tools layer (call boundary)
  - Skills and agents DECLARE policy (data); the graph and tools layer ENFORCE it (code).

PolicyResolver composes layers with most-restrictive-wins semantics:
    defaults (config) ← skill (YAML) ← agent ← request

This means:
  - max_cost_tokens: take the MINIMUM across all layers that specify it.
  - allowed_tools: take the INTERSECTION (most restrictive set).
  - requires_confirmation: True wins over False at any layer.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


# Budget


class BudgetPolicy(BaseModel):
    """Token / iteration / time budget for an agent run."""

    max_cost_tokens: int | None = Field(
        default=None,
        description="Maximum total tokens (input + output) for the run.",
    )
    max_iterations: int | None = Field(
        default=None,
        description="Maximum number of agent loop iterations.",
    )
    max_tool_calls: int | None = Field(
        default=None,
        description="Maximum number of tool invocations.",
    )
    max_duration_s: float | None = Field(
        default=None,
        description="Wall-clock time limit in seconds.",
    )

    def merge_restrictive(self, other: BudgetPolicy) -> BudgetPolicy:
        """Return a new BudgetPolicy that is the most restrictive of self and other."""

        def _min_int(a: int | None, b: int | None) -> int | None:
            if a is None:
                return b
            if b is None:
                return a
            return min(a, b)

        def _min_float(a: float | None, b: float | None) -> float | None:
            if a is None:
                return b
            if b is None:
                return a
            return min(a, b)

        return BudgetPolicy(
            max_cost_tokens=_min_int(self.max_cost_tokens, other.max_cost_tokens),
            max_iterations=_min_int(self.max_iterations, other.max_iterations),
            max_tool_calls=_min_int(self.max_tool_calls, other.max_tool_calls),
            max_duration_s=_min_float(self.max_duration_s, other.max_duration_s),
        )



# Approval rules (HITL)


class ApprovalRules(BaseModel):
    """Rules governing which tool calls require human confirmation."""

    tools_requiring_approval: set[str] = Field(
        default_factory=set,
        description="Tool names that always trigger an interrupt for user approval.",
    )
    require_approval_on_side_effects: bool = Field(
        default=False,
        description="If True, any tool with side_effects=True requires approval.",
    )

    def merge_restrictive(self, other: ApprovalRules) -> ApprovalRules:
        return ApprovalRules(
            tools_requiring_approval=self.tools_requiring_approval | other.tools_requiring_approval,
            require_approval_on_side_effects=(
                self.require_approval_on_side_effects or other.require_approval_on_side_effects
            ),
        )



# Policy (top-level declaration)


class Policy(BaseModel):
    """
    Complete policy declaration for a skill / agent run.

    Composed by PolicyResolver; enforced by BudgetEnforcer and ToolGate.
    """

    budget: BudgetPolicy = Field(default_factory=BudgetPolicy)
    allowed_tools: set[str] | None = Field(
        default=None,
        description=(
            "Allowlist of tool names available in this context. "
            "None means 'all registered tools are allowed' (only restrict at ToolGate level). "
            "Empty set means 'no tools allowed'."
        ),
    )
    approval_rules: ApprovalRules = Field(default_factory=ApprovalRules)
    requires_confirmation: bool = Field(
        default=False,
        description="If True, user must confirm before the skill/agent starts executing.",
    )

    def merge_restrictive(self, other: Policy) -> Policy:
        """Return the most restrictive composition of self and other."""

        # allowed_tools: intersection (None = unbounded → treat as full set)
        if self.allowed_tools is None and other.allowed_tools is None:
            merged_tools: set[str] | None = None
        elif self.allowed_tools is None:
            merged_tools = other.allowed_tools
        elif other.allowed_tools is None:
            merged_tools = self.allowed_tools
        else:
            merged_tools = self.allowed_tools & other.allowed_tools

        return Policy(
            budget=self.budget.merge_restrictive(other.budget),
            allowed_tools=merged_tools,
            approval_rules=self.approval_rules.merge_restrictive(other.approval_rules),
            requires_confirmation=self.requires_confirmation or other.requires_confirmation,
        )

    def is_tool_allowed(self, tool_name: str) -> bool:
        """Return True if the tool is permitted under this policy."""
        if self.allowed_tools is None:
            return True
        return tool_name in self.allowed_tools

    @model_validator(mode="after")
    def _validate_budget_tokens_positive(self) -> Policy:
        b = self.budget
        for field, value in [
            ("max_cost_tokens", b.max_cost_tokens),
            ("max_iterations", b.max_iterations),
            ("max_tool_calls", b.max_tool_calls),
            ("max_duration_s", b.max_duration_s),
        ]:
            if value is not None and value <= 0:
                raise ValueError(f"budget.{field} must be > 0, got {value}")
        return self



# PolicyResolver

# Layer precedence order (lower index = lower priority)
_LAYER_NAMES = ("defaults", "skill", "agent", "request")


class PolicyResolver:
    """
    Resolves the effective policy by composing up to four layers,
    most-restrictive-wins at every field.

    Usage:
        effective = PolicyResolver.resolve(
            defaults=system_defaults,
            skill=skill_policy,       # from YAML
            agent=agent_policy,       # from agent definition
            request=request_policy,   # from API request (optional)
        )
    """

    @staticmethod
    def resolve(
        *,
        defaults: Policy | None = None,
        skill: Policy | None = None,
        agent: Policy | None = None,
        request: Policy | None = None,
    ) -> Policy:
        """
        Compose layers from least to most specific, applying most-restrictive-wins.
        None layers are skipped.
        """
        layers: list[Policy] = [
            p for p in (defaults, skill, agent, request) if p is not None
        ]

        if not layers:
            return Policy()

        result = layers[0]
        for layer in layers[1:]:
            result = result.merge_restrictive(layer)
        return result

    @staticmethod
    def from_config(raw: dict[str, Any]) -> Policy:
        """Parse a policy from a dict (e.g., loaded from YAML skill definition)."""
        return Policy.model_validate(raw)



# Default system policy (loaded from Settings in practice)

def default_policy() -> Policy:
    """
    System-wide default policy.
    In production, these values come from Settings / env.
    Imported lazily to avoid circular imports with config.py.
    """
    from app.core.config import get_settings

    s = get_settings()
    return Policy(
        budget=BudgetPolicy(
            max_cost_tokens=s.agent.budget_tokens,
            max_iterations=s.agent.max_iterations,
            max_tool_calls=s.agent.max_tool_calls,
        ),
    )
