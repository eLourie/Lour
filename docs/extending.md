# Extending Lour

Extensibility is a first-class requirement (PROJECT_CONTEXT §10.8): every layer
has a clear interface, and the common extensions are **declarative**. This guide
is the contract — the five points where you plug in without touching the core.

| Extend a… | Where | Contract |
|-----------|-------|----------|
| Tool | `app/tools/builtins/` | subclass `BaseTool`, decorate `@tool` |
| Skill | `app/skills/definitions/` | drop a YAML file (+ optional Python override) |
| LLM provider | `app/services/llm/` | implement the `LLMProvider` Protocol, wire into the factory |
| Tool source / exposure | MCP | connect an MCP server as a tool source; expose yours via the MCP server |
| Hardware / topology | `.env` | change variables only — no code |

---

## 1. Add a Tool

A tool is an atomic, stateless function the LLM can call. Create a class in
`app/tools/builtins/`, give it typed args, and decorate it — the `@tool`
decorator registers it for discovery (it validates the metadata but does *not*
instantiate, so tools that need runtime dependencies stay lazy).

```python
# app/tools/builtins/word_count.py
from pydantic import BaseModel, Field

from app.tools.base import BaseTool, ToolResult
from app.tools.registry import tool


class WordCountArgs(BaseModel):
    text: str = Field(description="The text to count words in.")


@tool
class WordCount(BaseTool[WordCountArgs]):
    name = "word_count"                    # verb_object, ≤ 30 chars
    description = "Count the words in a piece of text. Use for length checks."
    args_schema = WordCountArgs

    async def execute(self, args: WordCountArgs) -> ToolResult:
        return ToolResult(ok=True, data={"words": len(args.text.split())})
```

Rules the decorator and design enforce: `name` is `verb_object` and ≤ 30 chars;
`description` ≤ 1024 chars, starts with a verb, and says when *not* to use it;
args are strictly typed; `execute` always returns a `ToolResult{ok, data, error,
metadata}`; tools never touch the LLM or agent state. Grant the tool to an agent
or skill by adding its `name` to that skill's `tools_allowed` — the **ToolGate**
enforces the allowlist at call time.

---

## 2. Add a Skill

A skill is a high-level, user-facing scenario. In the common case it is *just a
YAML file* in `app/skills/definitions/` — `SkillRegistry` discovers it on
startup and it appears in `GET /v1/skills`.

```yaml
# app/skills/definitions/explain_concept.yaml
name: explain_concept
description: >-
  Explain a technical concept clearly, grounding it in the knowledge base.
agent: researcher            # the skill declares its agent → routing happens once
tools_allowed:
  - rag_query

input_schema:
  concept:
    type: str
    required: true
    description: The concept to explain.

output_schema:
  explanation: str

prompt: >-
  Explain the concept "{concept}" clearly and concisely, using rag_query to
  ground it in the knowledge base. Cite what you use.

# Per-skill policy (§5.4). allowed_tools is derived from tools_allowed above.
budget:
  max_cost_tokens: 20000
  max_iterations: 4
requires_confirmation: false
```

Need custom pre/post-processing? Add an optional override in
`app/skills/implementations/<name>.py` (see `review_code.py` for the pattern) —
the registry binds it to the YAML declaration automatically.

---

## 3. Add an LLM provider

Providers sit behind the `LLMProvider` Protocol (`app/services/llm/base.py`), so
the agent loop is provider-agnostic (ADR-002). Implement the Protocol
(`chat` / `stream` / `complete_structured`) and wire it into the factory.

```python
# app/services/llm/factory.py — add a branch
match llm_settings.provider:
    case LLMProviderEnum.OLLAMA:
        ...
    case LLMProviderEnum.VLLM:              # your new tier
        provider = VLLMProvider(llm_settings)
        assert isinstance(provider, LLMProvider)
        return provider
```

Select it at runtime with `LLM_PROVIDER=vllm` in `.env` — nothing else changes.
Ollama is the local dev/test tier; a cloud provider is the reliability tier for
long tool chains. Both satisfy the same Protocol.

---

## 4. MCP — consume and expose tools

MCP is bidirectional (ADR-006).

- **Consume** external tools: point the MCP **client** (`app/tools/mcp/client.py`)
  at a server; the adapter surfaces its tools through the local `ToolRegistry`,
  so they behave like any builtin (allowlist and ToolGate still apply).
- **Expose** your builtins: run the MCP **server** (`app/tools/mcp/server.py`)
  and connect it from Claude Desktop or Cursor — your tools show up there.

---

## 5. Hardware & topology — config only

No code changes to run on different hardware or move backing services around.
Pick the main model, reranker mode and deployment profile in `.env`:

```bash
LLM_MAIN_MODEL=qwen3:7b        # smaller box → smaller model (see §3.3 table)
RERANKER_MODE=none             # local | cloud | none
DEPLOY_PROFILE=offloaded       # solo | split | offloaded
```

The reference in `.env.example` is an M4 24 GB (Split, `qwen3:14b`). The
hardware-to-model table in PROJECT_CONTEXT §3.3 suggests values for other boxes.

---

## Testing your extension

- Unit-test the pure logic (a tool's `execute`, a skill override) — see
  `tests/unit/`.
- Add it to the relevant eval dataset under `tests/eval/datasets/` (e.g. a new
  skill → `skill_routing.jsonl`) so `make eval` covers it.
- Keep `make lint` (ruff + `mypy --strict`) green — it is a hard gate.
