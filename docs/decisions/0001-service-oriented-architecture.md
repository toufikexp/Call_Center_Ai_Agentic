# ADR 0001 — Service-oriented architecture (no agents/tools)

- Status: Accepted
- Context: Initial implementation

## Context

LangGraph and LangChain make it easy to model a pipeline as autonomous agents
with tool calls. For a deterministic call-analytics pipeline with five fixed
stages, that pattern adds runtime indirection (function-calling on each step),
makes failure handling harder, and obscures data flow.

## Decision

We model each stage as a plain Python class — a "service" — that inherits
from `BaseService` (`src/core/base.py`). Orchestration is a LangGraph
`StateGraph` whose nodes call `service.process(...)` directly. There are no
LLM-driven routing decisions; routing is hard-coded in
`_route_decision` / `_check_error`.

## Consequences

- **Pro:** stages are unit-testable in isolation; the graph is small enough to
  reason about visually; behavior is deterministic given the same input.
- **Pro:** services can be reused outside the pipeline (e.g. backfill
  scripts) without re-creating the graph.
- **Pro:** no tool-call overhead per step.
- **Con:** we don't get LangChain's built-in retries / fallbacks. We rebuild
  what we need (`_execute_with_timing`).
- **Con:** adding a new branch requires editing `orchestrator.py`; we don't
  get "self-healing" behavior from an agent.

## Alternatives considered

- **LangChain agent with tools** — rejected; non-determinism, opacity.
- **Plain function pipeline (no LangGraph)** — viable, but LangGraph gives
  us conditional edges and a clear graph definition that we expect to grow.
