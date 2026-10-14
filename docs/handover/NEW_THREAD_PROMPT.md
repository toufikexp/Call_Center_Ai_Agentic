# Prompt for the new deployment thread

Paste everything below the line as the **first message** of the new Claude
Code session. It is self-contained; the full verified reference is
`docs/handover/PROJECT_CONTEXT.md` in this repository.

---

You are joining an existing project to deploy it into a new two-environment
production. Before proposing anything, build an accurate picture from the
**code**.

## The project

**Call Center AI Agentic Pipeline** (Python, LangGraph). It turns Ooredoo
Algeria customer-care call recordings (Algerian Darija/Arabic with French)
into structured analytics:

channel split + Silero VAD → Whisper large-v3 + LoRA transcription → Gemini
refinement → quality gate → subject classification and customer satisfaction
via a vLLM-served Qwen3-4B → JSON files + PostgreSQL.

It runs as a batch runner (`python -m src.batch run`) or a FastAPI job server
(`python -m src.server`), packaged with Docker Compose (`Makefile`,
`compose.yaml`, `compose.vllm.yaml`).

- Repository: `https://github.com/toufikexp/Call_Center_Ai_Agentic`, branch
  `main` at `ae86396`.
- Verified reference: `docs/handover/PROJECT_CONTEXT.md` (on branch
  `claude/add-project-docs-9sBU6` until merged). **Read it first.** It tags
  every claim with its source: code line, executed proof, observed in an
  environment, stated by the owner, or unknown.

## Environments

- **Today:** dev on WSL (GPU-capable); production on one RHEL VM, CPU only,
  Docker + Compose, no sudo/root, restricted internet.
- **Target:** production split across two environments:
  1. **VM (CPU)** — processing/calculation, PostgreSQL, backend. Docker and
     Python packages available; no sudo/root.
  2. **AI Fabric (GPU)** — only the AI parts that need a GPU. Kubernetes +
     JupyterHub: you start a Jupyter session, which is a pod with a predefined
     profile (resources).

## Your mission

1. Understand the current system from the code.
2. Get the facts about AI Fabric and the VM by asking me — the open questions
   are in `PROJECT_CONTEXT.md` §15. Do not assume any of them.
3. Propose a deployment architecture for the split, with trade-offs, and wait
   for my decision.
4. Implement it step by step, verifying each step before the next.

## Rules

1. **Code and real command output are the source of truth.** `CLAUDE.md` and
   `docs/` are partly outdated (`PROJECT_CONTEXT.md` §14 lists the known
   errors). When a document and the code disagree, the code wins.
2. **Never invent or guess** infrastructure facts: network paths, GPU types,
   driver/CUDA versions, ports, storage, permissions, what the cluster allows.
   If you need one, ask me. Ask only when the answer changes what you do.
3. **No changes without my explicit approval** — code, config, or anything on
   a server. Check deeply first; show the exact change and its impact before
   applying it.
4. **Never break working behavior.** For every change, state what it touches,
   what could break, and how to verify and roll it back.
5. **No secrets** (API keys, database passwords, tokens) in chat, commits or
   docs. `.env` is never committed.
6. When I ask for commands: the minimum set, copy-pasteable, and say which
   environment each one is for (dev / VM / AI Fabric). Prefer `make` targets
   where they exist.
7. Answer exactly what I ask, briefly unless I ask for detail. If you were
   wrong, correct it in one line and continue — no apologies.

## Verified facts you must not miss

Details and line references are in `PROJECT_CONTEXT.md`.

- **Whisper runs in-process**; preprocessing passes it in-memory numpy
  segments. Only vLLM, Gemini and PostgreSQL are already behind a network
  boundary — moving vLLM is configuration; moving Whisper alone is new code.
- **`model.generate()` is serialized per process** — scale GPU throughput with
  more processes/pods, not threads.
- **A transcription or refinement failure ends as `MANUAL_REVIEW`, not
  `ERROR`** (proven by executing the graph), and batch mode then archives the
  file. A remote GPU outage would silently move files out of the input folder.
- **`make build TORCH=gpu` is broken on `main`**: `Makefile:21` maps `gpu` →
  `cu130`, but the file is `requirements/torch-gpu.txt`.
- **GPU wheels are pinned to `torch==2.9.1+cu130`** — GPU nodes need a driver
  supporting CUDA 13.0.
- **Outside Docker, the repo's `.env` parsers keep inline comments** as part of
  the value (`.env.example` has four such lines, including `GEMINI_API_KEY`).
- The API server has **no authentication**, keeps jobs **in memory**, and
  `/ready` checks nothing.
- `--served-model-name` is **hard-coded** in `compose.vllm.yaml`; it must equal
  `VLLM_MODEL_NAME`.
- Output and log directories are **relative to the working directory**
  (`data/results`, `data/logs`) and not env-configurable.
- The repo has **no tests, no CI, no Kubernetes/JupyterHub artifacts** — that
  part starts from zero.

## First steps

1. Read `docs/handover/PROJECT_CONTEXT.md`, then the code it references:
   `src/pipeline/orchestrator.py`, `src/services/*`, `src/config/config.py`,
   `src/batch/runner.py`, `src/server/*`, `Dockerfile`, `compose*.yaml`,
   `Makefile`.
2. Tell me in at most 10 lines what you understood and what you need from me.
3. Ask the open questions you need for the AI Fabric design — grouped, in one
   message.
