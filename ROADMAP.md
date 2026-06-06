# ROADMAP.md

## Phase 1: MVP (Minimum Viable Product) - *The Micro-Accelerator*
- [ ] Set up Gradio UI with basic Chat-to-Code layout.
- [ ] Integrate DeepSeek-V4-Flash API.
- [ ] Implement **"Cell-Level Intent Interpreter"**: natural language input produces a single cell's worth of code, accompanied by a short explanation.
- [ ] Implement **"Error Diagnosis Playground"**: paste a full error log, get a layered explanation (plain language, technical root cause, and optional fix candidate).

## Phase 2: Beta - *Deep Insight & Comparison*
- [ ] **RAG Implementation**: Populate the vector database with official MindSpore 2.x documentation.
- [ ] Implement **"Framework Lens"**: interactive side-by-side comparison for code snippets (PyTorch vs MindSpore), highlighting syntactic and semantic differences.
- [ ] Integrate **"Execution Trace" visualization**: track a variable's history and type evolution across cells.

## Phase 3: Advanced - *Ecosystem & Education*
- [ ] **Cloud Brain Observer**: Read-only monitoring of ongoing tasks for telemetry correlation, without altering running jobs.
- [ ] **"Learn Mode" (Socratic Toggling)**: Force the agent to guide the user through a problem instead of giving the answer.
- [ ] **Asset Gallery**: Manage generated single-cell snippets and diagnostic reports.

## Phase 4: Final Polishing & Deployment
- [ ] Comprehensive README and video demo for AgentWeaver submission.
- [ ] Deployment on OpenI's online inference task area.
