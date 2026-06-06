# ARCHITECTURE.md

## 🏗 System Design

AscendMind Agent follows a decoupled architecture separating the UI, the Intelligence Layer, and the Execution Environment.

### 1. Presentation Layer (Gradio)
- **Chat Interface**: Handles multi-turn dialogue for intents and diagnosis.
- **Code Canvas**: Real-time code display for single-cell units with Monaco/Prism styling.
- **NPU Dashboard**: Visualizes resource telemetry and execution traces using Plotly.

### 2. Intelligence Layer (The Agent)
- **Core LLM**: DeepSeek-V4-Flash (provided by AgentWeaver).
- **RAG Engine**:
    - **Vector DB**: ChromaDB/Faiss storing MindSpore & Ascend docs.
    - **Knowledge Base**: Scraped from MindSpore Gitee, Ascend Community, and OpenI tutorials.
- **Agentic Tools**:
    - `SyntaxAccelerator`: Generates syntax for a single, well-scoped intent described by the user. No multi-file scaffolding.
    - `DiffExplainer`: Compares two code snippets (e.g., PyTorch vs. MindSpore) and explains semantic and performance differences without emitting production-ready code.
    - `LogParser`: Extracts error patterns from complex CANN/NPU logs.

### 3. Execution Layer (OpenI Cloud Brain)
- **Backend**: Python 3.9+ environment.
- **Hardware Interface**: Connectivity to Ascend 910B NPU for execution telemetry.

## 🔄 Data Flow
1. **User Input** (Natural Language Intent / Error Log) -> **Gradio**.
2. **Gradio** -> **Agent** (LLM + RAG).
3. **Agent** -> **Action** (Generate single-cell code / Explain error with RAG / Compare snippets / Search docs). Every code-generating action is scoped to a single user-reviewed unit.
4. **Action Output** -> **Code Canvas** & **Chatbot Response**.
5. (Optional) **Telemetry Correlation** -> **Cloud Brain Observer** -> **NPU Dashboard**.
