# ARCHITECTURE.md

## 🏗 System Design

AscendMind Agent follows a decoupled architecture separating the UI, the Intelligence Layer, and the Execution Environment.

### 1. Presentation Layer (Gradio)
- **Chat Interface**: Handles multi-turn dialogue.
- **Code Canvas**: Real-time code display with Monaco/Prism styling.
- **NPU Dashboard**: Visualizes resource telemetry using Plotly.

### 2. Intelligence Layer (The Agent)
- **Core LLM**: DeepSeek-V4-Flash (provided by AgentWeaver).
- **RAG Engine**:
    - **Vector DB**: ChromaDB/Faiss storing MindSpore & Ascend docs.
    - **Knowledge Base**: Scraped from MindSpore Gitee, Ascend Community, and OpenI tutorials.
- **Agentic Tools**:
    - `CodeAnalyzer`: Validates syntax.
    - `LogParser`: Extracts error patterns.
    - `Migrator`: Logic for framework conversion.

### 3. Execution Layer (OpenI Cloud Brain)
- **Backend**: Python 3.9+ environment.
- **Hardware Interface**: Connectivity to Ascend 910B NPU for light validation or execution status.

## 🔄 Data Flow
1. **User Input** (Natural Language) -> **Gradio**.
2. **Gradio** -> **Agent** (LLM + RAG).
3. **Agent** -> **Action** (Generate Code / Search Docs / Diagnose Error).
4. **Action Output** -> **Code Canvas** & **Chatbot Response**.
5. (Optional) **Execute** -> **Cloud Brain** -> **Real-time Logs** -> **NPU Dashboard**.
