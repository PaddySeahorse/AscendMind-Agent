# AscendMind Agent: AI-First Research & Development Workspace

> **Empowering the Ascend & MindSpore Ecosystem with Intelligent Autonomy.**

Start with a question, an error, or a blurry idea. The Agent accelerates every micro-decision by handling the syntax you shouldn't need to memorize, while keeping you in full control of the architecture.

This is not a project generator. It's a debugging-first, micro-task accelerator for developers who want to understand their code, not outsource it.

## 🌟 Key Features

- **Cell-Level Code Accelerator**: Describe the logic of a single cell in natural language, and get the precise MindSpore/CANN syntax. You review, you decide, you run.
- **Ascend Error Physician**: Automated, explainable, and verifiable diagnosis of complex CANN and NPU-specific errors with one-click fix suggestions.
- **Execution Trace Inspector**: Not just errors. Track how a variable's value and type evolve across cells, so you can catch silent bugs before they cascade.
- **Zero-Code Research Visualizer**: Turn raw training logs into publication-ready, interactive performance insights.
- **NPU Control Center**: Direct integration with OpenI Cloud Brain resources for light monitoring of tasks without altering code logic.
- **Socratic "Learn Mode" (Optional Toggle)**: Designed for students and explorers. When enabled, the Agent guides you through problems with pedagogical questioning instead of direct fixes, helping you build a deeper mental model of the Ascend/MindSpore stack.

## 🚀 Quick Start (Development)

```bash
# Clone the repository
git clone https://openi.pcl.ac.cn/your-username/AscendMind-Agent.git
cd AscendMind-Agent

# Install dependencies
pip install -r requirements.txt

# Run the Gradio App
python app.py
```

## 🛠 Tech Stack

- **Framework**: Gradio (Web UI)
- **Model**: DeepSeek-V4-Flash (via OpenI API)
- **Compute**: Huawei Ascend 910B / 310P
- **Software**: MindSpore, CANN
- **Intelligence**: RAG with Ascend/MindSpore documentation
