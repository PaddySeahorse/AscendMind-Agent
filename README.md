# AscendMind Agent: AI-First Research & Development Workspace

> **Empowering the Ascend & MindSpore Ecosystem with Intelligent Autonomy.**

AscendMind Agent is an AI-first, integrated workspace designed specifically for the **OpenI (启智社区)** ecosystem. Unlike traditional notebooks, it centers the developer experience around a conversational AI agent that possesses deep knowledge of **Huawei Ascend NPU** hardware and the **MindSpore** framework.

## 🌟 Key Features

- **AI-First Workflow**: No more blank cells. Start with an intent, and let the agent scaffold your MindSpore project.
- **Ascend Error Physician**: Automated diagnosis of complex CANN and NPU-specific errors with one-click fix suggestions.
- **MindSpore Code Transformer**: Seamlessly migrate PyTorch/TensorFlow code to MindSpore with NPU-optimized operators.
- **Zero-Code Research Visualizer**: Turn raw training logs into publication-ready, interactive performance insights.
- **NPU Control Center**: Direct integration with OpenI Cloud Brain resources to monitor and execute tasks.

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
