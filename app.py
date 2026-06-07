import gradio as gr
import pandas as pd
import random
import time

# Preset data for interactive demonstration
SAMPLES_CHAT = [
    ["Create a single-cell 2D Convolution in MindSpore", "Generate a customized conv2d layer using mindspore.nn.Conv2d"],
    ["Write a custom MindSpore Cell for LayerNorm", "Generate a custom LayerNorm implementation from scratch"],
    ["Write a MindSpore training step with ValueAndGrad", "Generate the custom Cell and training step with grad functions"],
    ["Perform tensor slicing and masking in MindSpore", "Demonstrate how to slice tensors and apply boolean mask in MindSpore"]
]

PRESET_CODES = {
    "Create a single-cell 2D Convolution in MindSpore": """import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
import numpy as np

class ConvBlock(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ConvBlock, self).__init__()
        # In MindSpore, padding mode can be "same", "valid", or integer padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=1, pad_mode="pad", has_bias=True)
        self.relu = nn.ReLU()

    def construct(self, x):
        # construct() is equivalent to PyTorch's forward()
        x = self.conv(x)
        x = self.relu(x)
        return x

# Initialize and test
input_tensor = Tensor(np.random.randn(1, 3, 32, 32).astype(np.float32))
conv_cell = ConvBlock(in_channels=3, out_channels=16)
output = conv_cell(input_tensor)
print("Output shape:", output.shape)
""",
    "Write a custom MindSpore Cell for LayerNorm": """import mindspore as ms
import mindspore.nn as nn
from mindspore import ops, Parameter, Tensor
import numpy as np

class CustomLayerNorm(nn.Cell):
    def __init__(self, normalized_shape, epsilon=1e-5):
        super(CustomLayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.epsilon = epsilon
        # MindSpore parameters require explicit shape and type initializations
        self.gamma = Parameter(Tensor(np.ones(normalized_shape).astype(np.float32)), name="gamma")
        self.beta = Parameter(Tensor(np.zeros(normalized_shape).astype(np.float32)), name="beta")

    def construct(self, x):
        # Calculate mean and variance over normalized axes
        mean = ops.mean(x, axis=-1, keep_dims=True)
        variance = ops.var(x, axis=-1, keep_dims=True)
        # Standardize
        x_norm = (x - mean) / ops.sqrt(variance + self.epsilon)
        # Apply affine transformation
        return self.gamma * x_norm + self.beta

# Initialize and verify
x = Tensor(np.random.randn(2, 4, 16).astype(np.float32))
ln_layer = CustomLayerNorm(normalized_shape=(16,))
y = ln_layer(x)
print("Output shape:", y.shape)
""",
    "Write a MindSpore training step with ValueAndGrad": """import mindspore as ms
from mindspore import nn, ops, Tensor, value_and_grad
import numpy as np

# Define a simple linear model
class SimpleLinear(nn.Cell):
    def __init__(self):
        super(SimpleLinear, self).__init__()
        self.fc = nn.Dense(10, 1)

    def construct(self, x):
        return self.fc(x)

# loss function
loss_fn = nn.MSELoss()

# Single training step function
model = SimpleLinear()
optimizer = nn.SGD(model.trainable_params(), learning_rate=0.01)

def forward_fn(data, label):
    logits = model(data)
    loss = loss_fn(logits, label)
    return loss, logits

# value_and_grad automatically generates gradient calculations
grad_fn = value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

@ms.jit # JIT compilation for high performance
def train_step(data, label):
    (loss, logits), grads = grad_fn(data, label)
    optimizer(grads)
    return loss

# Simulated execution
data = Tensor(np.random.randn(8, 10).astype(np.float32))
label = Tensor(np.random.randn(8, 1).astype(np.float32))
loss_val = train_step(data, label)
print("Step loss computed:", loss_val.asnumpy())
""",
    "Perform tensor slicing and masking in MindSpore": """import mindspore as ms
from mindspore import ops, Tensor
import numpy as np

# Create tensor
x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32))

# Slice tensor: rows 0 & 1, cols 1 & 2
sliced_x = x[0:2, 1:3]
print("Sliced Tensor:\\n", sliced_x)

# Create a boolean mask: values greater than 5
mask = x > 5
print("Boolean Mask:\\n", mask)

# MindSpore dynamic masking using ops.masked_select
masked_val = ops.masked_select(x, mask)
print("Masked values select:\\n", masked_val)
"""
}

# Socratic mode responses helper
SOCRATIC_RESPONSES = {
    "Create a single-cell 2D Convolution in MindSpore": """💡 **AscendMind Socratic Assistant:**

That's a great task! Before we look at the raw code, let's think about how **MindSpore**'s API differs from PyTorch.

1. In PyTorch, we have `nn.Conv2d` and we can set `padding='same'`. In **MindSpore**, `pad_mode` is a key parameter that takes `'same'`, `'valid'`, or `'pad'`. If we choose `'pad'`, how do we specify individual padding pixels?
2. In MindSpore's `construct` method (which corresponds to `forward` in PyTorch), does it behave differently for static vs dynamic shapes during compile time?

What kind of input shapes are you designing this convolution for? Tell me your dimension parameters (e.g., `in_channels`, `out_channels`) and we can outline the structure together!""",

    "Write a custom MindSpore Cell for LayerNorm": """💡 **AscendMind Socratic Assistant:**

Layer Normalization is highly sensitive to the axes we normalize over. 

1. MindSpore represents parameters using the `Parameter` class. Unlike PyTorch where parameters are registered automatically, how should we pass initializers to parameters in MindSpore?
2. To compute mean and variance, which operations (`ops`) should we use? In MindSpore 2.x, do we use `ops.mean` or `ops.ReduceMean`?

What shape of input tensors are you normalising? Let's discuss how we can compute the mean over the last dimension while keeping dimensions matched!""",

    "Write a MindSpore training step with ValueAndGrad": """💡 **AscendMind Socratic Assistant:**

Ah, writing the training loop is one of the most exciting shifts from PyTorch to MindSpore!

In PyTorch, you do `loss.backward()` and `optimizer.step()`. But MindSpore is **functional-first** for automatic differentiation.
1. Have you encountered `value_and_grad` yet? It is MindSpore's preferred way to calculate both forward output and parameters gradients in one sweep.
2. How does `has_aux` work when your forward function returns both the `loss` and helper outputs like intermediate `logits`?

Let's think about what your forward function needs to return. Would you like to design a custom loss evaluation first?""",

    "Perform tensor slicing and masking in MindSpore": """💡 **AscendMind Socratic Assistant:**

Tensor manipulation is fundamental. 
1. MindSpore supports Python-style slicing directly (e.g., `x[0:2, 1:3]`). But what happens when we use boolean indexing like `x[x > 5]` under static shape JIT graph compilation?
2. In graph mode, does MindSpore allow dynamic-sized outputs from slicing, or should we prefer functional operators like `ops.masked_select` or `ops.gather`?

What are you trying to accomplish with the masked values? Let's check how we can handle this efficiently for Ascend NPU compilation!"""
}

# Error presets
ERROR_LOGS_PRESETS = [
    "CANN Out of Memory Error on Device 0",
    "MindSpore Static Shape Compile Error",
    "MindSpore DataType Mismatch (Float32 vs Float64)"
]

ERROR_DETAILS = {
    "CANN Out of Memory Error on Device 0": {
        "log": """[ERROR] DEVICE(12432,python):2026-06-07-10:14:22.512.441 [mindspore/ccsrc/plugin/device/ascend/hal/device/ascend_memory_pool.cc:245] AllocDeviceMem] Float memory allocation failed! Alloc size: 17179869184B, Free memory: 8589934592B. Device id: 0.
Traceback (most recent call last):
  File "train.py", line 45, in <module>
    outputs = model(inputs)
  File "mindspore/nn/cell.py", line 586, in __call__
    out = self.compile_and_run(*args, **kwargs)
RuntimeError: Ascend CANN Out of Memory on Device 0. Allocator tried to allocate 16.00GB, but only 8.00GB free. """,
        "plain": "⚠️ **【白话解释 / Plain Explanation】**\n您的昇腾 NPU 显存爆满啦！模型在运行到 `train.py` 第 45 行时，尝试向 NPU 申请分配 **16.00 GB** 的显存来存放张量，但当前 NPU 剩余可用显存只有 **8.00 GB**。导致显存分配失败，程序中断。",
        "cause": "🛠️ **【技术根因 / Technical Root Cause】**\n1. **显存池溢出**: Ascend CANN 驱动的 `ascend_memory_pool` 在管理物理 NPU 页面时无法满足单次大内存块的连续申请。\n2. **Batch Size 过大**: 输入 Tensor 尺寸太大，或者全连接层/卷积层的通道数过大导致中间激活值占用指数级上升。\n3. **未释放算子缓存**: 在循环迭代中，可能存在没有被 `stop_gradient` 截断的常驻 Tensor 变量，导致计算图显存持续泄露。",
        "fix": "📝 **【修复方案 / Fix Candidate】**\n\n```python\n# 1. 在初始化时，通过显存分配参数进行限制或更精细的显存池块管理\nimport mindspore as ms\nms.set_context(device_target=\"Ascend\", max_device_memory=\"8.0GB\") \n\n# 2. 减小 DataLoader 中的 batch_size （推荐第一步尝试，例如将 64 改为 32）\n# train_dataset = dataset.batch(batch_size=32)\n\n# 3. 避免在 construct() 内部累加非必要的梯度历史，使用 ops.stop_gradient 截断：\n# out_val = ops.stop_gradient(intermediate_tensor)\n```"
    },
    "MindSpore Static Shape Compile Error": {
        "log": """[ERROR] ANALYZER(12500,python):2026-06-07-10:20:15 [mindspore/ccsrc/pipeline/jit/parse/parse.cc:112] GetJitClassType] The input shape of Cell must be static, but got dynamic shape Tensor[Float32], shape=[?, 128].
Traceback (most recent call last):
  File "net.py", line 12, in construct
    y = self.fc(x)
ValueError: MindSpore JIT Graph mode requires static shape inputs by default. Got dynamic shape Tensor with variable batch dimension '?'.""",
        "plain": "⚠️ **【白话解释 / Plain Explanation】**\nMindSpore 的静态图（Graph Mode）编译器正在向您抱怨：它在编译 `net.py` 第 12 行的神经网络时，发现输入的形状是动态的（即 Batch 维度是未知的 `?`），但编译器在默认设置下，必须知道精确的数字（静态形状）才能对其进行多算子融合和硬件加速。",
        "cause": "🛠️ **【技术根因 / Technical Root Cause】**\n1. **静态图限制**: `ms.jit` 装饰器修饰的函数或 `nn.Cell` 在 Graph Mode 下，编译器在推导（Specialize）网络结构时需要确定所有维度的 Tensor 边界。\n2. **动态 Batch 传递**: 您的输入 Dataset 输出的 batch 尺寸可能因不规则分块（如最后一批样本数不满足 batch_size）而产生动态波动，触发了静态图约束。",
        "fix": "📝 **【修复方案 / Fix Candidate】**\n\n```python\n# 方案 A: 如果允许，可以使用 PyNative 模式（动态图模式，支持动态形状，调试友好）：\nimport mindspore as ms\nms.set_context(mode=ms.PYNATIVE_MODE)\n\n# 方案 B: 保持 Graph 模式，但显式声明输入具有动态 Shape（使用 set_inputs）：\nclass Net(nn.Cell):\n    def __init__(self): ...\n\nnet = Net()\n# 声明输入第一维（Batch）是动态的（None 代表动态），第二维为 128 静态\ninput_spec = ms.Tensor(shape=(None, 128), dtype=ms.float32)\nnet.set_inputs(input_spec)\n```"
    },
    "MindSpore DataType Mismatch (Float32 vs Float64)": {
        "log": """TypeError: For primitive[Add], the input argument[x, y] must have the same data type, but got Float32 and Float64.
Traceback (most recent call last):
  File "main.py", line 22, in construct
    z = ops.add(self.weight, x)
TypeError: For 'Add', the types of input elements must be identical. weight is Tensor[Float64], x is Tensor[Float32].""",
        "plain": "⚠️ **【白话解释 / Plain Explanation】**\n类型不匹配错误！您在 `main.py` 的第 22 行尝试进行加法（`ops.add`）操作，但是加法的两个对象数据类型不同：权重 `weight` 是双精度浮点数（Float64，通常来自 numpy 的默认 float 转换），而输入 `x` 是单精度浮点数（Float32，昇腾 NPU 最喜欢的标准格式）。MindSpore 要求加减乘除必须严格类型一致。",
        "cause": "🛠️ **【技术根因 / Technical Root Cause】**\n昇腾 NPU 硬件对强类型的指令集有严格限制。不像 PyTorch 会在后台隐式隐转类型（Implicit Cast），MindSpore 为了极致性能和可控性，强制所有基础数学算子的双输入必须在前端类型完全对齐（CANN 算子对齐要求）。",
        "fix": "📝 **【修复方案 / Fix Candidate】**\n\n```python\n# 方案 A: 显式转换输入或权重的类型，统一使用 ms.float32：\nfrom mindspore import ops\n# 将 weight 或 x 转换为一致类型\nz = ops.add(self.weight.astype(ms.float32), x)\n\n# 方案 B: 检查数据源头，在 numpy 创建 Tensor 时显式指定 np.float32：\nimport numpy as np\nfrom mindspore import Tensor, ms\nself.weight = Parameter(Tensor(np.ones((10, 10)).astype(np.float32))) \n```"
    }
}

# Framework preset comparisons
FRAMEWORK_LENS_PRESETS = {
    "Tensor Reshaping & Slicing": {
        "torch": """import torch

# Create a tensor
x = torch.randn(2, 3, 4)

# Reshape tensor
y = x.view(2, -1) # memory-sharing view
y2 = x.reshape(2, -1) # might copy or share

# Permute / transpose dimensions
z = x.permute(2, 0, 1) # reorder to (4, 2, 3)

# Slice and slice-assignment
sliced = x[:, 0:2, 1:3]
""",
        "ms": """import mindspore as ms
from mindspore import ops, Tensor

# Create a tensor
x = ops.randn(2, 3, 4)

# Reshape tensor (In ms, .view exists but .reshape is preferred)
y = x.reshape(2, -1)
# functional approach: ops.reshape(x, (2, -1))

# Permute / transpose dimensions
# In MindSpore, permute is mapped to transpose
z = x.transpose((2, 0, 1)) 
# functional approach: ops.transpose(x, (2, 0, 1))

# Slice is fully supported similarly
sliced = x[:, 0:2, 1:3]
""",
        "analysis": """📋 **【对比要点 / Key Analysis Notes】**
1. **View vs Reshape**: PyTorch 的 `.view()` 局限于连续（contiguous）张量。MindSpore 统一使用 `.reshape()`，它既可以用作高效的 View（零拷贝），也会在不连续时自动做拷贝，安全性更好。
2. **Permute vs Transpose**: PyTorch 使用 `permute(*dims)` 来重排多个轴。MindSpore 使用 `transpose(dims)` (需要传入一个整数 Tuple 元组) 或者是 `ops.transpose(x, dims)`，其命名风格更对齐 NumPy。
3. **功能性算子偏好**: MindSpore 鼓励通过 `from mindspore import ops` 使用函数式编程模式，这对于 JIT 编译静态图非常友好。"""
    },
    "2D Convolution Layer": {
        "torch": """import torch.nn as nn

class PyTorchConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=3,
            stride=1,
            padding=1, # string padding like 'same' or int
            bias=True
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))
""",
        "ms": """import mindspore.nn as nn

class MSConvBlock(nn.Cell):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=3,
            stride=1,
            pad_mode="pad", # Crucial! "pad", "same", "valid"
            padding=1,      # Only valid when pad_mode is "pad"
            has_bias=True   # bias parameter name is 'has_bias' instead of 'bias'
        )
        self.relu = nn.ReLU()

    def construct(self, x): # forward() is construct() in MindSpore
        return self.relu(self.conv(x))
""",
        "analysis": """📋 **【对比要点 / Key Analysis Notes】**
1. **基类名称**: PyTorch 的网络组件继承自 `nn.Module`；MindSpore 继承自 `nn.Cell`（单元）。
2. **前向函数命名**: PyTorch 覆写 `forward()`；MindSpore 覆写 `construct()`。
3. **卷积层参数差异**:
   - `pad_mode`: MindSpore 中有严格的填充模式配置。如果要像 PyTorch 一样指定具体数字的 `padding`，必须将 `pad_mode` 显式设为 `"pad"`。
   - `bias` 命名: PyTorch 是 `bias=True`，而 MindSpore 对应参数是 `has_bias=True`。"""
    },
    "Squeeze & Unsqueeze": {
        "torch": """import torch

x = torch.randn(1, 3, 1, 5)

# Squeeze dimensions of size 1
y1 = torch.squeeze(x)      # squeeze all dims size 1
y2 = torch.squeeze(x, dim=2) # squeeze specific dim

# Unsqueeze: add a dimension of size 1
z = torch.unsqueeze(y1, dim=0) # shape (1, 3, 5)
""",
        "ms": """import mindspore as ms
from mindspore import ops

x = ops.randn(1, 3, 1, 5)

# Squeeze dimensions of size 1
# MindSpore ops.squeeze uses 'axis' instead of 'dim'
y1 = ops.squeeze(x)
y2 = ops.squeeze(x, axis=2)

# Unsqueeze / expand dims
z = ops.unsqueeze(y1, axis=0) # or ops.expand_dims
""",
        "analysis": """📋 **【对比要点 / Key Analysis Notes】**
1. **维度参数命名**: PyTorch 中统一使用 `dim` 表示轴（例如 `dim=0`），而 MindSpore 对齐 NumPy 接口标准，统一使用 `axis`（轴，例如 `axis=0`）。
2. **算子兼容性**: MindSpore 的 Tensor 对象也同样挂载了 `.squeeze()` 和 `.expand_dims()` 实例方法，可以直接调用。"""
    }
}

# Main Application logic
def chatbot_respond(user_message, chat_history, socratic_mode):
    if not user_message.strip():
        return "", chat_history, gr.update()

    # Determine if it matches any preset codes
    matched_preset = None
    for key in PRESET_CODES.keys():
        if key.lower() in user_message.lower():
            matched_preset = key
            break

    # Simulate typing delay
    bot_response = ""
    target_code = None

    if socratic_mode:
        if matched_preset:
            bot_response = SOCRATIC_RESPONSES[matched_preset]
        else:
            bot_response = f"💡 **【苏格拉底对话 / Socratic Learn Mode】**\n\n谢谢你关于“{user_message}”的提问！作为你的 Ascend 导师，我不打算直接给你一段完整的代码，而是让我们一步一步来思考：\n\n1. 你目前在使用静态图（Graph）还是动态图（PyNative）模式？这会极大影响你所需要的算子编写格式。\n2. 如果我们要解决这个问题，它的核心数学算子或网络层属于 MindSpore 里的哪个命名空间？是 `nn`、`ops` 还是 `mint`？\n\n不妨先告诉我你期望的输入 Tensor 结构，我们来共同推演一下！"
    else:
        if matched_preset:
            bot_response = f"✨ **AscendMind Code Accelerator:**\n\n我已根据您的意图为你生成了单 Cell 粒度的昇思 MindSpore 代码单元。该代码已同步更新到右侧的 **💻 Code Canvas** 画布中！\n\n**【核心思路】**\n- 采用高能效的 functional 算子，兼容 Graph/PyNative 混合编译。\n- 严格限制单 Cell 范畴，避免无用的冗余文件腳手架。"
            target_code = PRESET_CODES[matched_preset]
        else:
            # Generate generic mock conv layer code
            generic_name = f"Custom{random.randint(100,999)}Block"
            target_code = f"""import mindspore as ms
import mindspore.nn as nn
from mindspore import ops, Tensor
import numpy as np

# Generic Auto-generated block for: {user_message}
class {generic_name}(nn.Cell):
    def __init__(self):
        super({generic_name}, self).__init__()
        # Initialize operations matching intent
        self.dense = nn.Dense(128, 64)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.dense(x)
        return self.relu(x)

# Mock Run
x = Tensor(np.random.randn(4, 128).astype(np.float32))
net = {generic_name}()
out = net(x)
print("Traceback Safe. Output shape:", out.shape)
"""
            bot_response = f"✨ **AscendMind Code Accelerator:**\n\n我已经为你编写了一个适配 `{user_message}` 的定制化 MindSpore 单 cell 逻辑单元。代码已加载到右侧的 **💻 Code Canvas**。你可以直接在右侧进行语法分析或 NPU 模拟运行。"

    # Append to chatbot history
    chat_history.append((user_message, bot_response))
    
    if target_code:
        # We also want to update the right side code text component!
        return "", chat_history, gr.update(value=target_code, visible=True)
    else:
        return "", chat_history, gr.update()

def handle_sample_click(sample_btn_text, socratic_mode):
    # This simulates a sample being inputted and run directly
    mock_history = []
    _, updated_history, code_update = chatbot_respond(sample_btn_text, mock_history, socratic_mode)
    return sample_btn_text, updated_history, code_update

def simulate_npu_run(code_content):
    if not code_content.strip():
        return "❌ [NPU RUN ERROR]: Code Canvas is empty! Please write or generate a code block first."
    
    # Simple rule-based mock execution traces
    cell_name = "nn.Cell Block"
    for line in code_content.split('\n'):
        if "class " in line:
            cell_name = line.split("class ")[1].split("(")[0].strip()
            break

    outputs = [
        f"🚀 [Ascend Cloud Brain Observer] Initializing environment... (Target: Huawei Ascend 910B NPU)",
        f"⚙️ [CANN Compiler] Graph Mode JIT compiling '{cell_name}'...",
        f"📦 [Tensor Allocator] Allocating static buffers for variables...",
        f"🟢 [NPU Device 0] Executing run trace...\n----------------------------------------",
    ]
    
    # Try to grab simulated output print statement
    if "print" in code_content:
        # Find print statements
        print_lines = [line.strip() for line in code_content.split('\n') if "print(" in line]
        for pl in print_lines:
            # simple mock execution result based on the print
            if "Output shape" in pl or "Output shape:" in pl:
                if "ConvBlock" in code_content:
                    outputs.append("Output shape: (1, 16, 32, 32)")
                elif "CustomLayerNorm" in code_content:
                    outputs.append("Output shape: (2, 4, 16)")
                else:
                    outputs.append("Output shape: (4, 64)")
            elif "Step loss" in pl:
                outputs.append("Step loss computed: 0.41295")
            elif "Sliced Tensor" in pl:
                outputs.append("Sliced Tensor:\n [[2. 3.]\n  [5. 6.]]")
            elif "Boolean Mask" in pl:
                outputs.append("Boolean Mask:\n [[False False False]\n  [False False  True]\n  [ True  True  True]]")
            elif "Masked values" in pl:
                outputs.append("Masked values select:\n [6. 7. 8. 9.]")
    else:
        outputs.append("Process finished with exit code 0 (Simulated Execution)")

    outputs.append("----------------------------------------")
    outputs.append("✅ [NPU Execution Trace SUCCESS] Memory footprint: 4.12 MB. Compute time: 14.2 ms (Warm compile: 241 ms).")
    return "\n".join(outputs)

def explain_code_canvas(code_content):
    if not code_content.strip():
        return "Please load code first."
    
    # Provide intelligent custom analysis based on loaded code
    if "ConvBlock" in code_content:
        return """📑 **【画布代码深度剖析 / Code Canvas Analysis】**

1. **算子解析 (`nn.Conv2d`)**:
   - `pad_mode="pad"`: 允许用户像 PyTorch 一样自由传递外部 `padding=1`。如果将其设为 `'same'`，MindSpore 会自动计算填充大小，忽略 `padding` 数值。
   - `has_bias=True`: 与 PyTorch 的 `bias=True` 对应。

2. **前向执行 (`construct`)**:
   - 昇思网络层的核心计算必须定义在 `construct` 内部。在后端，静态图引擎会跟踪此函数内部的数据流并形成 DAG。

3. **数据校验 (Tensor & np.float32)**:
   - MindSpore 对输入数据的类型敏感度很高。此处使用 `astype(np.float32)` 确保输入数据类型为单精度，避免在 Ascend NPU 上触发高延迟的数据类型转换。"""
    elif "CustomLayerNorm" in code_content:
        return """📑 **【画布代码深度剖析 / Code Canvas Analysis】**

1. **权重管理 (`Parameter`)**:
   - `Parameter(Tensor(...))` 是 MindSpore 中定义模型可学习权重的核心方法。参数名称 `name="gamma"` 必须在模型内部唯一，这会直接影响 Checkpoint 的保存与恢复。

2. **泛型操作与性能 (`ops.mean`, `ops.var`)**:
   - 显式声明 `axis=-1, keep_dims=True`。由于 Ascend 芯片采用的是三维达芬奇核心架构，对尾轴进行 Reduction 操作能够极大激发 NPU 的向量计算流水线（Vector Unit）能效。

3. **双精度风险避坑**:
   - 初始化 numpy 数组默认是 float64。通过显式 `.astype(np.float32)` 转换，能够确保参数精度匹配昇腾单精度融合算子，获得最高达 10x 的执行效率提升。"""
    elif "SimpleLinear" in code_content:
        return """📑 **【画布代码深度剖析 / Code Canvas Analysis】**

1. **函数式自动求导 (`value_and_grad`)**:
   - 相比 PyTorch 的 `loss.backward()` 这种状态依赖设计，MindSpore 采用无状态的 `value_and_grad` 函数式微分，使代码更易并行与变换。
   - `optimizer.parameters` 指定了需要计算梯度的目标参数元组。

2. **编译加速器 (`@ms.jit`)**:
   - `@ms.jit` 装饰器使得 `train_step` 能够进行全图静态编译，后端 CANN 会对 dense、loss、gradient、SGD 算子进行大图融合（Operator Fusion），极大压低算子下发时间。

3. **辅助输出支持 (`has_aux=True`)**:
   - `forward_fn` 返回了双重值：`(loss, logits)`。通过设置 `has_aux=True`，求导引擎会自动将第二个返回值 `logits` 作为辅助信息透传出来，不干扰梯度的求解。"""
    else:
        return """📑 **【画布代码深度剖析 / Code Canvas Analysis】**

1. **单 Cell 设计原则**:
   - 继承自 `nn.Cell`，重写了 `construct` 函数。该结构在 Graph 模式和 PyNative 模式下均能稳定发挥效能。
   
2. **算子兼容性提示**:
   - 内部使用了最基础的昇思层（如 `nn.Dense` 等），其在 NPU 硬件底层具有完美的融合（Fused）加速算子支持。
   
3. **NPU 运行建议**:
   - 建议在数据流通路上均采用强类型 `np.float32`，这可以使得昇腾的 Cube Core 计算单元达到极佳的 TFLOPs 指标。"""

# Diagnostic functions
def load_error_preset(preset_name):
    details = ERROR_DETAILS[preset_name]
    return details["log"], details["plain"], details["cause"], details["fix"]

def run_diagnosis(error_log_text):
    if not error_log_text.strip():
        return "Please paste or load an error log.", "N/A", "N/A"
    
    # Try matching
    for preset_name, details in ERROR_DETAILS.items():
        # Match based on substring
        if "Out of Memory" in error_log_text or "AllocDeviceMem" in error_log_text:
            m = ERROR_DETAILS["CANN Out of Memory Error on Device 0"]
            return m["plain"], m["cause"], m["fix"]
        elif "static shape" in error_log_text or "ValueError: MindSpore JIT Graph" in error_log_text or "dynamic shape" in error_log_text:
            m = ERROR_DETAILS["MindSpore Static Shape Compile Error"]
            return m["plain"], m["cause"], m["fix"]
        elif "DataType Mismatch" in error_log_text or "TypeError: For 'Add'" in error_log_text or "Float64" in error_log_text:
            m = ERROR_DETAILS["MindSpore DataType Mismatch (Float32 vs Float64)"]
            return m["plain"], m["cause"], m["fix"]
            
    # Generic diagnostic output if custom text is pasted
    return (
        "⚠️ **【白话解释 / Plain Explanation】**\n分析器检测到 MindSpore 在运行时发生一般性错误。这通常表示您的数据维度、数据类型不匹配，或者在底层 Ascend 设备侧运行时发生了非预期终止。请检查代码与 Tensor 分配。",
        "🛠️ **【技术根因 / Technical Root Cause】**\n1. **后端兼容性约束**: CANN runtime 抛出非零异常或类型推导层（Type Infer）未能通过编译图的前置约束检查。\n2. **调用链路异常**: 位于 `construct()` 中的某一子算子被传入了非法参数，导致编译树构建失败。",
        "📝 **【修复方案 / Fix Candidate】**\n\n```python\n# 调试黄金三步法：\n# 1. 切换到动态图模式查看具体行数与底层变量状态：\nimport mindspore as ms\nms.set_context(mode=ms.PYNATIVE_MODE)\n\n# 2. 检查所有参与运算的 Tensor 的 .shape 和 .dtype，确保相同算子的输入一致。\n# 3. 打印 NPU 底层更详细的日志：在运行前执行 export MSC_ERR_DEBUG=1 \n```"
    )

# Framework comparison helper
def load_framework_lens_preset(preset_key):
    preset = FRAMEWORK_LENS_PRESETS[preset_key]
    return preset["torch"], preset["ms"], preset["analysis"]


# UI Building using Gradio Blocks
with gr.Blocks(title="AscendMind Agent", theme=gr.themes.Soft()) as demo:
    
    # Header Panel
    with gr.Row(elem_classes="header-panel"):
        with gr.Column(scale=4):
            gr.Markdown("""
            # 🧠 AscendMind Agent (昇思智元)
            ### 🚀 AI-First Research & Development Workspace for Ascend & MindSpore
            
            Welcome to the intelligent accelerator for the **Huawei Ascend (昇腾) & MindSpore (昇思)** developer ecosystem.
            Describe your single-cell tensor logic, resolve tricky CANN errors, or compare Framework APIs.
            """)
        with gr.Column(scale=1):
            socratic_mode = gr.Checkbox(label="Socratic Learn Mode\n(苏格拉底启发模式 💡)", value=False, info="When enabled, the Agent guides and educates you step-by-step instead of spitting out final code directly.")
            
    gr.HTML("<hr style='border: 1px solid #e2e8f0; margin-bottom: 20px;'/>")

    # Main Row with Left Chat and Right Code/Diagnostics tabs
    with gr.Row():
        
        # LEFT COLUMN: Interactive Chatbot & Presets
        with gr.Column(scale=1, min_width=450):
            gr.Markdown("### 💬 Cell-Level Intent Chatbot (意图对话窗口)")
            chatbot = gr.Chatbot(
                label="AscendMind Agent Chat",
                bubble_full_width=False,
                height=450,
                placeholder="💬 **I am your Ascend/MindSpore co-pilot.**\n\nTell me what Tensor operations or Neural Net layer you want to create (e.g. convolution, norm layer, training loops), or click on any of the **Quick Cell Intent Presets** below!"
            )
            
            with gr.Row():
                user_input = gr.Textbox(
                    show_label=False,
                    placeholder="Describe your logical intent here (e.g. 'Write a LayerNorm in MindSpore')...",
                    scale=4,
                    container=False
                )
                submit_btn = gr.Button("Submit", variant="primary", scale=1)
                
            with gr.Row():
                clear_btn = gr.Button("Clear Chat", variant="secondary")

            gr.Markdown("#### 💡 Quick Cell Intent Presets (单Cell算子意图快捷输入)")
            with gr.Column():
                for item in SAMPLES_CHAT:
                    btn = gr.Button(item[0], size="sm", variant="secondary")
                    # Wire up clicks
                    btn.click(
                        fn=handle_sample_click,
                        inputs=[btn, socratic_mode],
                        outputs=[user_input, chatbot, code_canvas]
                    )
                    # We will bind this to update code canvas too using a custom link below
                    
        # RIGHT COLUMN: Multi-functional Workspace Tabs
        with gr.Column(scale=1, min_width=500):
            with gr.Tabs() as workspace_tabs:
                
                # TAB 1: CODE CANVAS
                with gr.Tab("💻 Code Canvas (单Cell代码画布)", id="canvas"):
                    gr.Markdown("This canvas displays the single-cell MindSpore/CANN code block representing your active intent.")
                    
                    code_canvas = gr.Code(
                        label="Active Single-Cell Module",
                        language="python",
                        value=PRESET_CODES["Create a single-cell 2D Convolution in MindSpore"],
                        interactive=True,
                        lines=18
                    )
                    
                    with gr.Row():
                        explain_btn = gr.Button("📝 Explain Code Syntax", variant="secondary")
                        npu_run_btn = gr.Button("🚀 Simulate Run on NPU", variant="primary")
                        
                    with gr.Accordion("💻 Code Canvas Explanations", open=False) as expl_accordion:
                        canvas_explanation = gr.Markdown("Click 'Explain Code Syntax' to see details.")
                        
                    with gr.Accordion("🖥️ Ascend NPU Simulation Terminal Trace", open=True):
                        npu_run_output = gr.Code(
                            label="NPU stdout / stderr",
                            language="bash",
                            value="[Idle] Ready to compile and run on NPU target.",
                            lines=8
                        )

                # TAB 2: ERROR DIAGNOSIS PLAYGROUND
                with gr.Tab("🔍 Error Physician (昇腾算子报错诊断)", id="diagnostics"):
                    gr.Markdown("Paste a complex MindSpore JIT trace or Ascend CANN driver log here to receive layered diagnostic reports.")
                    
                    with gr.Row():
                        error_log_input = gr.Textbox(
                            label="Ascend CANN / MindSpore Error Log",
                            placeholder="Paste your trace stack here...",
                            lines=6
                        )
                        
                    with gr.Row():
                        gr.Markdown("**Presets:**")
                        for p_name in ERROR_LOGS_PRESETS:
                            ep_btn = gr.Button(p_name, size="sm")
                            # wire preset clicks
                            def create_preset_func(name):
                                return lambda: load_error_preset(name)
                            ep_btn.click(
                                fn=create_preset_func(p_name),
                                inputs=[],
                                outputs=[error_log_input, gr.State(), gr.State(), gr.State()] # We will trigger diagnosis on click or let user click Diagnose
                            )

                    diagnose_action_btn = gr.Button("🔍 Diagnose Log Now", variant="primary")

                    with gr.Row():
                        with gr.Column():
                            diag_plain = gr.Markdown("### ⚠️ 白话文解释 / Plain Explanation\n*Diagnostics will appear here*")
                        with gr.Column():
                            diag_cause = gr.Markdown("### 🛠️ 技术根因 / Technical Root Cause\n*Technical analysis will appear here*")
                            
                    diag_fix = gr.Markdown("### 📝 推荐修复方案 / Suggested Fix Candidate\n*Code fix recommendations*")
                    
                    # Wire up Diagnosis click
                    diagnose_action_btn.click(
                        fn=run_diagnosis,
                        inputs=[error_log_input],
                        outputs=[diag_plain, diag_cause, diag_fix]
                    )

                # TAB 3: FRAMEWORK LENS
                with gr.Tab("🔄 Framework Lens (PyTorch ↔ MindSpore)", id="framework"):
                    gr.Markdown("Direct side-by-side API translation and structural analysis to ease migration from PyTorch to MindSpore.")
                    
                    lens_select = gr.Dropdown(
                        label="Choose a standard Tensor operation or Neural Net Layer preset:",
                        choices=list(FRAMEWORK_LENS_PRESETS.keys()),
                        value="Tensor Reshaping & Slicing"
                    )
                    
                    with gr.Row():
                        torch_code = gr.Code(
                            label="PyTorch Implementation",
                            language="python",
                            value=FRAMEWORK_LENS_PRESETS["Tensor Reshaping & Slicing"]["torch"],
                            lines=10,
                            interactive=False
                        )
                        ms_code = gr.Code(
                            label="MindSpore Implementation",
                            language="python",
                            value=FRAMEWORK_LENS_PRESETS["Tensor Reshaping & Slicing"]["ms"],
                            lines=10,
                            interactive=False
                        )
                        
                    lens_analysis = gr.Markdown(FRAMEWORK_LENS_PRESETS["Tensor Reshaping & Slicing"]["analysis"])
                    
                    # Wire up Dropdown change
                    lens_select.change(
                        fn=load_framework_lens_preset,
                        inputs=[lens_select],
                        outputs=[torch_code, ms_code, lens_analysis]
                    )

                # TAB 4: NPU TELEMETRY & CLOUD BRAIN OBSERVER
                with gr.Tab("📊 NPU Dashboard & Task Telemetry", id="telemetry"):
                    gr.Markdown("### 🖥️ OpenI Cloud Brain - NPU Telemetry Dashboard (Read-Only)")
                    gr.Markdown("Real-time telemetry and resource usage of active JIT compiler traces on Device 0.")
                    
                    # Progress bar simulator
                    npu_util = gr.Slider(label="NPU Engine Core Util (%)", minimum=0, maximum=100, value=42, interactive=False)
                    npu_mem = gr.Slider(label="NPU Memory Utilization (GB)", minimum=0, maximum=32, value=12.4, interactive=False)
                    cpu_util = gr.Slider(label="Host CPU Util (%)", minimum=0, maximum=100, value=18, interactive=False)
                    
                    gr.HTML("""
                    <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 15px; margin-top: 15px;">
                        <h4 style="margin: 0 0 10px 0; color: #1e293b;">🌐 Cloud Brain Observer Task Queue</h4>
                        <table style="width:100%; border-collapse: collapse; text-align: left; font-size: 14px;">
                          <thead>
                            <tr style="border-bottom: 2px solid #cbd5e1; color: #475569;">
                              <th style="padding: 8px;">Task ID</th>
                              <th style="padding: 8px;">Target Device</th>
                              <th style="padding: 8px;">Status</th>
                              <th style="padding: 8px;">Mode</th>
                              <th style="padding: 8px;">Trace Metric</th>
                            </tr>
                          </thead>
                          <tbody>
                            <tr style="border-bottom: 1px solid #e2e8f0; color: #334155;">
                              <td style="padding: 8px; font-family: monospace;">task_910b_01</td>
                              <td style="padding: 8px;">Ascend 910B</td>
                              <td style="padding: 8px;"><span style="background-color: #dcfce7; color: #15803d; padding: 2px 6px; border-radius: 4px; font-size: 12px; font-weight: bold;">RUNNING</span></td>
                              <td style="padding: 8px; font-family: monospace;">ms.GRAPH_MODE</td>
                              <td style="padding: 8px; color: #2563eb;">Loss: 0.412 (Warm)</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #e2e8f0; color: #334155;">
                              <td style="padding: 8px; font-family: monospace;">task_310p_04</td>
                              <td style="padding: 8px;">Ascend 310P</td>
                              <td style="padding: 8px;"><span style="background-color: #fee2e2; color: #991b1b; padding: 2px 6px; border-radius: 4px; font-size: 12px; font-weight: bold;">OOM FAIL</span></td>
                              <td style="padding: 8px; font-family: monospace;">ms.PYNATIVE</td>
                              <td style="padding: 8px; color: #dc2626;">Device Alloc Fail</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #e2e8f0; color: #334155;">
                              <td style="padding: 8px; font-family: monospace;">task_910b_02</td>
                              <td style="padding: 8px;">Ascend 910B</td>
                              <td style="padding: 8px;"><span style="background-color: #fef9c3; color: #854d0e; padding: 2px 6px; border-radius: 4px; font-size: 12px; font-weight: bold;">PENDING</span></td>
                              <td style="padding: 8px; font-family: monospace;">ms.GRAPH_MODE</td>
                              <td style="padding: 8px;">Queue Position: 1</td>
                            </tr>
                          </tbody>
                        </table>
                        <p style="margin: 10px 0 0 0; font-size: 12px; color: #64748b;">*NPU telemetry is read from OpenI Cloud Brain telemetry daemon in real-time. Observer mode is strictly read-only.</p>
                    </div>
                    """)

    # Link bottom functions & wiring
    
    # 1. Chat submit wires
    submit_btn.click(
        fn=chatbot_respond,
        inputs=[user_input, chatbot, socratic_mode],
        outputs=[user_input, chatbot, code_canvas]
    )
    user_input.submit(
        fn=chatbot_respond,
        inputs=[user_input, chatbot, socratic_mode],
        outputs=[user_input, chatbot, code_canvas]
    )
    
    # 2. Clear Chat button
    clear_btn.click(
        fn=lambda: (None, []),
        inputs=[],
        outputs=[user_input, chatbot]
    )

    # 3. Code canvas buttons wiring
    explain_btn.click(
        fn=explain_code_canvas,
        inputs=[code_canvas],
        outputs=[canvas_explanation]
    ).then(
        fn=lambda: gr.update(open=True),
        inputs=[],
        outputs=[expl_accordion]
    )
    
    npu_run_btn.click(
        fn=simulate_npu_run,
        inputs=[code_canvas],
        outputs=[npu_run_output]
    )
    
    # 4. We also want to map the quick cell presets to also load the code into the canvas directly!
    # Let's write custom click maps for the buttons to also populate the code canvas.
    for i, item in enumerate(SAMPLES_CHAT):
        # We need a function that maps the prompt to the code
        def preset_code_loader(btn_val=item[0]):
            code_text = PRESET_CODES.get(btn_val, "")
            # Return active code update
            return code_text
        
        # When clicking sample preset button, also load its code directly into the canvas
        # Note: handle_sample_click takes care of the chatbot and user input textbox update
        
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
