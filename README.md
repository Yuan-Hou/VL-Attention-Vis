# VL-Attention-Vis

一个用于可视化视觉-语言模型（如 Qwen3-VL）注意力机制的工具。

![Screenshot](./img/screenshot.png)

## 项目简介

本项目提供了从数据提取到交互式可视化的完整工具链，帮助你深入理解视觉-语言模型如何处理图像和文本。

**核心功能：**
- 🔍 提取模型的 token 级别 attention 权重
- 🎨 交互式热力图可视化
- 📊 支持多层多头 attention 分析
- 💾 高效的数据存储（NPZ压缩格式）

## 快速开始

### 1. 提取 Attention 数据

使用 Python 脚本从 Qwen3-VL 模型中提取 attention 数据：

```bash
python extract_attention.py --image <图片路径> --text "你的提示文本" --output data/output.json
```

**示例：**
```bash
python extract_attention.py --image ./example.jpg --text "描述这张图片" --output data/attention_output.json
```

**可选参数：**
- `--model`: 模型名称或路径（默认: `Qwen/Qwen3-VL-4B-Instruct`）
- `--image-start-id`, `--image-end-id`, `--image-pad-id`: 图像token ID配置
- `--use-pad-mode`: 使用pad模式识别图像token

**输出文件：**
- `data/output.json` - 元数据和token信息（~2MB）
- `data/output.npz` - 压缩的attention数据（~4-5MB）

### 2. 启动可视化界面

使用 Python 内置 HTTP 服务器：

```bash
python -m http.server 8000
```

然后在浏览器中打开：
```
http://localhost:8000/visualize/index.html
```

## 可视化功能

- **左侧大图**：显示选中token的attention热力图覆盖在原图上
- **右侧网格**：显示当前layer所有heads的attention maps
- **底部文本**：输入/输出文本，点击token查看对应的attention
- **Layer切换**：按钮颜色反映该layer的attention强度
- **色彩控制**：调节热力图的色彩范围和对比度

**交互操作：**
- 点击输出token → 固定显示该token的attention
- 悬浮小图 → 在大图中查看该layer/head的详细attention
- 点击Layer按钮 → 切换显示不同layer的attention maps
- 调节滑块 → 自定义热力图色彩范围

## 依赖要求

```bash
pip install torch transformers pillow numpy
```

## License

MIT
