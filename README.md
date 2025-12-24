# 🎙️ MiraTTS - Voice Cloning

High-quality AI voice cloning using the MiraTTS model with a transformers backend. Clone any voice with just a few seconds of reference audio.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Gradio](https://img.shields.io/badge/Gradio-6.2.0-orange)
![CUDA](https://img.shields.io/badge/CUDA-12.8-green)

## ✨ Features

- **Voice Cloning** - Clone any voice from 3-30 seconds of reference audio
- **Smart Text Chunking** - Handles long text by splitting at sentence boundaries
- **Generation Settings** - Adjustable temperature, top_p, top_k, and seed
- **Progress Tracking** - Real-time progress bar during generation
- **Modern UI** - Clean dark-themed Gradio interface

## 🚀 Installation

### Prerequisites
- Python 3.10 or 3.11
- NVIDIA GPU with CUDA support
- [uv](https://docs.astral.sh/uv/) package manager (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/SUP3RMASS1VE/MiraTTS.git
cd MiraTTS
```

2. Create and activate a virtual environment:
```bash
uv venv
# Windows
.\env\Scripts\activate
```

3. Install dependencies:
```bash
uv pip install -r requirements.txt
```

4. Install PyTorch with CUDA 12.8 support:
```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
uv pip install torchaudio --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
```

## 💻 Usage

Run the web interface:
```bash
python app.py
```

Then open http://127.0.0.1:7860 in your browser.

### How to Use

1. **Upload or record** a reference audio sample (3-30 seconds of clear speech)
2. **Enter the text** you want to convert to speech
3. **Adjust settings** (optional) - temperature, top_p, top_k, seed
4. Click **Generate Speech**

## ⚙️ Generation Settings

| Setting | Default | Description |
|---------|---------|-------------|
| Temperature | 0.8 | Higher = more creative/varied output |
| Top-P | 0.95 | Nucleus sampling threshold |
| Top-K | 50 | Vocabulary size limit |
| Seed | -1 | Set for reproducible results (-1 = random) |

## 🔧 GPU Compatibility

This implementation uses the **transformers backend** instead of lmdeploy, providing better compatibility with newer GPUs including:

- RTX 50 series (Blackwell architecture)
- RTX 40 series
- RTX 30 series
- And other CUDA-capable GPUs

## 📁 Project Structure

```
MiraTTS/
├── app.py                    # Main Gradio web interface
├── requirements.txt          # Python dependencies
├── mira/
│   ├── model_transformers.py # Transformers-based TTS model
│   ├── model.py              # Original lmdeploy model (not used)
│   └── utils.py              # Utility functions
└── outputs/                  # Generated audio files
```

## 📝 Tips for Best Results

- Use **clear audio** without background noise or music
- Reference audio should be **3-30 seconds** long
- Higher quality input = better output
- Shorter text generates faster

## 🙏 Credits

Based on the [MiraTTS](https://huggingface.co/YatharthS/MiraTTS) model by YatharthS.

## 📄 License

MIT License

