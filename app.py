import os
import torch
import soundfile as sf
import logging
import gradio as gr
import librosa
import numpy as np
import warnings
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Use transformers backend instead of lmdeploy for better GPU compatibility
from mira.model_transformers import MiraTTS

# Suppress verbose logging
logging.basicConfig(level=logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("gradio").setLevel(logging.WARNING)

MODEL = None

def initialize_model():
    """Initialize MiraTTS model with error handling for HF Spaces."""
    global MODEL
    
    if MODEL is not None:
        return MODEL
        
    try:
        logging.info("Initializing MiraTTS model...")
        model_dir = "YatharthS/MiraTTS"
        
        # Initialize with transformers backend (works on RTX 5050/Blackwell)
        MODEL = MiraTTS(
            model_dir=model_dir,
            device="cuda",
            dtype=torch.bfloat16
        )
        
        logging.info("Model initialized successfully")
        return MODEL
        
    except Exception as e:
        logging.error(f"Model initialization failed: {e}")
        raise e

def validate_audio_input(audio_path):
    """Validate and preprocess audio input for HF Spaces."""
    if not audio_path or not os.path.exists(audio_path):
        raise ValueError("Audio file not found")
    
    try:
        audio, sr = librosa.load(audio_path, sr=None, duration=30)
        
        if len(audio) == 0:
            raise ValueError("Audio file is empty")
        
        min_length = int(0.5 * sr)
        if len(audio) < min_length:
            raise ValueError(f"Audio too short: {len(audio)/sr:.2f}s, minimum 0.5s required")
        
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        audio = audio / np.max(np.abs(audio))
        
        temp_dir = "/tmp" if os.path.exists("/tmp") else "."
        temp_path = os.path.join(temp_dir, f"processed_{os.path.basename(audio_path)}")
        sf.write(temp_path, audio, samplerate=sr)
        
        return temp_path, len(audio), sr
        
    except Exception as e:
        raise ValueError(f"Audio processing failed: {e}")


def generate_speech(text, prompt_audio_path, temperature=0.8, top_p=0.95, top_k=50, seed=-1):
    """Generate speech with GPU acceleration."""
    try:
        model = initialize_model()
        
        if not text or not text.strip():
            raise ValueError("Text input is empty")
        
        if seed >= 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            print(f"Using seed: {seed}")
        
        model.set_params(
            temperature=temperature,
            top_p=top_p,
            top_k=int(top_k),
            max_new_tokens=2048,
            repetition_penalty=1.2
        )
            
        processed_audio, length, sr = validate_audio_input(prompt_audio_path)
        
        context_tokens = model.encode_audio(processed_audio)
        if context_tokens is None:
            raise ValueError("Failed to encode reference audio")
            
        output_audio = model.generate(text, context_tokens)
        if output_audio is None:
            raise ValueError("Speech generation failed")
        
        if torch.is_tensor(output_audio):
            output_audio = output_audio.cpu().numpy()
            
        if output_audio.dtype == 'float16':
            output_audio = output_audio.astype('float32')
        
        if os.path.exists(processed_audio):
            os.remove(processed_audio)
            
        return output_audio, 48000
        
    except Exception as e:
        logging.error(f"Generation error: {e}")
        raise e

def voice_clone_interface(text, prompt_audio_upload, prompt_audio_record, temperature, top_p, top_k, seed):
    """Interface for voice cloning."""
    try:
        if not text or not text.strip():
            return None, "Please enter text to synthesize."
            
        prompt_audio = prompt_audio_upload if prompt_audio_upload else prompt_audio_record
        if not prompt_audio:
            return None, "Please upload or record reference audio."
        
        audio, sample_rate = generate_speech(
            text, prompt_audio, 
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed
        )
        
        os.makedirs("outputs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"outputs/mira_tts_{timestamp}.wav"
        sf.write(output_path, audio, samplerate=sample_rate)
        
        return output_path, "Generation successful!"
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        logging.error(error_msg)
        return None, error_msg

def build_interface():
    """Build fancy Gradio interface with consistent dark theme."""
    
    # Create a custom theme that looks the same in light and dark mode
    # by using explicit dark colors for everything
    theme = gr.themes.Base(
        primary_hue="violet",
        secondary_hue="purple", 
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    ).set(
        # Force dark background colors
        body_background_fill="#0f0f1a",
        body_background_fill_dark="#0f0f1a",
        background_fill_primary="#1a1a2e",
        background_fill_primary_dark="#1a1a2e",
        background_fill_secondary="#16162a",
        background_fill_secondary_dark="#16162a",
        # Force light text colors
        body_text_color="#e2e8f0",
        body_text_color_dark="#e2e8f0",
        body_text_color_subdued="#94a3b8",
        body_text_color_subdued_dark="#94a3b8",
        # Input/component backgrounds
        input_background_fill="#1e1e3a",
        input_background_fill_dark="#1e1e3a",
        input_border_color="#3d3d6b",
        input_border_color_dark="#3d3d6b",
        input_border_color_focus="#7c3aed",
        input_border_color_focus_dark="#7c3aed",
        # Block backgrounds
        block_background_fill="#1a1a2e",
        block_background_fill_dark="#1a1a2e",
        block_border_color="#2d2d4a",
        block_border_color_dark="#2d2d4a",
        block_label_background_fill="#1a1a2e",
        block_label_background_fill_dark="#1a1a2e",
        block_label_text_color="#c4b5fd",
        block_label_text_color_dark="#c4b5fd",
        block_title_text_color="#e2e8f0",
        block_title_text_color_dark="#e2e8f0",
        # Panel/accordion
        panel_background_fill="#16162a",
        panel_background_fill_dark="#16162a",
        panel_border_color="#2d2d4a",
        panel_border_color_dark="#2d2d4a",
        # Buttons
        button_primary_background_fill="linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        button_primary_background_fill_dark="linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        button_primary_text_color="white",
        button_primary_text_color_dark="white",
        button_secondary_background_fill="#2d2d4a",
        button_secondary_background_fill_dark="#2d2d4a",
        button_secondary_text_color="#e2e8f0",
        button_secondary_text_color_dark="#e2e8f0",
        # Borders and shadows
        border_color_primary="#3d3d6b",
        border_color_primary_dark="#3d3d6b",
        shadow_drop="0 4px 20px rgba(0, 0, 0, 0.3)",
        shadow_drop_lg="0 8px 32px rgba(0, 0, 0, 0.4)",
        # Checkbox/radio
        checkbox_background_color="#1e1e3a",
        checkbox_background_color_dark="#1e1e3a",
        checkbox_border_color="#3d3d6b",
        checkbox_border_color_dark="#3d3d6b",
        # Table
        table_even_background_fill="#1a1a2e",
        table_even_background_fill_dark="#1a1a2e",
        table_odd_background_fill="#16162a",
        table_odd_background_fill_dark="#16162a",
    )
    
    custom_css = """
/* Force dark mode colors everywhere */
*, *::before, *::after {
    --bg-primary: #1a1a2e !important;
    --bg-secondary: #16162a !important;
    --text-primary: #e2e8f0 !important;
    --text-secondary: #94a3b8 !important;
    --border-color: #3d3d6b !important;
    --input-bg: #1e1e3a !important;
}

/* Override any light mode styles */
body, .gradio-container, .dark, .light {
    background: #0f0f1a !important;
    color: #e2e8f0 !important;
}

/* Force all inputs to dark style */
input, textarea, select, 
.gr-textbox textarea, .gr-textbox input,
.gr-dropdown, .gr-number input,
[data-testid="textbox"], [data-testid="number-input"] {
    background: #1e1e3a !important;
    color: #e2e8f0 !important;
    border-color: #3d3d6b !important;
}

input::placeholder, textarea::placeholder {
    color: #64748b !important;
}

/* Audio components */
.gr-audio, audio {
    background: #1e1e3a !important;
}

/* Slider styling */
.gr-slider input[type="range"] {
    background: #3d3d6b !important;
}

/* Labels */
label, .gr-label, span.svelte-1gfkn6j {
    color: #c4b5fd !important;
}

/* Info text */
.gr-info, .info {
    color: #94a3b8 !important;
}

/* Header styling */
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
}

.main-header h1 {
    color: white !important;
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    margin: 0 !important;
}

.main-header p {
    color: rgba(255,255,255,0.9) !important;
    font-size: 1.1rem !important;
    margin-top: 0.5rem !important;
}

.section-title {
    color: #c4b5fd !important;
    font-weight: 600 !important;
    font-size: 1.2rem !important;
    margin-bottom: 1rem !important;
}

/* Generate button */
.generate-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    transition: all 0.3s ease !important;
}

.generate-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5) !important;
}

/* Tips card */
.tips-card {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    border-left: 4px solid #7c3aed;
    margin-top: 1rem;
    color: #e2e8f0 !important;
}

.tips-card h4 {
    color: #c4b5fd !important;
    margin: 0 0 0.5rem 0 !important;
}

.tips-card ul {
    color: #e2e8f0 !important;
}

.audio-hint {
    color: #94a3b8 !important;
    font-size: 0.85rem;
    font-style: italic;
    margin-top: 0.5rem;
}

/* Full width container */
.gradio-container {
    max-width: 100% !important;
    padding: 2rem 3rem !important;
}

/* Accordion */
.gr-accordion {
    background: #16162a !important;
    border-color: #2d2d4a !important;
}

.gr-accordion summary {
    color: #e2e8f0 !important;
}

/* Group styling */
.gr-group {
    background: #1a1a2e !important;
    border-color: #2d2d4a !important;
}

/* Block styling */
.gr-block {
    background: #1a1a2e !important;
}

/* Form elements */
.gr-form {
    background: transparent !important;
}
"""
    
    with gr.Blocks(title="MiraTTS - Voice Cloning", theme=theme, css=custom_css, fill_width=True) as demo:
        
        gr.HTML("""
        <div class="main-header">
            <h1>🎙️ MiraTTS Voice Cloning</h1>
            <p>Transform any voice with high-quality AI synthesis</p>
        </div>
        """)
        
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                gr.HTML('<div class="section-title">🎵 Reference Audio</div>')
                
                with gr.Group():
                    prompt_upload = gr.Audio(
                        sources=["upload"],
                        type="filepath",
                        label="Upload Audio File"
                    )
                    gr.HTML('<p class="audio-hint">Upload a clear voice sample (3-30 seconds)</p>')
                
                with gr.Group():
                    prompt_record = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        label="Or Record Your Voice"
                    )
                    gr.HTML('<p class="audio-hint">Click to record directly in browser</p>')
            
            with gr.Column(scale=1):
                gr.HTML('<div class="section-title">✍️ Text to Speak</div>')
                
                text_input = gr.Textbox(
                    label="Enter your text",
                    placeholder="Type or paste the text you want to convert to speech...",
                    lines=6,
                    value="Hello! This is a demonstration of MiraTTS, an advanced text-to-speech model that can clone any voice."
                )
                
                with gr.Accordion("⚙️ Generation Settings", open=False):
                    with gr.Row():
                        temperature = gr.Slider(
                            minimum=0.1, maximum=1.5, step=0.05, value=0.8,
                            label="Temperature", info="Higher = more creative/varied"
                        )
                        top_p = gr.Slider(
                            minimum=0.1, maximum=1.0, step=0.05, value=0.95,
                            label="Top-P", info="Nucleus sampling threshold"
                        )
                    with gr.Row():
                        top_k = gr.Slider(
                            minimum=1, maximum=100, step=1, value=50,
                            label="Top-K", info="Vocabulary size limit"
                        )
                        seed = gr.Number(
                            value=-1, label="Seed",
                            info="-1 for random, or set for reproducibility",
                            precision=0
                        )
                
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear All", variant="secondary", elem_classes=["clear-btn"], scale=1)
                    generate_btn = gr.Button("🚀 Generate Speech", variant="primary", elem_classes=["generate-btn"], scale=2)
        
        gr.HTML('<div class="section-title" style="margin-top: 1.5rem;">🔊 Generated Output</div>')
        
        with gr.Row():
            with gr.Column(scale=2):
                output_audio = gr.Audio(label="Generated Speech", type="filepath", autoplay=True)
            with gr.Column(scale=1):
                status_text = gr.Textbox(label="Status", interactive=False, value="Ready to generate")
        
        gr.HTML("""
        <div class="tips-card">
            <h4>💡 Tips for Best Results</h4>
            <ul>
                <li>Use clear audio without background noise or music</li>
                <li>Reference audio should be 3-30 seconds long</li>
                <li>Higher quality input = better output</li>
                <li>Shorter text generates faster</li>
            </ul>
        </div>
        """)
        
        generate_btn.click(
            voice_clone_interface,
            inputs=[text_input, prompt_upload, prompt_record, temperature, top_p, top_k, seed],
            outputs=[output_audio, status_text],
            show_progress="full"
        )
        
        def clear_all():
            return None, None, "", None, "Ready to generate", 0.8, 0.95, 50, -1
        
        clear_btn.click(
            clear_all,
            outputs=[prompt_upload, prompt_record, text_input, output_audio, status_text, temperature, top_p, top_k, seed]
        )
    
    return demo

if __name__ == "__main__":
    demo = build_interface()
    # Force dark mode by redirecting to ?__theme=dark
    demo.launch(share=False, show_error=True)
