"""
Alternative MiraTTS implementation using transformers instead of lmdeploy.
This version works on newer GPUs like RTX 5050 (Blackwell) that aren't yet supported by lmdeploy.
"""

import gc
import re
import torch
import warnings
import sys
from itertools import cycle
from ncodec.codec import TTSCodec
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TextStreamer

warnings.filterwarnings("ignore", message="`torch_dtype` is deprecated")
warnings.filterwarnings("ignore", message="`generation_config` default values")

from mira.utils import clear_cache, split_text


class ProgressStreamer(TextStreamer):
    """Custom streamer that shows a progress bar during generation."""
    
    def __init__(self, tokenizer, max_tokens=1024):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.token_count = 0
        self.max_tokens = max_tokens
        self.bar_width = 40
        
    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.token_count += 1
        progress = min(self.token_count / self.max_tokens, 1.0)
        filled = int(self.bar_width * progress)
        bar = "█" * filled + "░" * (self.bar_width - filled)
        sys.stdout.write(f"\rGenerating: [{bar}] {self.token_count} tokens")
        sys.stdout.flush()
        if stream_end:
            print()  # New line when done


def smart_chunk_text(text, max_chars=250):
    """
    Split text into chunks at sentence boundaries.
    Ensures no words are cut off or merged.
    """
    # Clean up the text
    text = text.strip()
    if not text:
        return []
    
    # Split into sentences using regex (handles ., !, ?, and keeps the punctuation)
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text)
    
    # Remove empty sentences and strip whitespace
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would exceed max_chars
        if current_chunk and len(current_chunk) + len(sentence) + 1 > max_chars:
            # Save current chunk and start new one
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            # Add sentence to current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Handle case where a single sentence is longer than max_chars
    # Split on commas or other natural breaks
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_chars * 1.5:  # Allow some flexibility
            # Try splitting on commas
            parts = re.split(r',\s*', chunk)
            sub_chunk = ""
            for part in parts:
                if sub_chunk and len(sub_chunk) + len(part) + 2 > max_chars:
                    final_chunks.append(sub_chunk.strip())
                    sub_chunk = part
                else:
                    if sub_chunk:
                        sub_chunk += ", " + part
                    else:
                        sub_chunk = part
            if sub_chunk:
                final_chunks.append(sub_chunk.strip())
        else:
            final_chunks.append(chunk)
    
    return final_chunks


class MiraTTS:

    def __init__(self, model_dir="YatharthS/MiraTTS", device="cuda", dtype=torch.bfloat16):
        """
        Initialize MiraTTS with transformers backend.
        
        Args:
            model_dir: HuggingFace model ID or local path
            device: Device to run on ("cuda" or "cpu")
            dtype: Model dtype (torch.bfloat16 or torch.float16)
        """
        self.device = device
        self.dtype = dtype
        
        print(f"Loading model from {model_dir}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True
        )
        self.model.eval()
        
        self.gen_config = GenerationConfig(
            top_p=0.95,
            top_k=50,
            temperature=0.8,
            max_new_tokens=1024,
            repetition_penalty=1.2,
            do_sample=True,
        )
        
        self.codec = TTSCodec()
        print("Model loaded successfully!")

    def set_params(self, top_p=0.95, top_k=50, temperature=0.8, max_new_tokens=1024, repetition_penalty=1.2, min_p=0.05):
        """Sets sampling parameters for the LLM."""
        self.gen_config = GenerationConfig(
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            do_sample=True,
        )

    def c_cache(self):
        clear_cache()

    def split_text(self, text):
        return split_text(text)

    def encode_audio(self, audio_file):
        """Encodes audio into context tokens."""
        context_tokens = self.codec.encode(audio_file)
        return context_tokens

    def generate(self, text, context_tokens):
        """Generates speech from input text with automatic chunking for long text."""
        # Split text into manageable chunks
        chunks = smart_chunk_text(text, max_chars=250)
        
        if not chunks:
            return None
            
        print(f"\n{'='*50}")
        print(f"Processing {len(chunks)} chunk(s)")
        print(f"{'='*50}")
        
        all_audio = []
        
        for i, chunk in enumerate(chunks):
            print(f"\n[Chunk {i+1}/{len(chunks)}] {chunk[:60]}{'...' if len(chunk) > 60 else ''}")
            
            formatted_prompt = self.codec.format_prompt(chunk, context_tokens, None)
            
            # Tokenize the prompt
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
            
            # Create progress streamer
            streamer = ProgressStreamer(self.tokenizer, max_tokens=self.gen_config.max_new_tokens)
            
            # Store input length to extract only new tokens later
            input_length = inputs['input_ids'].shape[1]
            
            # Generate with progress bar
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=self.gen_config,
                    pad_token_id=self.tokenizer.eos_token_id,
                    streamer=streamer,
                )
            
            # Extract only the newly generated tokens (not the prompt)
            new_tokens = outputs[0][input_length:]
            response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
            
            # Decode audio from the response
            print("Decoding audio...")
            audio = self.codec.decode(response_text, context_tokens)
            
            if audio is not None:
                if torch.is_tensor(audio):
                    all_audio.append(audio)
                else:
                    all_audio.append(torch.tensor(audio))
            
            # Clear cache between chunks to manage memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f"\n{'='*50}")
        print(f"Combining {len(all_audio)} audio segments...")
        print(f"{'='*50}")
        
        # Concatenate all audio chunks
        if all_audio:
            # Add small silence between chunks for natural pacing (0.15 seconds at 48kHz)
            silence = torch.zeros(7200)  # ~150ms silence
            
            combined = []
            for i, audio in enumerate(all_audio):
                if audio.dim() > 1:
                    audio = audio.squeeze()
                combined.append(audio)
                if i < len(all_audio) - 1:  # Don't add silence after last chunk
                    combined.append(silence)
            
            final_audio = torch.cat(combined, dim=0)
            print("Done!")
            return final_audio
        
        return None

    def batch_generate(self, prompts, context_tokens):
        """
        Generates speech from text for larger batch sizes.

        Args:
            prompts (list): Input for TTS model, list of prompts
            context_tokens (list): List of context tokens
        """
        formatted_prompts = []
        for prompt, context_token in zip(prompts, cycle(context_tokens)):
            formatted_prompt = self.codec.format_prompt(prompt, context_token, None)
            formatted_prompts.append(formatted_prompt)

        # Tokenize all prompts
        inputs = self.tokenizer(
            formatted_prompts, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)

        # Generate for all prompts
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=self.gen_config,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode all outputs
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)

        audios = []
        for generated_text, formatted_prompt, context_token in zip(
            generated_texts, formatted_prompts, cycle(context_tokens)
        ):
            prompt_len = len(formatted_prompt)
            response_text = generated_text[prompt_len:].strip()
            audio = self.codec.decode(response_text, context_token)
            audios.append(audio)

        audios = torch.cat(audios, dim=0)
        return audios
