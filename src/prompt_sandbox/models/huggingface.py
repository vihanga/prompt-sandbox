"""
Hugging Face Transformers backend for local model inference
"""

import time
import asyncio
from typing import List, Optional
from .base import ModelBackend, GenerationResult

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class HuggingFaceBackend(ModelBackend):
    """
    Backend for Hugging Face transformers models (local inference)

    Supports any causal LM model from HuggingFace Hub:
    - meta-llama/Llama-2-7b-chat-hf
    - mistralai/Mistral-7B-Instruct-v0.2
    - Qwen/Qwen2.5-7B-Instruct
    etc.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        load_in_8bit: bool = False,
        torch_dtype: str = "float16",
        **kwargs
    ):
        """
        Initialize HuggingFace model backend

        Args:
            model_name: HuggingFace model identifier (e.g., "meta-llama/Llama-2-7b-hf")
            device: Device placement ("auto", "cuda", "cpu")
            load_in_8bit: Use 8-bit quantization (reduces memory)
            torch_dtype: Torch data type (float16, float32, bfloat16)
            **kwargs: Additional transformers loading arguments
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers and torch are required for HuggingFaceBackend. "
                "Install with: pip install torch transformers accelerate"
            )

        super().__init__(model_name, **kwargs)

        self.device = device
        self.load_in_8bit = load_in_8bit
        self.torch_dtype = getattr(torch, torch_dtype, torch.float16)

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        load_kwargs = {
            "device_map": device,
            "torch_dtype": self.torch_dtype,
        }

        if load_in_8bit:
            load_kwargs["load_in_8bit"] = True

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> GenerationResult:
        """Generate text from prompt (synchronous)"""

        start_time = time.time()

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs
            )

        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],  # Skip prompt tokens
            skip_special_tokens=True
        )

        generation_time = time.time() - start_time

        return GenerationResult(
            text=generated_text.strip(),
            model_name=self.model_name,
            prompt=prompt,
            generation_time=generation_time,
            metadata={
                "tokens_generated": outputs.shape[1] - inputs['input_ids'].shape[1],
                "temperature": temperature,
                "top_p": top_p,
            }
        )

    async def generate_async(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> GenerationResult:
        """Async generation (runs in thread pool to avoid blocking)"""

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate(prompt, max_new_tokens, temperature, top_p, **kwargs)
        )

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> List[GenerationResult]:
        """Batch generation (processes all prompts together)"""

        start_time = time.time()

        # Tokenize all prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)

        # Generate for all prompts
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs
            )

        total_time = time.time() - start_time

        # Decode all outputs
        results = []
        for i, output in enumerate(outputs):
            prompt_len = inputs['input_ids'][i].shape[0]
            generated_text = self.tokenizer.decode(
                output[prompt_len:],
                skip_special_tokens=True
            )

            results.append(
                GenerationResult(
                    text=generated_text.strip(),
                    model_name=self.model_name,
                    prompt=prompts[i],
                    generation_time=total_time / len(prompts),  # Amortized time
                    metadata={
                        "batch_size": len(prompts),
                        "tokens_generated": output.shape[0] - prompt_len,
                        "temperature": temperature,
                        "top_p": top_p,
                    }
                )
            )

        return results

    async def generate_batch_async(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> List[GenerationResult]:
        """Async batch generation"""

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate_batch(prompts, max_new_tokens, temperature, top_p, **kwargs)
        )

    def close(self):
        """Clean up model and free GPU memory"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __repr__(self) -> str:
        return (
            f"HuggingFaceBackend(model_name='{self.model_name}', "
            f"device='{self.device}', load_in_8bit={self.load_in_8bit})"
        )
