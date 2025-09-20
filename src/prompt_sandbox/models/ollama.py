"""
Ollama backend for local model inference (100% free)
"""

import time
import asyncio
import json
from typing import List, Optional
from .base import ModelBackend, GenerationResult

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class OllamaBackend(ModelBackend):
    """
    Backend for Ollama local models (100% free, easy setup)

    Supports models like:
    - llama3.1
    - mistral
    - phi3
    - qwen2.5
    etc.

    Requires Ollama daemon running: `ollama serve`
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        **kwargs
    ):
        """
        Initialize Ollama backend

        Args:
            model_name: Ollama model name (e.g., "llama3.1", "mistral")
            base_url: Ollama API endpoint (default localhost:11434)
            **kwargs: Additional configuration
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError(
                "requests is required for OllamaBackend. "
                "Install with: pip install requests"
            )

        super().__init__(model_name, **kwargs)
        self.base_url = base_url.rstrip('/')
        self.generate_url = f"{self.base_url}/api/generate"

        # Verify Ollama is running
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Make sure Ollama is running: `ollama serve`\n"
                f"Error: {e}"
            )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> GenerationResult:
        """Generate text from prompt"""

        start_time = time.time()

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        }

        response = requests.post(self.generate_url, json=payload, timeout=300)
        response.raise_for_status()

        result = response.json()
        generated_text = result.get("response", "")

        generation_time = time.time() - start_time

        return GenerationResult(
            text=generated_text.strip(),
            model_name=self.model_name,
            prompt=prompt,
            generation_time=generation_time,
            metadata={
                "tokens_generated": result.get("eval_count", 0),
                "temperature": temperature,
                "top_p": top_p,
                "backend": "ollama"
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
        """Async generation"""

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
        """Batch generation (sequential for Ollama)"""

        return [
            self.generate(prompt, max_new_tokens, temperature, top_p, **kwargs)
            for prompt in prompts
        ]

    async def generate_batch_async(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> List[GenerationResult]:
        """Async batch generation (parallel)"""

        tasks = [
            self.generate_async(prompt, max_new_tokens, temperature, top_p, **kwargs)
            for prompt in prompts
        ]
        return await asyncio.gather(*tasks)

    def close(self):
        """No resources to cleanup for Ollama (stateless HTTP)"""
        pass

    def __repr__(self) -> str:
        return f"OllamaBackend(model_name='{self.model_name}', base_url='{self.base_url}')"
