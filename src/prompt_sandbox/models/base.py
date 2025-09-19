"""
Abstract base class for model backends
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum


class ModelType(Enum):
    """Supported model backend types"""

    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    OLLAMA = "ollama"
    VLLM = "vllm"


@dataclass
class GenerationResult:
    """Result from model generation"""

    text: str
    model_name: str
    prompt: str
    generation_time: float
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ModelBackend(ABC):
    """Abstract base class for all model backends"""

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize model backend

        Args:
            model_name: Model identifier (HF model name, OpenAI model ID, etc.)
            **kwargs: Backend-specific configuration
        """
        self.model_name = model_name
        self.config = kwargs

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> GenerationResult:
        """
        Generate text from prompt

        Args:
            prompt: Input prompt string
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            top_p: Nucleus sampling threshold
            **kwargs: Additional generation parameters

        Returns:
            GenerationResult with generated text and metadata
        """
        pass

    @abstractmethod
    async def generate_async(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> GenerationResult:
        """
        Async version of generate()

        Same parameters as generate()
        """
        pass

    @abstractmethod
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> List[GenerationResult]:
        """
        Generate text for multiple prompts (batch processing)

        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens per generation
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            **kwargs: Additional generation parameters

        Returns:
            List of GenerationResult objects
        """
        pass

    @abstractmethod
    async def generate_batch_async(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> List[GenerationResult]:
        """
        Async batch generation

        Same parameters as generate_batch()
        """
        pass

    @abstractmethod
    def close(self):
        """Clean up resources (close connections, unload models, etc.)"""
        pass

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources"""
        self.close()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"
