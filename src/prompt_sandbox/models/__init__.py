"""
Model backends for LLM inference
"""

from .base import ModelBackend, GenerationResult
from .huggingface import HuggingFaceBackend

__all__ = ["ModelBackend", "GenerationResult", "HuggingFaceBackend"]
