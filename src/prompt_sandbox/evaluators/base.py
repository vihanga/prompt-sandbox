"""
Abstract base class for evaluators
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class EvaluationResult:
    """Result from evaluation metric"""

    score: float
    metric_name: str
    generated_text: str
    reference_text: str
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Evaluator(ABC):
    """Abstract base class for all evaluators"""

    def __init__(self, name: str, **kwargs):
        """
        Initialize evaluator

        Args:
            name: Evaluator identifier
            **kwargs: Evaluator-specific configuration
        """
        self.name = name
        self.config = kwargs

    @abstractmethod
    def evaluate(
        self,
        generated: str,
        reference: str,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate generated text against reference

        Args:
            generated: Model-generated text
            reference: Reference (ground truth) text
            **kwargs: Additional evaluation parameters

        Returns:
            EvaluationResult with score and metadata
        """
        pass

    def evaluate_batch(
        self,
        generated_texts: list,
        reference_texts: list,
        **kwargs
    ) -> list:
        """
        Evaluate multiple generated texts

        Args:
            generated_texts: List of generated texts
            reference_texts: List of reference texts
            **kwargs: Additional parameters

        Returns:
            List of EvaluationResult objects
        """
        if len(generated_texts) != len(reference_texts):
            raise ValueError(
                f"Length mismatch: {len(generated_texts)} generated vs "
                f"{len(reference_texts)} references"
            )

        return [
            self.evaluate(gen, ref, **kwargs)
            for gen, ref in zip(generated_texts, reference_texts)
        ]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
