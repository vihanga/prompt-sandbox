"""
ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metric evaluator
"""

from typing import Optional
from .base import Evaluator, EvaluationResult

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False


class ROUGEEvaluator(Evaluator):
    """
    ROUGE score evaluator (recall-based n-gram overlap)

    Common for summarization tasks, measures how much of the reference
    is captured in the generated text.

    Score range: 0-1 (higher is better)
    """

    def __init__(
        self,
        name: str = "rouge",
        rouge_types: list = None,
        use_stemmer: bool = True,
        **kwargs
    ):
        """
        Initialize ROUGE evaluator

        Args:
            name: Evaluator name
            rouge_types: List of ROUGE types to compute (default: ['rouge1', 'rouge2', 'rougeL'])
            use_stemmer: Use Porter stemmer
            **kwargs: Additional parameters
        """
        if not ROUGE_AVAILABLE:
            raise ImportError(
                "rouge-score is required for ROUGEEvaluator. "
                "Install with: pip install rouge-score"
            )

        super().__init__(name, **kwargs)

        if rouge_types is None:
            rouge_types = ['rouge1', 'rouge2', 'rougeL']

        self.rouge_types = rouge_types
        self.use_stemmer = use_stemmer
        self.scorer = rouge_scorer.RougeScorer(
            rouge_types=rouge_types,
            use_stemmer=use_stemmer
        )

    def evaluate(
        self,
        generated: str,
        reference: str,
        **kwargs
    ) -> EvaluationResult:
        """Evaluate ROUGE scores"""

        # Calculate ROUGE scores
        scores = self.scorer.score(reference, generated)

        # Use ROUGE-L F1 as primary score
        primary_score = scores['rougeL'].fmeasure

        # Build metadata with all ROUGE scores
        metadata = {
            "use_stemmer": self.use_stemmer,
        }

        for rouge_type, score in scores.items():
            metadata[f"{rouge_type}_precision"] = score.precision
            metadata[f"{rouge_type}_recall"] = score.recall
            metadata[f"{rouge_type}_fmeasure"] = score.fmeasure

        return EvaluationResult(
            score=primary_score,
            metric_name=self.name,
            generated_text=generated,
            reference_text=reference,
            metadata=metadata
        )
