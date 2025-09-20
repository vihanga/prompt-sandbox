"""
BLEU (Bilingual Evaluation Understudy) metric evaluator
"""

from typing import Optional
from .base import Evaluator, EvaluationResult

try:
    from sacrebleu import sentence_bleu
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False


class BLEUEvaluator(Evaluator):
    """
    BLEU score evaluator (precision-based n-gram overlap)

    Common for translation tasks, measures how many n-grams in the
    generated text match the reference.

    Score range: 0-100 (higher is better)
    """

    def __init__(
        self,
        name: str = "bleu",
        lowercase: bool = False,
        **kwargs
    ):
        """
        Initialize BLEU evaluator

        Args:
            name: Evaluator name
            lowercase: Convert to lowercase before evaluation
            **kwargs: Additional sacrebleu parameters
        """
        if not SACREBLEU_AVAILABLE:
            raise ImportError(
                "sacrebleu is required for BLEUEvaluator. "
                "Install with: pip install sacrebleu"
            )

        super().__init__(name, **kwargs)
        self.lowercase = lowercase

    def evaluate(
        self,
        generated: str,
        reference: str,
        **kwargs
    ) -> EvaluationResult:
        """Evaluate BLEU score"""

        # Apply preprocessing
        if self.lowercase:
            generated = generated.lower()
            reference = reference.lower()

        # Calculate BLEU score
        # sacrebleu expects list of references
        bleu_result = sentence_bleu(generated, [reference])

        return EvaluationResult(
            score=bleu_result.score,
            metric_name=self.name,
            generated_text=generated,
            reference_text=reference,
            metadata={
                "precision_1": bleu_result.precisions[0] if bleu_result.precisions else 0,
                "precision_2": bleu_result.precisions[1] if len(bleu_result.precisions) > 1 else 0,
                "precision_3": bleu_result.precisions[2] if len(bleu_result.precisions) > 2 else 0,
                "precision_4": bleu_result.precisions[3] if len(bleu_result.precisions) > 3 else 0,
                "bp": bleu_result.bp,  # Brevity penalty
                "lowercase": self.lowercase,
            }
        )
