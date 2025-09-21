"""
BERTScore evaluator - semantic similarity using BERT embeddings
"""

from typing import Optional
from .base import Evaluator, EvaluationResult

try:
    from bert_score import score as bert_score_fn
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False


class BERTScoreEvaluator(Evaluator):
    """
    BERTScore evaluator (semantic similarity using contextual embeddings)

    More advanced than BLEU/ROUGE - captures semantic meaning rather than
    just lexical overlap. Uses BERT model to compute embedding similarity.

    Score range: 0-1 (higher is better)
    """

    def __init__(
        self,
        name: str = "bertscore",
        model_type: str = "microsoft/deberta-xlarge-mnli",
        lang: str = "en",
        rescale_with_baseline: bool = True,
        **kwargs
    ):
        """
        Initialize BERTScore evaluator

        Args:
            name: Evaluator name
            model_type: BERT model to use for embeddings
            lang: Language code
            rescale_with_baseline: Rescale scores with baseline
            **kwargs: Additional parameters
        """
        if not BERTSCORE_AVAILABLE:
            raise ImportError(
                "bert-score is required for BERTScoreEvaluator. "
                "Install with: pip install bert-score"
            )

        super().__init__(name, **kwargs)
        self.model_type = model_type
        self.lang = lang
        self.rescale_with_baseline = rescale_with_baseline

    def evaluate(
        self,
        generated: str,
        reference: str,
        **kwargs
    ) -> EvaluationResult:
        """Evaluate BERTScore"""

        # Calculate BERTScore
        # Returns precision, recall, F1 for each example
        P, R, F1 = bert_score_fn(
            [generated],
            [reference],
            model_type=self.model_type,
            lang=self.lang,
            rescale_with_baseline=self.rescale_with_baseline,
            verbose=False
        )

        # Use F1 as primary score
        f1_score = F1.item()

        return EvaluationResult(
            score=f1_score,
            metric_name=self.name,
            generated_text=generated,
            reference_text=reference,
            metadata={
                "precision": P.item(),
                "recall": R.item(),
                "f1": f1_score,
                "model_type": self.model_type,
                "lang": self.lang,
            }
        )
