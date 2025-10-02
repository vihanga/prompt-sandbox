"""
Unit tests for evaluators
"""

import pytest
from prompt_sandbox.evaluators import BLEUEvaluator, ROUGEEvaluator, BERTScoreEvaluator


class TestBLEUEvaluator:
    """Tests for BLEU evaluator"""

    def setup_method(self):
        """Setup test fixtures"""
        self.evaluator = BLEUEvaluator()

    def test_perfect_match(self):
        """Test BLEU score for perfect match"""
        generated = "The quick brown fox jumps over the lazy dog"
        reference = "The quick brown fox jumps over the lazy dog"

        result = self.evaluator.evaluate(generated, reference)

        assert result.score == pytest.approx(100.0, rel=1e-9)
        assert result.metric_name == "bleu"

    def test_partial_match(self):
        """Test BLEU score for partial match"""
        generated = "The quick brown fox"
        reference = "The quick brown fox jumps over the lazy dog"

        result = self.evaluator.evaluate(generated, reference)

        assert 0 < result.score < 100
        assert result.metric_name == "bleu"

    def test_no_match(self):
        """Test BLEU score for no overlap"""
        generated = "Hello world"
        reference = "The quick brown fox"

        result = self.evaluator.evaluate(generated, reference)

        assert result.score == 0.0


class TestROUGEEvaluator:
    """Tests for ROUGE evaluator"""

    def setup_method(self):
        """Setup test fixtures"""
        self.evaluator = ROUGEEvaluator()

    def test_perfect_match(self):
        """Test ROUGE score for perfect match"""
        generated = "The cat sat on the mat"
        reference = "The cat sat on the mat"

        result = self.evaluator.evaluate(generated, reference)

        assert result.score == 1.0
        assert result.metric_name == "rouge"
        assert "rouge1_fmeasure" in result.metadata

    def test_partial_match(self):
        """Test ROUGE score for partial match"""
        generated = "The cat sat"
        reference = "The cat sat on the mat"

        result = self.evaluator.evaluate(generated, reference)

        assert 0 < result.score < 1.0

    def test_no_match(self):
        """Test ROUGE score for no overlap"""
        generated = "Hello world"
        reference = "The quick brown fox"

        result = self.evaluator.evaluate(generated, reference)

        assert result.score == 0.0


class TestBERTScoreEvaluator:
    """Tests for BERTScore evaluator"""

    @pytest.mark.slow  # Skip in fast test runs
    def test_semantic_similarity(self):
        """Test BERTScore for semantic similarity"""
        evaluator = BERTScoreEvaluator(model_type="distilbert-base-uncased")

        # Semantically similar but different wording
        generated = "A dog is running in the park"
        reference = "The canine is sprinting through the garden"

        result = evaluator.evaluate(generated, reference)

        # Should have reasonable similarity despite different words
        assert result.score > 0.5
        assert "precision" in result.metadata
        assert "recall" in result.metadata
        assert "f1" in result.metadata
