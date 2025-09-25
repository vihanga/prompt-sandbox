"""
Integration tests for full experiment workflow
"""

import asyncio
import pytest
from pathlib import Path
import tempfile
import shutil

from prompt_sandbox.config.schema import PromptConfig
from prompt_sandbox.prompts.template import PromptTemplate
from prompt_sandbox.models.base import ModelBackend, GenerationResult
from prompt_sandbox.evaluators import BLEUEvaluator, ROUGEEvaluator
from prompt_sandbox.experiments import AsyncExperimentRunner, ExperimentConfig
from prompt_sandbox.experiments.storage import ResultStorage


class MockModelBackend(ModelBackend):
    """Mock model for testing without external dependencies"""

    def __init__(self, name: str = "mock-model"):
        self.model_name = name

    def generate(self, prompt: str, max_new_tokens: int = 512,
                 temperature: float = 0.7, top_p: float = 0.9, **kwargs) -> GenerationResult:
        # Simple deterministic response for testing
        if "2+2" in prompt:
            response = "The answer is 4."
        elif "capital" in prompt and "France" in prompt:
            response = "The capital of France is Paris."
        else:
            response = "This is a test response."

        return GenerationResult(
            prompt=prompt,
            generated_text=response,
            tokens_generated=len(response.split()),
            generation_time=0.1,
            model_name=self.model_name
        )

    async def generate_async(self, prompt: str, max_new_tokens: int = 512,
                            temperature: float = 0.7, top_p: float = 0.9, **kwargs) -> GenerationResult:
        await asyncio.sleep(0.01)  # Simulate async delay
        return self.generate(prompt, max_new_tokens, temperature, top_p, **kwargs)


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs"""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_prompts():
    """Create sample prompts for testing"""
    prompt1 = PromptTemplate(PromptConfig(
        name="direct_prompt",
        template="Q: {{question}}\nA:",
        variables=["question"]
    ))

    prompt2 = PromptTemplate(PromptConfig(
        name="cot_prompt",
        template="Q: {{question}}\nLet's think step by step:\nA:",
        variables=["question"]
    ))

    return [prompt1, prompt2]


@pytest.fixture
def sample_test_cases():
    """Create sample test cases"""
    return [
        {
            "input": {"question": "What is 2+2?"},
            "expected_output": "4"
        },
        {
            "input": {"question": "What is the capital of France?"},
            "expected_output": "Paris"
        }
    ]


@pytest.mark.asyncio
async def test_full_experiment_workflow(temp_output_dir, sample_prompts, sample_test_cases):
    """Test complete experiment workflow from config to results"""

    # Setup
    model = MockModelBackend("test-model-v1")
    evaluators = [BLEUEvaluator(), ROUGEEvaluator()]

    config = ExperimentConfig(
        name="integration_test_experiment",
        prompts=sample_prompts,
        models=[model],
        evaluators=evaluators,
        test_cases=sample_test_cases,
        save_results=True,
        output_dir=temp_output_dir
    )

    # Run experiment
    runner = AsyncExperimentRunner(config)
    results = await runner.run_async()

    # Verify results structure
    assert len(results) == len(sample_prompts) * len(sample_test_cases)

    for result in results:
        assert "prompt_name" in result
        assert "model_name" in result
        assert "test_case_idx" in result
        assert "input" in result
        assert "expected_output" in result
        assert "actual_output" in result
        assert "evaluation_scores" in result
        assert "bleu" in result["evaluation_scores"]
        assert "rouge" in result["evaluation_scores"]

    # Verify summary generation
    summary = runner.get_summary()
    assert len(summary) == len(sample_prompts)

    for key, stats in summary.items():
        prompt_name, model_name = key
        assert prompt_name in ["direct_prompt", "cot_prompt"]
        assert model_name == "test-model-v1"
        assert "scores" in stats
        assert "bleu" in stats["scores"]
        assert "rouge" in stats["scores"]


@pytest.mark.asyncio
async def test_result_storage_and_retrieval(temp_output_dir, sample_prompts, sample_test_cases):
    """Test saving and loading experiment results"""

    # Run experiment with storage
    model = MockModelBackend("storage-test-model")
    evaluators = [BLEUEvaluator()]

    config = ExperimentConfig(
        name="storage_test_experiment",
        prompts=sample_prompts,
        models=[model],
        evaluators=evaluators,
        test_cases=sample_test_cases,
        save_results=True,
        output_dir=temp_output_dir
    )

    runner = AsyncExperimentRunner(config)
    original_results = await runner.run_async()

    # Load results using storage
    storage = ResultStorage(temp_output_dir)
    loaded_results = storage.load_results("storage_test_experiment")

    # Verify loaded results match original
    assert len(loaded_results) == len(original_results)

    for orig, loaded in zip(original_results, loaded_results):
        assert orig["prompt_name"] == loaded["prompt_name"]
        assert orig["model_name"] == loaded["model_name"]
        assert orig["actual_output"] == loaded["actual_output"]
        assert orig["evaluation_scores"] == loaded["evaluation_scores"]


@pytest.mark.asyncio
async def test_multi_model_comparison(temp_output_dir, sample_prompts, sample_test_cases):
    """Test comparing multiple models"""

    # Setup multiple models
    model1 = MockModelBackend("model-a")
    model2 = MockModelBackend("model-b")
    evaluators = [BLEUEvaluator(), ROUGEEvaluator()]

    config = ExperimentConfig(
        name="multi_model_experiment",
        prompts=[sample_prompts[0]],  # Use just one prompt
        models=[model1, model2],
        evaluators=evaluators,
        test_cases=sample_test_cases,
        save_results=True,
        output_dir=temp_output_dir
    )

    runner = AsyncExperimentRunner(config)
    results = await runner.run_async()

    # Verify results for both models
    model_names = {r["model_name"] for r in results}
    assert model_names == {"model-a", "model-b"}

    # Verify summary has entries for both models
    summary = runner.get_summary()
    summary_models = {key[1] for key in summary.keys()}
    assert summary_models == {"model-a", "model-b"}


@pytest.mark.asyncio
async def test_experiment_with_comparator(temp_output_dir, sample_prompts, sample_test_cases):
    """Test using ResultComparator to find best configurations"""

    model = MockModelBackend("comparator-test")
    evaluators = [BLEUEvaluator(), ROUGEEvaluator()]

    config = ExperimentConfig(
        name="comparator_test",
        prompts=sample_prompts,
        models=[model],
        evaluators=evaluators,
        test_cases=sample_test_cases,
        save_results=True,
        output_dir=temp_output_dir
    )

    runner = AsyncExperimentRunner(config)
    results = await runner.run_async()

    # Use comparator to find best prompt
    from prompt_sandbox.experiments.comparator import ResultComparator
    comparator = ResultComparator(results)

    best_prompt, score = comparator.get_best_prompt("comparator-test", "bleu")
    assert best_prompt in ["direct_prompt", "cot_prompt"]
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_prompt_template_rendering(sample_prompts):
    """Test prompt template variable substitution"""

    for prompt in sample_prompts:
        rendered = prompt.render(question="What is 5+5?")
        assert "What is 5+5?" in rendered
        assert "{{question}}" not in rendered

    # Test direct vs COT prompts have different templates
    direct_rendered = sample_prompts[0].render(question="Test")
    cot_rendered = sample_prompts[1].render(question="Test")
    assert direct_rendered != cot_rendered
    assert "step by step" in cot_rendered
    assert "step by step" not in direct_rendered


@pytest.mark.asyncio
async def test_error_handling_in_experiment(temp_output_dir, sample_test_cases):
    """Test experiment handles errors gracefully"""

    class FailingModel(ModelBackend):
        """Model that fails occasionally"""

        def __init__(self):
            self.model_name = "failing-model"
            self.call_count = 0

        async def generate_async(self, prompt: str, **kwargs) -> GenerationResult:
            self.call_count += 1
            # Fail on first call, succeed on retry
            if self.call_count == 1:
                raise Exception("Simulated API error")

            return GenerationResult(
                prompt=prompt,
                generated_text="Success after retry",
                tokens_generated=3,
                generation_time=0.1,
                model_name=self.model_name
            )

        def generate(self, prompt: str, **kwargs) -> GenerationResult:
            return asyncio.run(self.generate_async(prompt, **kwargs))

    model = FailingModel()
    prompt = PromptTemplate(PromptConfig(
        name="test_prompt",
        template="{{question}}",
        variables=["question"]
    ))

    config = ExperimentConfig(
        name="error_handling_test",
        prompts=[prompt],
        models=[model],
        evaluators=[BLEUEvaluator()],
        test_cases=[sample_test_cases[0]],
        save_results=True,
        output_dir=temp_output_dir
    )

    runner = AsyncExperimentRunner(config)
    results = await runner.run_async()

    # Should succeed after retry
    assert len(results) == 1
    assert results[0]["actual_output"] == "Success after retry"
