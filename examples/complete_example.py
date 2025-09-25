"""
Complete example: Run experiments comparing prompts across models
"""

import asyncio
from pathlib import Path

from prompt_sandbox.config.schema import PromptConfig
from prompt_sandbox.prompts.template import PromptTemplate
from prompt_sandbox.models import OllamaBackend
from prompt_sandbox.evaluators import BLEUEvaluator, ROUGEEvaluator
from prompt_sandbox.experiments import AsyncExperimentRunner, ExperimentConfig


async def main():
    """Run a complete prompt engineering experiment"""

    print("🚀 Prompt Engineering Experiment\n")

    # Define prompts to test
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

    # Setup model (requires Ollama running)
    model = OllamaBackend("llama3.1")

    # Setup evaluators
    evaluators = [
        BLEUEvaluator(),
        ROUGEEvaluator()
    ]

    # Define test cases
    test_cases = [
        {
            "input": {"question": "What is 2+2?"},
            "expected_output": "4"
        },
        {
            "input": {"question": "What is the capital of France?"},
            "expected_output": "Paris"
        }
    ]

    # Configure experiment
    config = ExperimentConfig(
        name="prompt_comparison",
        prompts=[prompt1, prompt2],
        models=[model],
        evaluators=evaluators,
        test_cases=test_cases,
        save_results=True,
        output_dir=Path("results")
    )

    # Run experiment
    runner = AsyncExperimentRunner(config)
    results = await runner.run_async()

    # Get summary
    summary = runner.get_summary()
    print("\n📊 Summary:")
    for key, stats in summary.items():
        print(f"\n{key[0]} + {key[1]}:")
        print(f"  BLEU: {stats['scores']['bleu']['mean']:.2f}")
        print(f"  ROUGE: {stats['scores']['rouge']['mean']:.2f}")

    # Compare and find winner
    from prompt_sandbox.experiments.comparator import ResultComparator
    comparator = ResultComparator(results)

    best_prompt, score = comparator.get_best_prompt("llama3.1", "bleu")
    print(f"\n🏆 Best prompt: {best_prompt} (BLEU: {score:.2f})")


if __name__ == "__main__":
    asyncio.run(main())
