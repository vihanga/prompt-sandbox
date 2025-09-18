"""
Quickstart example for Prompt-Sandbox

Demonstrates basic usage of prompt templates and configuration loading.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.loader import ConfigLoader
from prompts.template import PromptTemplate


def main():
    print("=" * 60)
    print("Prompt-Sandbox Quickstart Example")
    print("=" * 60)

    # Load prompt configuration
    config_path = Path(__file__).parent.parent / "configs/prompts/qa_assistant.yaml"

    print(f"\n1. Loading prompt config from: {config_path}")
    prompt_config = ConfigLoader.load_prompt_config(config_path)

    print(f"   ✓ Loaded: {prompt_config.name} v{prompt_config.version}")
    print(f"   Required variables: {prompt_config.variables}")

    # Create prompt template
    print("\n2. Creating prompt template...")
    template = PromptTemplate(prompt_config)
    print(f"   ✓ Template ready: {template}")

    # Render basic prompt
    print("\n3. Rendering prompt with variables...")
    rendered = template.render(question="What is machine learning?")

    print("\n" + "-" * 60)
    print("RENDERED PROMPT:")
    print("-" * 60)
    print(rendered)
    print("-" * 60)

    # Render with system message
    print("\n4. Rendering with system message...")
    full_prompt = template.render_with_system(question="What is reinforcement learning?")

    print("\n" + "-" * 60)
    print("FULL PROMPT (with system):")
    print("-" * 60)
    print(full_prompt)
    print("-" * 60)

    # Render with few-shot examples
    print("\n5. Rendering with few-shot examples...")
    few_shot_prompt = template.render_with_few_shot(
        question="What is the capital of Germany?"
    )

    print("\n" + "-" * 60)
    print("FEW-SHOT PROMPT:")
    print("-" * 60)
    print(few_shot_prompt)
    print("-" * 60)

    print("\n" + "=" * 60)
    print("✓ Quickstart complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
