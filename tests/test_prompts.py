"""
Unit tests for prompt templates
"""

import pytest
from prompt_sandbox.prompts.template import PromptTemplate
from prompt_sandbox.config.schema import PromptConfig


class TestPromptTemplate:
    """Tests for PromptTemplate"""

    def test_basic_rendering(self):
        """Test basic variable substitution"""
        config = PromptConfig(
            name="test_prompt",
            template="Hello {{name}}!",
            variables=["name"]
        )

        template = PromptTemplate(config)
        result = template.render(name="World")

        assert result == "Hello World!"

    def test_multiple_variables(self):
        """Test multiple variable substitution"""
        config = PromptConfig(
            name="test_prompt",
            template="{{greeting}} {{name}}, you have {{count}} messages",
            variables=["greeting", "name", "count"]
        )

        template = PromptTemplate(config)
        result = template.render(greeting="Hello", name="Alice", count=5)

        assert result == "Hello Alice, you have 5 messages"

    def test_missing_variable_raises_error(self):
        """Test that missing required variables raise error"""
        config = PromptConfig(
            name="test_prompt",
            template="Hello {{name}}!",
            variables=["name"]
        )

        template = PromptTemplate(config)

        with pytest.raises(ValueError, match="Missing required variables"):
            template.render()  # Missing 'name'

    def test_system_message(self):
        """Test rendering with system message"""
        config = PromptConfig(
            name="test_prompt",
            system="You are a helpful assistant.",
            template="User question: {{question}}",
            variables=["question"]
        )

        template = PromptTemplate(config)
        result = template.render_with_system(question="What is 2+2?")

        assert "You are a helpful assistant" in result
        assert "User question: What is 2+2?" in result

    def test_few_shot_examples(self):
        """Test rendering with few-shot examples"""
        from prompt_sandbox.config.schema import FewShotExample

        config = PromptConfig(
            name="test_prompt",
            template="Question: {{question}}\nAnswer:",
            variables=["question"],
            few_shot_examples=[
                FewShotExample(input="What is 2+2?", output="4"),
                FewShotExample(input="What is 3+3?", output="6")
            ]
        )

        template = PromptTemplate(config)
        result = template.render_with_few_shot(question="What is 4+4?")

        assert "Example 1:" in result
        assert "Input: What is 2+2?" in result
        assert "Output: 4" in result
        assert "Question: What is 4+4?" in result
