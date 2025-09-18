"""
Prompt template rendering with Jinja2
"""

from jinja2 import Template, TemplateSyntaxError, UndefinedError
from typing import Dict, Any, List, Optional
from ..config.schema import PromptConfig, FewShotExample


class PromptTemplate:
    """
    Handles prompt rendering with variable substitution and few-shot examples
    """

    def __init__(self, config: PromptConfig):
        """
        Initialize prompt template from configuration

        Args:
            config: Validated PromptConfig object
        """
        self.config = config
        self.name = config.name
        self.version = config.version

        try:
            self.template = Template(config.template)
        except TemplateSyntaxError as e:
            raise ValueError(f"Invalid Jinja2 template: {e}")

    def render(self, **kwargs: Any) -> str:
        """
        Render prompt with provided variables

        Args:
            **kwargs: Variable key-value pairs

        Returns:
            Rendered prompt string

        Raises:
            ValueError: If required variables are missing
        """
        # Check for required variables
        missing_vars = set(self.config.variables) - set(kwargs.keys())
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")

        try:
            rendered = self.template.render(**kwargs)
        except UndefinedError as e:
            raise ValueError(f"Template rendering error: {e}")

        return rendered.strip()

    def render_with_system(self, **kwargs: Any) -> str:
        """
        Render complete prompt with system message (if defined)

        Args:
            **kwargs: Variable key-value pairs

        Returns:
            Full prompt string with system message prepended
        """
        user_prompt = self.render(**kwargs)

        if self.config.system:
            return f"{self.config.system}\n\n{user_prompt}"

        return user_prompt

    def render_with_few_shot(
        self, include_examples: bool = True, **kwargs: Any
    ) -> str:
        """
        Render prompt with few-shot examples (if defined)

        Args:
            include_examples: Whether to include few-shot examples
            **kwargs: Variable key-value pairs

        Returns:
            Prompt with few-shot examples prepended
        """
        prompt_parts = []

        # Add system message
        if self.config.system:
            prompt_parts.append(self.config.system)

        # Add few-shot examples
        if include_examples and self.config.few_shot_examples:
            examples_text = self._format_few_shot_examples(
                self.config.few_shot_examples
            )
            prompt_parts.append(examples_text)

        # Add user prompt
        user_prompt = self.render(**kwargs)
        prompt_parts.append(user_prompt)

        return "\n\n".join(prompt_parts)

    def _format_few_shot_examples(self, examples: List[FewShotExample]) -> str:
        """
        Format few-shot examples for inclusion in prompt

        Args:
            examples: List of FewShotExample objects

        Returns:
            Formatted examples string
        """
        formatted = ["Here are some examples:\n"]

        for i, example in enumerate(examples, 1):
            formatted.append(f"Example {i}:")
            formatted.append(f"Input: {example.input}")
            formatted.append(f"Output: {example.output}")
            formatted.append("")  # Blank line between examples

        return "\n".join(formatted)

    def get_required_variables(self) -> List[str]:
        """Get list of required template variables"""
        return self.config.variables

    def validate_variables(self, **kwargs: Any) -> bool:
        """
        Check if all required variables are provided

        Args:
            **kwargs: Variable key-value pairs

        Returns:
            True if all required variables present, False otherwise
        """
        return all(var in kwargs for var in self.config.variables)

    def __repr__(self) -> str:
        return f"PromptTemplate(name='{self.name}', version='{self.version}')"
