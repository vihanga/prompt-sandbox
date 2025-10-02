"""
Pydantic schemas for configuration validation
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Dict, Optional, Any
from pathlib import Path


class GenerationConfig(BaseModel):
    """Configuration for text generation"""

    model_config = ConfigDict(extra="forbid")

    max_new_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    num_return_sequences: int = Field(default=1, ge=1, le=10)
    do_sample: bool = Field(default=True)
    repetition_penalty: float = Field(default=1.0, ge=1.0, le=2.0)


class FewShotExample(BaseModel):
    """Few-shot learning example"""

    input: str = Field(..., min_length=1)
    output: str = Field(..., min_length=1)
    metadata: Optional[Dict[str, Any]] = Field(default=None)


class PromptConfig(BaseModel):
    """Schema for prompt configuration files"""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Unique prompt identifier", min_length=1)
    version: str = Field(default="1.0", description="Prompt version")

    system: Optional[str] = Field(None, description="System message")
    role: str = Field(default="assistant", description="Role identifier")

    template: str = Field(..., description="Jinja2 template string", min_length=1)
    variables: List[str] = Field(default=[], description="Required template variables")

    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata (tags, author, etc.)"
    )

    few_shot_examples: Optional[List[FewShotExample]] = Field(
        default=None, description="Few-shot learning examples"
    )

    generation_config: Optional[GenerationConfig] = Field(
        default=None, description="Default generation parameters"
    )

    @field_validator("template")
    @classmethod
    def validate_template(cls, v, info):
        """Validate that template contains required variable placeholders"""
        import re

        # Extract Jinja2 variables from template
        jinja_vars = set(re.findall(r"{{\s*(\w+)\s*}}", v))

        # Check if declared variables are actually used
        declared_vars = set(info.data.get("variables", []))

        # All declared variables should be in template
        unused_vars = declared_vars - jinja_vars
        if unused_vars:
            raise ValueError(
                f"Declared variables not used in template: {unused_vars}"
            )

        return v


class ModelConfig(BaseModel):
    """Configuration for model loading"""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Model identifier (HF model name or path)")
    backend: str = Field(
        default="huggingface",
        description="Backend to use",
        pattern="^(huggingface|vllm|llamacpp|openai)$",
    )
    device: str = Field(default="auto", description="Device placement")
    dtype: str = Field(
        default="float16", description="Data type", pattern="^(float16|float32|int8)$"
    )
    load_in_8bit: bool = Field(default=False, description="Use 8-bit quantization")
    cache_dir: Optional[Path] = Field(default=None, description="Model cache directory")


class EvaluatorConfig(BaseModel):
    """Configuration for evaluators"""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Evaluator name")
    type: str = Field(
        ...,
        description="Evaluator type",
        pattern="^(bleu|bertscore|rouge|faithfulness|perplexity)$",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict, description="Evaluator-specific parameters"
    )


class DatasetConfig(BaseModel):
    """Configuration for evaluation datasets"""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Dataset identifier")
    path: Path = Field(..., description="Path to dataset file (JSONL format)")
    num_samples: Optional[int] = Field(
        default=None, description="Number of samples to use (None = all)"
    )
    shuffle: bool = Field(default=False, description="Shuffle dataset before sampling")
    seed: int = Field(default=42, description="Random seed for shuffling")

    @field_validator("path")
    @classmethod
    def validate_path_exists(cls, v):
        """Check if dataset file exists"""
        if not v.exists():
            raise ValueError(f"Dataset file not found: {v}")
        return v


class ExperimentConfig(BaseModel):
    """Configuration for experiment runs"""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Experiment name")
    description: Optional[str] = Field(None, description="Experiment description")

    prompt_configs: List[Path] = Field(
        ..., description="Paths to prompt YAML files", min_length=1
    )
    model_configs: List[ModelConfig] = Field(
        ..., description="Models to evaluate", min_length=1
    )
    dataset_config: DatasetConfig = Field(..., description="Evaluation dataset")
    evaluators: List[EvaluatorConfig] = Field(
        ..., description="Evaluation metrics", min_length=1
    )

    batch_size: int = Field(default=8, ge=1, description="Batch size for inference")
    max_workers: int = Field(
        default=4, ge=1, description="Max parallel workers for async execution"
    )

    save_outputs: bool = Field(
        default=True, description="Save generated outputs to file"
    )
    output_dir: Path = Field(default=Path("results"), description="Output directory")

    cache_enabled: bool = Field(
        default=True, description="Enable caching of inference results"
    )

    @field_validator("prompt_configs")
    @classmethod
    def validate_prompt_paths(cls, v):
        """Check if all prompt config files exist"""
        for path in v:
            if not path.exists():
                raise ValueError(f"Prompt config file not found: {path}")
        return v
