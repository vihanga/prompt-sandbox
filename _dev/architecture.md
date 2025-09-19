# Prompt-Sandbox Development Documentation

**Last Updated**: January 2025
**Status**: Active Development
**Lead**: Vihanga Gamage

---

## 🎯 Project Vision

Build a production-ready framework for systematic prompt engineering that enables:
1. **Rapid experimentation** with prompt variations
2. **Objective evaluation** across multiple LLMs
3. **Reproducible results** through config-driven workflows
4. **Insight discovery** through comparative analysis

---

## 🏗️ Architecture Overview

### High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                     Prompt-Sandbox                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │   Config     │───▶│  Experiment  │───▶│  Evaluation  │ │
│  │   Manager    │    │   Runner     │    │   Engine     │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│         │                    │                    │        │
│         ▼                    ▼                    ▼        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │    YAML      │    │    Model     │    │   Metrics    │ │
│  │   Prompts    │    │   Loaders    │    │  (BLEU,      │ │
│  │              │    │   (Hugging   │    │   BERTScore, │ │
│  │              │    │    Face)     │    │   Custom)    │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           Results Dashboard & Comparison            │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

#### 1. Config Manager
- **Purpose**: Load and validate YAML prompt configurations
- **Key Features**:
  - Schema validation (Pydantic models)
  - Variable interpolation
  - Config inheritance/composition
  - Version tracking

#### 2. Experiment Runner
- **Purpose**: Orchestrate multi-model inference
- **Key Features**:
  - Async/parallel execution
  - Progress tracking
  - Error handling & retry logic
  - Caching for expensive operations

#### 3. Model Loaders
- **Purpose**: Abstract model loading/inference
- **Supported Backends**:
  - Hugging Face Transformers
  - vLLM (optional, for speed)
  - Llama.cpp (optional, for quantized models)
  - OpenAI API (optional, for GPT comparisons)

#### 4. Evaluation Engine
- **Purpose**: Score outputs against ground truth/references
- **Metrics**:
  - **BLEU**: N-gram overlap (translation quality)
  - **BERTScore**: Semantic similarity via BERT embeddings
  - **ROUGE**: Recall-oriented summarization metric
  - **Faithfulness**: Custom NLI-based factual consistency
  - **Perplexity**: Model confidence (internal metric)

---

## 📂 Directory Structure (Detailed)

```
prompt-sandbox/
├── src/
│   └── prompt_sandbox/        # Main package (snake_case)
│       ├── __init__.py
│       ├── config/
│       │   ├── __init__.py
│       │   ├── schema.py          # Pydantic models for validation
│       │   ├── loader.py          # YAML loading with Hydra
│       │   └── validator.py       # Config consistency checks
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py            # Abstract model interface
│       │   ├── hf_model.py        # Hugging Face implementation
│       │   ├── vllm_model.py      # vLLM implementation (optional)
│       │   └── loader.py          # Model factory pattern
│       ├── prompts/
│       │   ├── __init__.py
│       │   ├── template.py        # Jinja2 template rendering
│       │   ├── stack.py           # System+role+content composition
│       │   └── variables.py       # Variable injection logic
│       ├── evaluators/
│       │   ├── __init__.py
│       │   ├── base.py            # Abstract evaluator
│       │   ├── bleu.py            # BLEU score implementation
│       │   ├── bertscore.py       # BERTScore wrapper
│       │   ├── rouge.py           # ROUGE metrics
│       │   ├── faithfulness.py    # NLI-based checker
│       │   └── perplexity.py      # Model-intrinsic metric
│       ├── experiments/
│       │   ├── __init__.py
│       │   ├── runner.py          # Main experiment orchestration
│       │   ├── results.py         # Results data models
│       │   └── comparison.py      # Side-by-side analysis
│       └── utils/
│           ├── __init__.py
│           ├── logging.py         # Structured logging setup
│           ├── caching.py         # Result caching (joblib)
│           └── metrics.py         # Metric aggregation helpers
├── configs/
│   ├── prompts/
│   │   ├── qa_assistant.yaml
│   │   ├── summarization.yaml
│   │   ├── code_generation.yaml
│   │   └── chain_of_thought.yaml
│   ├── experiments/
│   │   ├── truthfulqa_eval.yaml
│   │   ├── gsm8k_eval.yaml
│   │   └── custom_benchmark.yaml
│   └── models/
│       ├── llama2_7b.yaml
│       ├── mistral_7b.yaml
│       └── phi2.yaml
├── tests/
│   ├── unit/
│   │   ├── test_config.py
│   │   ├── test_prompts.py
│   │   ├── test_evaluators.py
│   │   └── test_models.py
│   ├── integration/
│   │   ├── test_experiment.py
│   │   └── test_end_to_end.py
│   └── fixtures/
│       ├── sample_prompts.yaml
│       └── sample_outputs.json
├── examples/
│   ├── quickstart.py
│   ├── custom_evaluator.py
│   ├── batch_comparison.py
│   └── notebooks/
│       ├── 01_basic_usage.ipynb
│       ├── 02_custom_metrics.ipynb
│       └── 03_advanced_prompting.ipynb
├── scripts/
│   ├── download_datasets.py
│   ├── run_benchmark.py
│   └── generate_report.py
├── docs/
│   ├── architecture.md
│   ├── api_reference.md
│   ├── prompt_design_guide.md
│   └── evaluation_metrics.md
├── pyproject.toml
├── README.md
└── _dev/                      # Public development docs (in git)
    ├── architecture.md (this file)
    ├── implementation_notes.md
    ├── technical_challenges.md
    └── api_reference.md
```

---

## 🔧 Technical Implementation Details

### 1. YAML Prompt Configuration

**Schema Design** (Pydantic):

```python
# src/prompt_sandbox/config/schema.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class PromptConfig(BaseModel):
    """Schema for prompt configuration files"""

    name: str = Field(..., description="Unique prompt identifier")
    version: str = Field(default="1.0", description="Prompt version")

    system: Optional[str] = Field(None, description="System message")
    role: str = Field(default="assistant", description="Role identifier")

    template: str = Field(..., description="Jinja2 template string")
    variables: List[str] = Field(default=[], description="Required variables")

    metadata: Optional[Dict[str, any]] = Field(
        default={},
        description="Additional metadata (tags, author, etc.)"
    )

    few_shot_examples: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Few-shot learning examples"
    )
```

**Example YAML**:

```yaml
# configs/prompts/qa_assistant.yaml
name: "qa_assistant_v1"
version: "1.0"

system: |
  You are a helpful AI assistant that provides accurate, concise answers.
  Always cite sources when possible and admit uncertainty when appropriate.

role: "assistant"

template: |
  Question: {{ question }}

  {% if context %}
  Context: {{ context }}
  {% endif %}

  Please provide a clear, factual answer.

variables:
  - question
  - context  # Optional

metadata:
  tags: ["qa", "general"]
  author: "portfolio"
  created: "2025-01-08"

few_shot_examples:
  - question: "What is the capital of France?"
    answer: "Paris is the capital of France."
  - question: "Who wrote Hamlet?"
    answer: "William Shakespeare wrote Hamlet."
```

### 2. Model Abstraction Layer

**Interface Design**:

```python
# src/prompt_sandbox/models/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    num_return_sequences: int = 1
    do_sample: bool = True

class BaseModel(ABC):
    """Abstract base class for all LLM backends"""

    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = device

    @abstractmethod
    def load(self) -> None:
        """Load model into memory"""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: GenerationConfig
    ) -> List[str]:
        """Generate text from prompt"""
        pass

    @abstractmethod
    def get_perplexity(self, text: str) -> float:
        """Calculate perplexity of text"""
        pass

    def unload(self) -> None:
        """Free model from memory"""
        pass
```

**Hugging Face Implementation**:

```python
# src/prompt_sandbox/models/hf_model.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base import BaseModel, GenerationConfig

class HuggingFaceModel(BaseModel):
    """Hugging Face Transformers implementation"""

    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )

    def generate(self, prompt: str, config: GenerationConfig) -> List[str]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            num_return_sequences=config.num_return_sequences,
            do_sample=config.do_sample,
            pad_token_id=self.tokenizer.eos_token_id
        )

        return [
            self.tokenizer.decode(out, skip_special_tokens=True)
            for out in outputs
        ]

    def get_perplexity(self, text: str) -> float:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])

        return torch.exp(outputs.loss).item()
```

### 3. Evaluation Metrics Implementation

**Faithfulness Checker** (Advanced):

```python
# src/prompt_sandbox/evaluators/faithfulness.py
from transformers import pipeline
from .base import BaseEvaluator

class FaithfulnessEvaluator(BaseEvaluator):
    """
    Uses Natural Language Inference (NLI) to check if generated
    text is entailed by the source context.
    """

    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        self.nli_pipeline = pipeline(
            "text-classification",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1
        )

    def evaluate(
        self,
        generated_text: str,
        context: str,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Compute faithfulness score.

        Returns:
            - score: Float [0, 1], higher = more faithful
            - label: "entailment" | "neutral" | "contradiction"
            - confidence: Model confidence
        """
        # Split generated text into sentences
        sentences = self._split_sentences(generated_text)

        scores = []
        for sent in sentences:
            result = self.nli_pipeline(
                f"{context} [SEP] {sent}",
                return_all_scores=True
            )[0]

            # Find entailment score
            entailment_score = next(
                r["score"] for r in result if r["label"] == "ENTAILMENT"
            )
            scores.append(entailment_score)

        avg_score = sum(scores) / len(scores) if scores else 0.0

        return {
            "score": avg_score,
            "label": "faithful" if avg_score >= threshold else "unfaithful",
            "confidence": avg_score,
            "sentence_scores": scores
        }

    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting (can be improved with spaCy)"""
        import re
        return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
```

### 4. Experiment Orchestration

**Async Execution Pattern**:

```python
# src/prompt_sandbox/experiments/runner.py
import asyncio
from typing import List, Dict
from dataclasses import dataclass
from ..models import BaseModel
from ..evaluators import BaseEvaluator
from ..prompts import PromptTemplate

@dataclass
class ExperimentConfig:
    """Configuration for an experiment run"""
    prompt_configs: List[str]  # Paths to prompt YAML files
    models: List[str]  # Model identifiers
    eval_dataset: str  # Path to evaluation dataset
    evaluators: List[BaseEvaluator]
    num_samples: int = 100
    batch_size: int = 8

class ExperimentRunner:
    """Orchestrates multi-model, multi-prompt experiments"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = []

    async def run(self) -> Dict[str, Any]:
        """Execute experiment with async parallel execution"""

        # Load all components
        prompts = self._load_prompts()
        models = self._load_models()
        dataset = self._load_dataset()

        # Create task matrix: prompts × models × samples
        tasks = []
        for prompt in prompts:
            for model in models:
                for sample in dataset[:self.config.num_samples]:
                    tasks.append(
                        self._run_single(prompt, model, sample)
                    )

        # Execute with concurrency limit
        results = await self._execute_batched(tasks, self.config.batch_size)

        # Aggregate and analyze
        return self._analyze_results(results)

    async def _run_single(
        self,
        prompt: PromptTemplate,
        model: BaseModel,
        sample: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute single inference + evaluation"""

        # Render prompt with sample variables
        rendered_prompt = prompt.render(**sample)

        # Generate
        outputs = model.generate(rendered_prompt)

        # Evaluate
        scores = {}
        for evaluator in self.config.evaluators:
            scores[evaluator.name] = evaluator.evaluate(
                generated=outputs[0],
                reference=sample.get("reference", ""),
                context=sample.get("context", "")
            )

        return {
            "prompt_name": prompt.name,
            "model_name": model.model_name,
            "sample_id": sample["id"],
            "output": outputs[0],
            "scores": scores
        }

    async def _execute_batched(
        self,
        tasks: List,
        batch_size: int
    ) -> List[Dict]:
        """Execute tasks in batches to control memory"""
        results = []

        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch)
            results.extend(batch_results)

            # Log progress
            print(f"Completed {len(results)}/{len(tasks)} tasks")

        return results
```

---

## 📊 Evaluation Strategy

### Metrics Selection Guide

| Metric | Best For | Pros | Cons |
|--------|----------|------|------|
| BLEU | Translation, exact match | Fast, established | Ignores semantics |
| BERTScore | Semantic similarity | Context-aware | Slower, requires BERT |
| ROUGE | Summarization | Recall-focused | N-gram based |
| Faithfulness | Factual accuracy | Catches hallucinations | Requires context |
| Perplexity | Model confidence | Intrinsic | Not human-aligned |

### Benchmark Datasets

1. **TruthfulQA** (817 questions)
   - Tests model truthfulness
   - Categories: health, law, science, politics
   - Metric: % truthful + informative answers

2. **GSM8K** (8.5K grade school math problems)
   - Tests reasoning ability
   - Metric: Exact match accuracy

3. **MMLU** (Massive Multitask Language Understanding)
   - 57 subjects across STEM, humanities, social sciences
   - Metric: Multiple-choice accuracy

---

## 🚀 Development Phases

### Phase 1: Core Infrastructure (Week 1)
- [x] Directory structure
- [ ] Pydantic schemas for configs
- [ ] YAML loader with Hydra
- [ ] Base model interface
- [ ] HuggingFace model implementation
- [ ] Basic prompt rendering (Jinja2)

### Phase 2: Evaluation Suite (Week 2)
- [ ] BLEU evaluator
- [ ] BERTScore evaluator
- [ ] Faithfulness evaluator
- [ ] Aggregated metrics reporting
- [ ] Unit tests for evaluators

### Phase 3: Experiment System (Week 3)
- [ ] Experiment runner with async
- [ ] Results storage (JSON/SQLite)
- [ ] Progress tracking & logging
- [ ] Caching mechanism
- [ ] Integration tests

### Phase 4: Comparison & Visualization (Week 4)
- [ ] Side-by-side comparison tool
- [ ] Statistical significance tests
- [ ] Visualization dashboard (Plotly)
- [ ] Report generation (PDF/HTML)

---

## 🧪 Testing Strategy

### Unit Tests
- Config validation edge cases
- Prompt rendering with various inputs
- Metric calculations with known examples
- Model interface mocking

### Integration Tests
- End-to-end experiment runs (small dataset)
- Multi-model comparisons
- Error handling & recovery

### Performance Tests
- Async execution speedup verification
- Memory usage monitoring
- Cache hit rate analysis

---

## 📈 Success Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Unit Test Coverage | >85% | 0% |
| Integration Tests | >5 scenarios | 0 |
| Experiment Runtime (100 samples) | <5 min | N/A |
| Memory Usage (3 models loaded) | <16GB | N/A |
| Config Validation Errors | 100% caught | N/A |

---

## 🔮 Future Enhancements

### Short-term (Next 2-4 weeks)
- [ ] vLLM backend for 10x faster inference
- [ ] OpenAI API integration for GPT comparisons
- [ ] Chain-of-thought prompt templates
- [ ] Few-shot learning examples in configs

### Medium-term (1-2 months)
- [ ] Web UI for experiment management (Streamlit)
- [ ] Automatic hyperparameter tuning (Optuna)
- [ ] Multi-turn conversation evaluation
- [ ] Cost estimation for API-based models

### Long-term (3+ months)
- [ ] LoRA fine-tuning integration
- [ ] Custom reward modeling
- [ ] Active learning for prompt discovery
- [ ] Multi-language support

---

## 📚 References & Resources

### Key Papers
1. **"Scaling Instruction-Finetuned Language Models"** (Chung et al., 2022)
   - Flan-T5 prompt design patterns

2. **"Chain-of-Thought Prompting"** (Wei et al., 2022)
   - Advanced prompting techniques

3. **"BERTScore: Evaluating Text Generation with BERT"** (Zhang et al., 2019)
   - Semantic evaluation metrics

### Useful Tools
- **Hugging Face Transformers**: Model library
- **Hydra**: Configuration management
- **Pydantic**: Data validation
- **pytest**: Testing framework

---

## 👥 Development Notes

### Code Style
- **Formatter**: Black (line length: 88)
- **Linter**: Flake8 + mypy (type checking)
- **Docstrings**: Google style
- **Imports**: isort for organization

### Git Workflow
- **Main branch**: `main` (protected)
- **Feature branches**: `feature/prompt-rendering`, `feature/bert-evaluator`
- **Commit messages**: Conventional commits (`feat:`, `fix:`, `docs:`)

### Performance Considerations
- Use `torch.float16` for inference (2x memory reduction)
- Enable `gradient_checkpointing` for large models
- Cache tokenized inputs when possible
- Use `accelerate` for multi-GPU support

---

**Next Steps**: Implement Phase 1 core infrastructure (estimated: 3-5 days)
