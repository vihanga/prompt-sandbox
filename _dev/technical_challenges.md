# Technical Challenges & Solutions

Documentation of problems encountered during development and how they were solved.

---

## Challenge 1: Template Variable Validation

**Problem**: How to ensure that variables declared in config are actually used in the template, and vice versa?

**Solution**: Implemented a Pydantic validator that:
1. Parses the Jinja2 template to extract variable names using regex
2. Compares declared variables with template variables
3. Raises validation error if mismatch

**Code**:
```python
@validator("template")
def validate_template(cls, v, values):
    import re
    jinja_vars = set(re.findall(r"{{\s*(\w+)\s*}}", v))
    declared_vars = set(values.get("variables", []))

    unused_vars = declared_vars - jinja_vars
    if unused_vars:
        raise ValueError(f"Declared variables not used: {unused_vars}")

    return v
```

**Lessons Learned**: Validate early, fail fast. Better to catch config errors at load time than render time.

---

## Challenge 2: Optional Variables in Templates

**Problem**: Some variables are optional (e.g., context in Q&A), but Jinja2 {% if %} blocks make variable detection harder.

**Solution**:
- Don't require variables used only in conditionals to be declared
- Document that required variables should always be rendered
- Future: Could parse Jinja2 AST for more accurate detection

**Status**: Acceptable for now, may revisit with proper AST parsing

---

## Challenge 3: Few-Shot Example Formatting

**Problem**: How to format few-shot examples consistently across different prompt types?

**Solution**: Created structured `FewShotExample` Pydantic model with input/output fields. Template method `_format_few_shot_examples()` handles formatting.

**Alternative Considered**: Free-form string examples
- Rejected because: Less structured, harder to validate, no type safety

---

## Future Challenges to Address

### 1. Model Loading Performance
**Issue**: Loading multiple LLMs is memory-intensive
**Potential Solutions**:
- Model quantization (int8, int4)
- Model offloading to CPU/disk
- vLLM for efficient serving
- Lazy loading with context managers

### 2. Evaluation Speed
**Issue**: Running evaluations on large datasets is slow
**Potential Solutions**:
- Async/parallel evaluation
- Batch inference
- Caching results
- Sampling strategies

### 3. Prompt Versioning
**Issue**: How to track prompt evolution over time?
**Potential Solutions**:
- Git-based versioning (configs in git)
- Database with version history
- Prompt registry with metadata
- Integration with MLflow/W&B

---

## Best Practices Discovered

1. **Configuration Validation**: Always validate configs at load time
2. **Type Safety**: Use Pydantic models everywhere for type safety
3. **Separation of Concerns**: Keep config, rendering, and evaluation separate
4. **Documentation**: Write docstrings immediately while context is fresh
5. **Examples**: Include working examples from day 1

---

## Things That Worked Well

- ✅ Pydantic for schemas - caught many bugs early
- ✅ YAML for configs - very readable, easy to edit
- ✅ Jinja2 for templates - powerful and familiar
- ✅ Early example (quickstart.py) - helps validate design

## Things to Improve

- ⚠️ Need more comprehensive error messages
- ⚠️ Could use better template syntax validation
- ⚠️ Missing logging infrastructure
- ⚠️ No tests yet (need to add)
