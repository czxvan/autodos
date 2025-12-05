# AutoDoS - Automated Denial of Service Attack

A clean async reimplementation of [AutoDoS](https://github.com/shuita2333/AutoDoS), built on AnyLLM with multi-agent collaboration.

## Quick Start

```bash
# Run attack with default config
python main.py

# Use custom config
python main.py --config configs/deepseek_official.yaml

# With API key
export DEEPSEEK_API_KEY="your-key"
python main.py --config configs/deepseek_official.yaml
```

**Note:** Free APIs (gpt4free) may be unstable. Official APIs recommended for production.

## Configuration

```yaml
attack:
  n_subproblems: 5              # Range: 5-50
  question_length: 200          # Range: 50-1000
  optimize_iterations: 3        # Range: 1-50
  n_optimization_streams: 2     # Range: 1-10
  max_concurrent_requests: 8    # Range: 1-10

agents:
  target:
    backend_type: "gpt4free"
    model: "deepseek-v3"
    provider: "DeepInfra"
    max_tokens: 2048
```

See `configs/` for more examples.

## Architecture

```
AutoDoSAttack
├── TargetAgent           # Target system interaction
├── DeepBacktrackingAgent # Problem decomposition
├── BreadthExpansionAgent # Subproblem expansion
├── OptimizeAgent         # Prompt optimization
└── JudgeAgent            # Response evaluation
```

All agents built on `anyllm.AsyncClient`, supporting OpenAI, gpt4free, vLLM, and more.

## Key Features

- **Async-first**: Full asyncio with semaphore-based concurrency control
- **Robust parsing**: json-repair for malformed JSON responses
- **Smart retries**: 5 attempts with exponential backoff, 180s timeout
- **Clean logging**: Separate INFO/DEBUG levels for clarity
- **Auto-saving**: Results, history, and summaries saved automatically
- **Minimal design**: Direct AnyLLM usage, no extra wrappers
- **Type safety**: TypedDict + dataclass + Pydantic validation

## Project Structure

```
autodos/
├── main.py               # CLI entry point
├── autodos/
│   ├── config.py         # Configuration & types
│   ├── attack.py         # Attack orchestration
│   └── agents/           # Agent implementations
└── configs/              # YAML configs
```

## Improvements over Original

- **Simplified codebase**: ~500 lines vs. 2000+ lines
- **Pure async**: No sync wrappers, native asyncio throughout
- **Better concurrency**: Semaphore-based dynamic control vs. static batching
- **Cleaner types**: Minimal type system (TypedDict + dataclass + Pydantic)
- **Robust parsing**: json-repair instead of manual regex fixes
- **Production-ready**: CLI, logging, auto-saving, error handling


## License

MIT License. See original [AutoDoS](https://github.com/shuita2333/AutoDoS) for reference.
