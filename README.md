# Harness-MU: Model-Agnostic Hard Constraints for Multi-User LLM Agents via Harness Engineering

[Wangxuan Fan](https://github.com/YuanJrShiuan)<sup>1,2</sup>, [Xiaoyu Nie]<sup>1</sup>, [Zhongxiang Dai](https://sds.cuhk.edu.cn/en/teacher/504)<sup>1</sup>

<sup>1</sup> Chinese University of Hong Kong, Shenzhen &nbsp;|&nbsp; <sup>2</sup> Shenzhen Research Institute of Big Data

---

**Harness-MU** is the first model-agnostic, zero-tuning framework for multi-user LLM agents. It decouples language generation from safety orchestration through a four-module runtime harness, guaranteeing unbreakable permission boundaries while maximizing compliant demand satisfaction across heterogeneous users.

## Overview

Contemporary LLMs are trained under a **Single Principal–Agent** paradigm — they optimize for one user at a time. When multiple users interact with a shared agent, three systematic failures emerge: unauthorized information disclosure, chaotic resolution of conflicting instructions, and failure to prioritize high-authority directives. Prompt-based safeguards offer zero deterministic guarantees.

Harness-MU addresses this through **Harness Engineering**: governance constraints (who is authorized, what is restricted, whose instructions take precedence) are enforced by deterministic execution hooks operating outside the LLM boundary, rather than entrusted to the model.

```
                   ┌──────────────┐
   Users ─────────►│  Gatekeeper  │──► frozen permission map
                   └──────┬───────┘
                          │
                   ┌──────▼───────┐
                   │   Mediator   │──► per-user policy plan πₖ
                   └──────┬───────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
   ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
   │  Worker u₁  │ │  Worker u₂  │ │  Worker uₙ  │  (parallel, isolated)
   └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
          │               │               │
          └───────────────┼───────────────┘
                          │
                   ┌──────▼───────────┐
                   │ ComplianceChecker│──► fail-closed projection
                   └──────────────────┘
```

### Four Decoupled Modules

| Module | Role |
|--------|------|
| **Gatekeeper** | Admission control — computes immutable per-user permission records from declared policy before any LLM invocation |
| **Mediator** | Policy orchestration — maps global multi-principal objective into per-user execution plans with authority weights |
| **Workers** | Isolated generation — each user gets a parallel, single-principal context; cross-user history contamination is structurally impossible |
| **ComplianceChecker** | Deterministic projection — post-generation enforcement that fail-closes any response leaking unauthorized content |

Harness-MU instantiates the six-component harness taxonomy formalized by [Meng et al. (2026)](https://arxiv.org/abs/2605.xxxxx): Execution Loop, Tool Registry, Context Manager, State Store, Lifecycle Hooks, and Evaluation Interface — specialized for multi-principal governance.

## Key Results

Evaluated on the [Muses-Bench](https://github.com/Kordi-AI/Multi-User-LLM-Agent) benchmark across four frontier models (Deepseek, Qwen, Gemini, GPT):

- **100% privacy preservation** across all adversarial access-control attack vectors (vs. 0.926–0.995 baseline)
- **Overall score improvement** of +0.240 to +0.427 across model families
- **Instruction-following accuracy** improved by up to +48.9 percentage points (GPT, aligned)
- **Inverse scaling**: the ComplianceChecker correction loop activates most where the base model is weakest

## Repository Structure

```
Harness_MulUser/
├── harness_mu/                  # Harness-MU core implementation
│   ├── core/                    # Pipeline orchestration & type system
│   ├── modules/                 # Gatekeeper, Mediator, Worker, ComplianceChecker
│   ├── adapters/                # Scenario adapters (access_control, instruction_following)
│   ├── llm/                     # LLM client factory & provider wrappers
│   ├── config/                  # YAML configs & schema
│   ├── prompts/                 # Prompt templates
│   ├── metrics/                 # Metrics collection & custom evaluators
│   ├── utils/                   # Utilities (formatting, file I/O, timer)
│   ├── validation/              # Instruction checkers
│   ├── data/                    # Sample datasets for testing
│   ├── run.py                   # Main entry point
│   └── __main__.py              # python -m harness_mu entry
├── Multi-User-LLM-Agent/        # Muses-Bench (extended)
│   ├── muses_bench/             # Benchmark framework
│   │   ├── agents/              # SimulatedUser agent
│   │   ├── evaluators/          # AC, IF, meeting_scheduling evaluators
│   │   ├── envs/                # Conversation environments
│   │   └── utils/               # LLM utilities with metrics instrumentation
│   ├── run.py                   # Muses-Bench entry point
│   └── scripts/                 # Batch submission & evaluation scripts
├── LICENSE
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.10+
- An OpenAI-compatible API endpoint (OpenRouter, DeepSeek, MAAS, OpenAI, etc.)

### Installation

```bash
git clone https://github.com/YuanJrShiuan/Harness_MulUser.git
cd Harness_MulUser

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r harness_mu/requirements.txt 2>/dev/null || pip install litellm openai pyyaml httpx
```

### Configuration

```bash
cp .env.example .env
# Edit .env with your API credentials:
#   OPENAI_BASE_URL=https://openrouter.ai/api/v1
#   OPENAI_API_KEY=your-key-here
```

### Running Experiments

```bash
# Harness-MU access control (live mode)
python -m harness_mu --config harness_mu/config/ac_live_deepseek.yaml

# Harness-MU instruction following
python -m harness_mu --config harness_mu/config/if_aligned_deepseek.yaml

# Muses-Bench baseline (from Multi-User-LLM-Agent/)
cd Multi-User-LLM-Agent && python run.py
```

## Scenarios & Metrics

### Access Control (AC)

Tests whether the system maintains permission boundaries under multi-turn adversarial pressure. Crosses 3 serialization templates (colon, says, XML) × 4 attack vectors (none, fake_authorized, pressure, roleplaying) = 12 datasets.

| Metric | Definition |
|--------|------------|
| **Privacy** | 1 − leakage rate (fraction of unauthorized users who receive sensitive tokens) |
| **Utility** | Fraction of authorized users who successfully receive requested resources |
| **Overall** | Harmonic mean of Privacy and Utility |

### Instruction Following (IF)

Tests arbitration among conflicting user instructions under an authority hierarchy.

| Metric | Definition |
|--------|------------|
| **Accuracy** | Fraction of atomic instruction constraints satisfied across evaluation targets |
| **W.Acc** | Authority-weighted accuracy |
| **Per-User Acc** | Macro-averaged accuracy across individual users |

## Citation

If you use Harness-MU in your research, please cite:

```bibtex
@article{fan2026harnessmu,
  title={Harness-{MU}: Model-Agnostic Hard Constraints for Multi-User {LLM} Agents
         via Harness Engineering},
  author={Fan, Wangxuan and Nie, Xiaoyu and Dai, Zhongxiang},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

Harness-MU builds upon and extends [Muses-Bench](https://github.com/Kordi-AI/Multi-User-LLM-Agent) (Yang et al., 2026) for multi-principal evaluation, [IHEval](https://github.com/asafpr/IHEval) for deterministic instruction-following checking, and the agent harness taxonomy formalized by [Meng et al. (2026)](https://arxiv.org/abs/2605.xxxxx).
