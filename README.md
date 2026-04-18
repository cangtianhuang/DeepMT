<div align="center">

[![Header](https://capsule-render.vercel.app/api?type=waving&color=0:0f2744,50:1a4b8a,100:4F9BF7&height=220&section=header&text=DeepMT&fontSize=96&fontColor=ffffff&animation=fadeIn&fontAlignY=42&desc=Deep%20Metamorphic%20Testing%20for%20Deep%20Learning%20Frameworks&descAlignY=62&descSize=18&descColor=c8e0ff)](https://github.com/cangtianhuang/DeepMT)

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=500&size=17&pause=1200&color=4F9BF7&center=true&vCenter=true&width=640&lines=Auto-generate+Metamorphic+Relations;Catch+Bugs+in+PyTorch+%2F+TensorFlow+%2F+PaddlePaddle;No+Oracle%3F+No+Problem.;Symbolic+Proof+%2B+LLM+Hypothesis+%2B+Batch+Testing." alt="Typing SVG" />

<table border="0" width="100%">
<tr>
<td width="60%" valign="top">

<!-- Build & quality -->
[![CI](https://github.com/cangtianhuang/DeepMT/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/cangtianhuang/DeepMT/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10_%7C_3.11_%7C_3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-766%20passing-brightgreen)](tests/)

<!-- Framework & tools -->
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D1.9-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-F7DF1E?logo=opensourceinitiative&logoColor=black)](LICENSE)

</td>
<td width="40%" align="center" valign="middle">

```bash
# No LLM needed — 30 sec demo
source .venv/bin/activate
python demo/golden_path.py
```

</td>
</tr>
</table>

[🇨🇳 中文文档](README_CN.md) &nbsp;·&nbsp; [📖 Docs](docs/) &nbsp;·&nbsp; [⚡ Quick Start](#-quick-start) &nbsp;·&nbsp; [🎬 Demo](#-golden-demo-path) &nbsp;·&nbsp; [🏗️ Architecture](#%EF%B8%8F-architecture)

</div>

---

## 🧠 What is DeepMT?

DeepMT is a research system that **automatically discovers and verifies metamorphic relations (MRs)** for deep learning operators, then uses them to stress-test DL framework implementations — catching numerical errors, precision regressions, and cross-framework inconsistencies that traditional test oracles miss.

<table border="0" width="100%">
<tr>
<td width="50%" valign="top">

### The Problem

Testing deep learning frameworks is hard. There is no **ground truth oracle** — you can't simply check if `torch.conv2d(x)` returns the "correct" answer, because correctness itself depends on the implementation under test.

</td>
<td width="50%" valign="top">

### The Solution

**Metamorphic Relations** sidestep the oracle problem by checking *relationships* between outputs. For example:

```
relu(2x) == 2 · relu(x)    ∀ x ≥ 0
```

DeepMT **generates and proves** such relations automatically, then uses them as test oracles.

</td>
</tr>
</table>

### ✨ Key Capabilities

| Capability | Description |
|---|---|
| 🤖 **Auto MR Generation** | LLM hypothesis → template matching → SymPy symbolic proof — 3 layers (operator / model / application) |
| 🧪 **Batch Metamorphic Testing** | Load verified MRs from knowledge base, auto-generate test inputs via `RandomGenerator` (with boundary injection) |
| 🐛 **Fault Injection** | `FaultyPyTorchPlugin` / `FaultyTensorFlowPlugin` inject known bugs — 3-layer mutation evaluation |
| 🔀 **Cross-Framework Consistency** | PyTorch / NumPy / PaddlePaddle / TensorFlow — `--matrix` runs all pairs at once |
| 📊 **Web Dashboard** | 7-page dashboard (frameworks / MR quality / defect cases) via Chart.js + Bootstrap 5 |
| 📦 **Evidence Packs** | Self-contained, copy-paste Python scripts for every detected defect |

---

## 🚀 Quick Start

### Install

<table border="0" width="100%">
<tr>
<td width="33%" valign="top">

**uv (recommended)**
```bash
git clone https://github.com/\
  cangtianhuang/DeepMT.git
cd DeepMT
pip install uv
uv sync --all-extras
source .venv/bin/activate
```

</td>
<td width="33%" valign="top">

**pip**
```bash
git clone https://github.com/\
  cangtianhuang/DeepMT.git
cd DeepMT
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

</td>
<td width="34%" valign="top">

**Docker (GPU)**
```bash
git clone https://github.com/\
  cangtianhuang/DeepMT.git
cd DeepMT
docker build -t deepmt .
docker run --gpus all \
  -e OPENAI_API_KEY=sk-... \
  deepmt deepmt health check
```

</td>
</tr>
</table>

> **CPU-only Docker?** Add `--build-arg PYTORCH_INDEX=https://download.pytorch.org/whl/cpu` to the build command.

### Verify

```bash
deepmt health check
```

<details>
<summary>Expected output ▸</summary>

```
================================================================
DeepMT System Health Report
================================================================
Overall Status: ✅ HEALTHY    Passed: 38  Warnings: 0  Errors: 0
...
All core modules are running normally.
================================================================
```

</details>

---

## 🎬 Golden Demo Path

> No LLM API · No network · Completes in ~30 seconds

```bash
source .venv/bin/activate
PYTHONPATH=$(pwd) python demo/golden_path.py
```

What it covers end-to-end:

```
Step 1  Operator catalog & MR knowledge base ── show verified MRs
Step 2  Normal batch testing ─────────────────── PyTorch baseline (all pass)
Step 3  Open testing with fault injection ────── FaultyPyTorchPlugin reveals bugs
Step 4  Test report generation ───────────────── pass rate & failure distribution
Step 5  Reproducible evidence packs ──────────── copy-paste Python scripts
```

---

## 🔄 Core Workflow

### Generate MRs

```bash
deepmt mr generate torch.nn.functional.relu --save   # single operator
deepmt mr batch-generate --framework pytorch          # all catalog operators
```

### Run Tests

```bash
deepmt test batch   --framework pytorch                       # batch metamorphic testing
deepmt test open    --inject-faults all --collect-evidence    # fault injection testing
deepmt test cross   relu --matrix --save                      # all framework pairs at once
```

### Analyze Results

```bash
deepmt test report                  # aggregated pass/fail report
deepmt test evidence list           # evidence pack index
deepmt test evidence show <id>      # one defect in detail
deepmt ui start                     # web dashboard → http://localhost:8000
```

---

## 🏗️ Architecture

**MR Generation — 4-stage pipeline:**

```
┌────────────┐    ┌────────────────┐    ┌──────────────┐    ┌──────────────┐
│ ① Info Prep│───▶│ ② Candidate Gen│───▶│ ③ Pre-check  │───▶│ ④ Formal     │
│  docs/code │    │  LLM + templates│    │  random nums │    │  Proof(SymPy)│
└────────────┘    └────────────────┘    └──────────────┘    └──────────────┘
```

**Package layout:**

```
deepmt/
├── mr_generator/     🧬  MR Generation Engine (3 layers)
│   ├── operator/     │     LLM hypothesis · template pool · SymPy proof
│   ├── model/        │     Graph analysis → strategy library
│   ├── application/  │     Scene knowledge · LLM/template fallback
│   └── base/         │     SQLite knowledge base · MR library
├── benchmarks/       📐  Benchmark Registry
│   ├── models/       │     ResNet-18 · VGG-16 · LSTM · BERT-encoder
│   └── applications/ │     ImageClassification · TextSentiment
├── engine/           ⚙️   Batch Test Executor (BatchTestRunner)
├── analysis/         🔍  Input Generator · Oracle Verifier · Reporter · Evidence
├── plugins/          🔌  Framework Adapters (Phase O — 4 frameworks, contract-aligned)
│   ├── pytorch       │     PyTorch — primary implementation
│   ├── numpy         │     NumPy   — float64 gold-standard reference
│   ├── paddle        │     PaddlePaddle — full operator parity
│   ├── tensorflow    │     TensorFlow — lazy-load, CPU-first
│   └── faulty_*      │     Fault injection backends (PyTorch & TensorFlow)
├── ui/               📊  Web Dashboard — 7 pages (Phase P)
├── commands/         💻  CLI sub-commands
└── core/             🎛️   Config · Logger · Plugin Manager · Health Checker
```

---

## 🛠️ Configuration

```bash
cp config.yaml.example config.yaml
# then edit config.yaml:
```

```yaml
llm:
  provider: "openai"
  api_key: "sk-..."        # or: export OPENAI_API_KEY=sk-...
  model_base: "gpt-4o-mini"
  model_max:  "gpt-4o"
```

Key environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | LLM API key (MR generation only) |
| `DEEPMT_LOG_LEVEL` | `INFO` | Verbosity — `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `DEEPMT_LOG_CONSOLE_STYLE` | `colored` | Terminal style — `colored` / `file` |
| `DEEPMT_INJECT_FAULTS` | — | Fault spec — `all` or `op:mutant,...` |

Full reference → [README_CONFIG.md](README_CONFIG.md) &nbsp;·&nbsp; [docs/environment_variables.md](docs/environment_variables.md)

---

## 🧪 Running Tests

```bash
source .venv/bin/activate
# All 766 unit tests — no LLM or network needed
PYTHONPATH=$(pwd) python -m pytest tests/unit/ -v

# With HTML coverage report
python -m pytest tests/unit/ --cov=deepmt --cov-report=html
```

---

## 📚 Documentation

| Document | Description |
|---|---|
| [README_CONFIG.md](README_CONFIG.md) | Configuration guide & all environment variables |
| [docs/cli_reference.md](docs/cli_reference.md) | Full CLI command reference (20+ commands) |
| [docs/quick_start.md](docs/quick_start.md) | Python API quick start |
| [docs/tech/operator_mr.md](docs/tech/operator_mr.md) | Operator-level MR technical details |
| [docs/environment_variables.md](docs/environment_variables.md) | Environment variable reference |
| [docs/dev/status.md](docs/dev/status.md) | Development status & completed modules |

---

## ⚙️ Requirements

| Component | Requirement |
|---|---|
| Python | ≥ 3.10 |
| PyTorch | ≥ 1.9.0 · GPU recommended |
| LLM API | Only for MR generation (`OPENAI_API_KEY`) |
| Browser | Any modern browser (Web Dashboard) |

---

## 📄 License

[MIT License](LICENSE) © 2026 cangtianhuang

<div align="center">
<br>
<sub>⭐ If DeepMT helps your research, a star keeps the project visible — thank you!</sub>
<br><br>

[![Footer](https://capsule-render.vercel.app/api?type=waving&color=0:4F9BF7,50:1a4b8a,100:0f2744&height=120&section=footer)](https://github.com/cangtianhuang/DeepMT)

</div>
