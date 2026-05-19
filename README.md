
<h1 align="center">Rethinking LLM Ensembling from the Perspective of Mixture Models</h1>

<p align="center">
<a href="https://arxiv.org/abs/2605.00419">
<img alt="Static Badge" src="https://img.shields.io/badge/arXiv-2605.00419-red"></a>
</p>

Mixture-model-like Ensemble (ME) is a training-free, plug-and-play ensembling method that reinterprets LLM ensembling as a mixture model and samples from the same ensemble distribution while invoking only one model per step. ME is mathematically equivalent to sampling from the ensemble distribution and only requires evaluating one model per step, making it 1.78x-2.68x faster than conventional ensembling.

## Setup

### 1. Create environment

```bash
conda create -y -n me python=3.11
conda activate me
pip install -r requirements.txt
```

### 2. Specify model paths in `.env`

Model paths are loaded from `.env`. Use `.env.example` as the template:

```bash
cp .env.example .env
```

Each entry can point to a local checkpoint directory or a Hugging Face model id. The default config uses:

```yaml
models:
  - qwen2_5-3b-instruct
  - qwen2_5-math-1_5b-instruct
```

Single-model configs live under `configs/model/`; `configs/default.yaml` selects an ensemble by listing those config names in `models`.

## How to Run

All methods use the same Hydra + lm-eval entrypoint:

```bash
python eval.py method=me eval.tasks=mmlu_gen eval.limit=50
```

Supported methods:

- `single`: run only the first model in `models`.
- `ce_single`: conventional ensemble with all models in one Ray actor.
- `ce_parallel`: conventional ensemble with one Ray actor per model.
- `me`: Mixture-Model-like Ensemble.

Example commands:

```bash
python eval.py method=single eval.tasks=gsm8k eval.limit=10
python eval.py method=ce_single eval.tasks=arc_challenge_chat eval.limit=10
python eval.py method=ce_parallel eval.tasks=mmlu_gen eval.limit=10
python eval.py method=me eval.tasks=bbh_gen eval.limit=10
python eval.py method=me models='[openchat-3_5-0106,deepseek-llm-7b-chat]' eval.tasks=gsm8k eval.limit=10
```

## Recommended Citation
```
@article{fu2026rethinking,
  title={Rethinking LLM Ensembling from the Perspective of Mixture Models},
  author={Fu, Jiale and Jiang, Yuchu and Wu, Peijun and Liu, Chonghan and Zhou, Joey Tianyi and Yang, Xu},
  journal={arXiv preprint arXiv:2605.00419},
  year={2026}
}
```

## Acknowledgements
We would like to thank the authors of [GaC](https://github.com/yaoching0/GaC) for their heterogeneous model ensembling method.
