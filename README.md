# Making Open-Source Text LLM Watermarks Durable Against Merging

Official code for the NeurIPS 2026 paper [**Making Open-Source Text LLM Watermarks Durable Against Merging**](https://openreview.net/forum?id=3fkF2rVnDA) by Luisa Scharff, Thibaud Gloaguen, Robin Staab, and Martin Vechev.

Open-source LLM watermarks live in the model weights, and model merging strongly removes them. We propose Merge-Adversarial Training (MAT): during watermark distillation, every training step merges the student $\theta$ with the unwatermarked base $\theta_0$, $\theta_M = \alpha\theta + (1 - \alpha)\theta_0$ with $\alpha \sim U(0.1, 1)$, and distills the watermark through the merged model. This repository contains:

- watermark distillation with and without the merge adversary for KGW, AAR and KTH, and the GaussMark baseline,
- the merge pipeline (Linear, SLERP, TIES, DARE-TIES and Model Breadcrumbs via [MergeKit](https://github.com/arcee-ai/mergekit)) with the merge scenarios of the paper,
- detection, generation-quality and benchmark evaluation

## Table of Contents 

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Reproducing the Paper](#reproducing-the-paper)
- [Merge Scenarios](#merge-scenarios)
- [Repository Structure](#repository-structure)
- [Third-Party Code](#third-party-code)
- [License](#license)
- [Citation](#citation)

## Installation

Training and detection need Linux and an NVIDIA GPU. The experiments were run on NVIDIA GH200 and RTX PRO 6000 Blackwell (96 GB) GPUs. On GH200 we used NVIDIA's PyTorch 25.06 container (Python 3.12, torch 2.8.0a0, Triton 3.3.0). Direct dependencies are pinned in `requirements.txt` (training, merging and evaluation) and `requirements-sft.txt` (finetuning). We use [uv](https://docs.astral.sh/uv/):

```bash
# finetuning environment: per-domain SFT (transformers 5.5.4, TRL 1.1.0)
uv venv --python 3.12 .venv-sft
source .venv-sft/bin/activate
uv pip install -r requirements-sft.txt
deactivate

# main environment: watermark training, merging, evaluation
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt   
export SFT_ACCELERATE=$PWD/.venv-sft/bin/accelerate
```

KTH detection uses the Rust detector from [eth-sri/diffusion-lm-watermark](https://github.com/eth-sri/diffusion-lm-watermark) when it is installed. Otherwise it compiles `src/mat/watermarks/levenshtein.pyx` with Cython on first use (needs a C compiler). 

Request access to the gated models `meta-llama/Llama-3.1-8B-Instruct`, `meta-llama/Llama-3.2-1B-Instruct`,
`meta-llama/Llama-2-13b-hf`, `meta-llama/Llama-3.1-70B-Instruct`, then log in:

```bash
hf auth login
```

## Quickstart

Train MAT (KGW) on Llama-3.1-8B-Instruct, check detection, and merge the watermarked model with the unwatermarked base:

```bash
accelerate launch scripts/train_watermark.py --config configs/watermark/llama8b_mat.yaml

python scripts/eval_detection.py --model output/watermark/llama8b_mat/checkpoint-5000 \
    --watermark_config configs/watermark/llama8b_mat.yaml --domain en --output_dir results/standalone/mat

python scripts/run_merge_sweep.py --variant mat --scenario BBp --method linear --domains en
```

Training uses whatever setup `accelerate config` selects. Each config sets an `effective_batch_size`, which must equal `per_device_train_batch_size` times the number of processes. To train MAT on several GPUs, choose DeepSpeed ZeRO-1 or ZeRO-2.

### Examples

The commands below use the MAT checkpoint from the [Quickstart](#quickstart). Each evaluation prints its metrics and writes them to `--output_dir`. Add `--n_samples 50` for a quick check.

Detect the watermark, merge the model 50/50 with the unwatermarked base, and detect again:

```bash
CKPT=output/watermark/llama8b_mat/checkpoint-5000
CONFIG=configs/watermark/llama8b_mat.yaml

python scripts/eval_detection.py --model $CKPT --watermark_config $CONFIG --domain en --output_dir results/example/watermarked
python scripts/merge_pair.py --a meta-llama/Llama-3.1-8B-Instruct --b $CKPT --method linear --alpha 0.5 --output_dir output/example/merged
python scripts/eval_detection.py --model output/example/merged --watermark_config $CONFIG --domain en --output_dir results/example/merged
```

Further evaluations:

```bash
# generation quality: median perplexity under Llama-2-13B and seq-rep-3
python scripts/eval_quality.py --model $CKPT --watermark_config $CONFIG --domain en --output_dir results/example/watermarked

# detection after 4-bit quantization of the merged model
python scripts/eval_detection.py --model output/example/merged --watermark_config $CONFIG --domain en --quantization nf4 --output_dir results/example/merged_nf4

# decoding-time KGW reference (no trained model needed)
python scripts/eval_detection.py --logit_processor --watermark_config configs/watermark/llama8b_kgwd.yaml --domain en --output_dir results/example/kgw
```

## Reproducing the Paper

Each experiment has a driver in [`experiments/`](experiments), run from the repository root:

```bash
bash experiments/main_results/run.sh
```

Run `main_results` first, it trains the KGW-D and MAT models and the finetunes that the other drivers reuse.

| Experiment | Driver |
|---|---|
| Watermark training, finetuning, merge scenarios, benchmarks | `main_results` |
| Behavioral audit | `behavioral_audit` |
| Code-finetune lower bound | `code_ft_lower_bound` |
| AAR, KTH and GaussMark | `watermark_families` |
| Qwen and Ministral | `architectures/run_qwen.sh`, `architectures/run_ministral.sh` |
| Post-merge finetuning and re-merging | `post_merge_sft` |
| Merge-robustness mechanism | `mechanism` |
| DARE-TIES and Model Breadcrumbs | `merge_operators` |
| Quantization | `quantization` |
| Learned merging and query-limited removal | `learned_merging` |
| Adaptive merge ratio | `adaptive_ratio` |
| Text attacks | `text_attacks` |
| Ablations and training-time merge reference | `ablations`, `reference_ablation` |

Notes:

- **Setup.** Llama-3.1-8B-Instruct watermarked with KGW ($\gamma = 0.25$, $\delta = 2.3$) for 5,000 steps. MAT adds top-k = 100 gating and $\alpha \sim U(0.1, 1)$. Detection reports TPR at 1% FPR. Merges sweep $\alpha$ from 0.1 to 0.9 (TIES with density 0.5). Perplexity is computed with Llama-2-13B.
- **Finetuning.** `main_results`, `post_merge_sft` and `code_ft_lower_bound` need `SFT_ACCELERATE` (see [Installation](#installation)).

### Outputs

- `output/`: trained models and merges. Merged weights are deleted after evaluation unless `--keep_models` is set.
- `detection_<domain>.json`: per-sample p-values and z-scores, and `tpr_at_1pct_fpr` with Wald and bootstrap 95% intervals.
- `quality_<domain>.json`: per-sample perplexity under Llama-2-13B and seq-rep-3, and the median perplexity.

## Merge Scenarios

B is a watermarked Llama-3.1-8B-Instruct, B' the unwatermarked one, and F_math, F_german, F_french finetunes of B. In each scenario, $\alpha$ is the weight of one parent, as in the paper's figures. Scenarios are defined in [`configs/merge/scenarios.yaml`](configs/merge/scenarios.yaml).

| Scenario | Merge |
|---|---|
| `FF` | F_german ⊗ F_math |
| `BF` | B ⊗ F_math |
| `F2BF` | BF ⊗ FF |
| `FBF` | F_german ⊗ BF |
| `F2F` | F_french ⊗ FF |
| `BBp` | B' ⊗ B |
| `BFp_FuseChat`, `BFp_OpenMath`, `BFp_Tulu` | F' ⊗ B, with F' an unwatermarked third-party finetune |

## Repository Structure

```text
├── configs/
│   ├── watermark/                     # distillation configs 
│   ├── sft/                           # per-domain finetuning
│   ├── merge/                         # models, merge scenarios and merge methods
│   └── eval/                          # lm-evaluation-harness benchmarks
├── experiments/
├── scripts/
│   ├── generate_kth_samples.py        # KTH-watermarked samples
│   ├── train_on_samples.py            # KTH-D and MAT KTH training on the samples
│   ├── train_watermark.py             # watermark distillation, with or without MAT
│   ├── finetune.py                    # per-domain SFT
│   ├── embed_gaussmark.py             # GaussMark baseline
│   ├── prepare_ministral.py           # checkpoint of Ministral-3
│   ├── run_merge_sweep.py             # merge a scenario across α and evaluate every merge
│   ├── merge_pair.py                  # single pairwise merge
│   ├── eval_detection.py              # TPR@1% FPR
│   ├── eval_gaussmark.py              # TPR@1% FPR for GaussMark
│   ├── eval_quality.py                # reference perplexity and seq-rep-3
│   ├── eval_benchmarks.py             # downstream accuracy (lm-evaluation-harness)
│   ├── analyze_lower_bound.py         # BB' as a lower bound for FBF 
│   └── analyze_single_ft_baseline.py  # FF vs. the single-FT baseline 
└── src/mat/
    ├── trainer.py                     # merge-adversarial distillation
    ├── watermarks/                    # KGW, AAR, KTH, GaussMark
    ├── data.py                        # distillation mixture and SFT corpora
    ├── merging.py                     # MergeKit configurations
    ├── evaluation.py                  # prompts, sampling, detection metrics
    ├── config.py                      # training configs
    ├── rng.py                         # RNG for the KGW green list
    └── utils.py                       # model and tokenizer loading
```

## Third-Party Code

The 8-shot GSM8K chain-of-thought exemplars are adapted from [Wei et al. (2022)](https://arxiv.org/abs/2201.11903). The refusal classifier uses the prompt format from the [WildGuard model card](https://huggingface.co/allenai/wildguard) (Apache-2.0), and the DIPPER attack follows the usage in the [DIPPER model card](https://huggingface.co/kalpeshk2011/dipper-paraphraser-xxl) (Apache-2.0). AAR and KTH distillation follow [Gu et al. (2024)](https://arxiv.org/abs/2312.04469). The optional Rust KTH detector is built from [eth-sri/diffusion-lm-watermark](https://github.com/eth-sri/diffusion-lm-watermark).

## License

This code is released under the [MIT License](LICENSE). Datasets and models used by the code keep their own licenses.

## Citation

```bibtex
@inproceedings{scharff2026making,
  title={Making Open-Source Text {LLM} Watermarks Durable Against Merging},
  author={Luisa Scharff and Thibaud Gloaguen and Robin Staab and Martin Vechev},
  booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
  year={2026},
  url={https://openreview.net/forum?id=3fkF2rVnDA}
}
```
