# BA-AMIL: Boundary-Aware Attention Multiple Instance Learning

Boundary-Aware Attention MIL (**BA-AMIL**) for breast-cancer lymph-node **metastasis detection** in whole-slide pathological images (CAMELYON16), built on precomputed **UNI** foundation-model features. This repository accompanies the ECE531 Computer Vision term-project paper and reproduces every table and figure in it.

BA-AMIL augments standard attention-based MIL with (i) a **graph-attention stream** over a spatial *k*-nearest-neighbor patch graph, (ii) a **learned gating** fusion of the global and boundary-aware streams, and (iii) **attention-entropy** and **graph-smoothness** regularizers.

## Results (CAMELYON16, 5-fold CV, UNI features)

| Method   | AUC | Accuracy | F1 | Attn. entropy |
|----------|-----|----------|----|----|
| ABMIL    | 0.9846 ± 0.014 | 0.953 ± 0.027 | 0.940 ± 0.037 | 0.745 |
| CLAM-SB  | 0.9854 ± 0.014 | 0.956 ± 0.033 | 0.943 ± 0.044 | 0.751 |
| **BA-AMIL (ours)** | **0.9903 ± 0.013** | **0.959 ± 0.024** | **0.948 ± 0.031** | **0.620** |

Mean ± standard deviation across five folds and three seeds. Ablations confirm: learned gating > fixed fusion, optimal `λ_ent = 0.01`, `λ_smooth = 0.1`, `k = 8` (robust to `k`).

## 1. Setup

- Open `ba_amil_camelyon16.ipynb` in **Google Colab** (Runtime → Change runtime type → **GPU**, e.g. T4).
- Dependencies (`torch`, `huggingface_hub`, `h5py`, `scikit-learn`, `pandas`, `matplotlib`) are installed by the first cell; no local install is required.
- A free **Hugging Face account** and a **read access token** (<https://huggingface.co/settings/tokens>) are needed to download the gated feature dataset.

## 2. Dataset preparation

The slides are **not** raw WSIs; we use precomputed UNI patch embeddings + patch coordinates.

1. Accept the dataset terms at <https://huggingface.co/datasets/kaczmarj/camelyon16-uni> (the dataset is gated).
2. When the notebook prompts, paste your Hugging Face read token. The download cell fetches the embedding (`.pt`) and coordinate (`.h5`) files for the **270 CAMELYON16 training slides** (111 tumor / 159 normal). Labels are read from the slide-name prefix (`tumor_` / `normal_`).
3. Data is cached under `/content`; only small result files are written to Google Drive (optional cell).

## 3. Training

- Run the notebook cells top to bottom. Set `CONFIG["FAST"] = True` first for a quick end-to-end check, then set it to `False` and re-run the training cells for the full numbers.
- Default configuration: Adam (lr `2e-4`, weight decay `1e-5`), ≤40 epochs with early stopping (patience 8), dropout 0.25, hidden dim 256, 4 GAT heads, `k = 8`, `λ_ent = 0.01`, `λ_smooth = 0.1`.
- Protocol: stratified **5-fold cross-validation**; main comparison over seeds `{0, 1, 2}`, ablations over seed `{0}`.

## 4. Inference

- The training cells evaluate each held-out fold internally. To visualize a single slide, run the attention-heatmap cell: it trains a BA-AMIL model and overlays the per-patch attention on the slide's patch coordinates (`fig_attention_heatmap.png`).

## 5. Evaluation

- Metrics: **AUC** (primary), **Accuracy**, **F1** (at 0.5 threshold), and **normalized attention entropy** `H(a)`.
- The headline comparison (ABMIL / CLAM-SB / BA-AMIL) and the ablation studies (entropy, gating, `k`, smoothness) are produced automatically and written to `RESULTS_BUNDLE.json` and the CSV/PNG files in `results/`.

## Files

- `ba_amil_camelyon16.ipynb` — full pipeline: download, models, training, ablations, figures.
- `results/` — `RESULTS_BUNDLE.json`, CSV summaries, and the figures used in the paper.

## Method summary

- **Features:** frozen UNI (ViT-L) patch embeddings (1024-d) + patch coordinates.
- **Baselines:** ABMIL [Ilse et al., 2018], CLAM-SB [Lu et al., 2021].
- **Proposed:** dual-stream attention (global + GAT boundary) with learned gating; loss = BCE + λ_ent·H(a) + λ_smooth·smoothness.

## Dataset & credits

- CAMELYON16: <https://camelyon16.grand-challenge.org/>
- Precomputed UNI features: `kaczmarj/camelyon16-uni` on Hugging Face (CC-BY-NC-SA 4.0).

## GenAI acknowledgment

Generative AI (Anthropic Claude) was used in a supporting role: literature search assistance, language editing and manuscript structuring, and producing auxiliary materials such as figures and table formatting. The method design, implementation, and all experiments were carried out by the author, and all reported numbers come from the author's own runs.

## License

For academic coursework use. CAMELYON16 and the UNI feature release are subject to their respective licenses.
