# BA-AMIL: Boundary-Aware Attention Multiple Instance Learning

Boundary-Aware Attention MIL (**BA-AMIL**) for breast-cancer lymph-node **metastasis detection** in whole-slide pathological images (CAMELYON16), using precomputed **UNI** foundation-model features. This repository accompanies the ECE531 Computer Vision term-project paper.

BA-AMIL augments standard attention-based MIL with (i) a **graph-attention stream** over a spatial *k*-nearest-neighbor patch graph, (ii) a **learned gating** fusion of the global and boundary-aware streams, and (iii) **attention-entropy** and **graph-smoothness** regularizers.

## Results (CAMELYON16, 5-fold CV, UNI features)

| Method   | AUC    | Accuracy | F1    | Attn. entropy |
|----------|--------|----------|-------|---------------|
| ABMIL    | 0.9846 | 0.953    | 0.940 | 0.745         |
| CLAM-SB  | 0.9854 | 0.956    | 0.943 | 0.751         |
| **BA-AMIL (ours)** | **0.9903** | **0.959** | **0.948** | **0.620** |

Ablations confirm: learned gating > fixed fusion, optimal `λ_ent = 0.01`, `λ_smooth = 0.1`, `k = 8` (robust to `k`).

## How to run

1. **Hugging Face access (required, gated dataset).** Create a free account, open
   <https://huggingface.co/datasets/kaczmarj/camelyon16-uni> and accept the conditions,
   then create a *read* token at <https://huggingface.co/settings/tokens>.
2. Open the notebook in **Google Colab** (Runtime → GPU, e.g. T4).
3. Run the cells top to bottom. The first cell asks for your Hugging Face token.
   - Leave `CONFIG["FAST"] = True` for a quick end-to-end check, then set it to
     `False` and re-run cells for the full results.
4. Outputs (metrics + figures) are written to `/content/results/` and can be copied to Google Drive.

## Files

- `ba_amil_camelyon16.ipynb` — full pipeline: data download, models (ABMIL, CLAM-SB, BA-AMIL), training, ablations, figures.
- `results/` — `RESULTS_BUNDLE.json`, CSV summaries, and figures produced by the notebook.

## Method summary

- **Features:** frozen UNI (ViT-L) patch embeddings (1024-d) + patch coordinates.
- **Baselines:** ABMIL [Ilse et al., 2018], CLAM-SB [Lu et al., 2021].
- **Proposed:** dual-stream attention (global + GAT boundary) with learned gating; loss = BCE + λ_ent·H(a) + λ_smooth·smoothness.
- **Data:** CAMELYON16 training slides (111 tumor / 159 normal), stratified 5-fold CV.

## Dataset & credits

- CAMELYON16: <https://camelyon16.grand-challenge.org/>
- Precomputed UNI features: `kaczmarj/camelyon16-uni` on Hugging Face.

## License

For academic coursework use. CAMELYON16 and UNI features are subject to their respective licenses (CC-BY-NC-SA 4.0 for the feature release).
