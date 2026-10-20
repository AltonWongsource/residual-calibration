# DTD ViT-B/32 CoOp/DAC results

The paper experiment uses feature-cached CoOp training on the DTD base/new class split. One CoOp checkpoint is evaluated over five labelled novel-class reference/test splits for the LAPE extension. DAC metrics without novel-class labels are reported separately. See [`experiments/dtd_coop_dac/README.md`](../../experiments/dtd_coop_dac/README.md) for setup and commands.

Across split seeds 0–4, aggregate NLL is CoOp 2.176879 → CoOp+LAPE 1.613748 and DAC 1.778696 → DAC+LAPE 1.539453. Ratios of the aggregate means give the paper's +25.87% and +13.45% NLL improvements.
