# DTD: feature-cached CoOp, DAC, and LAPE

This folder reproduces the paper's DTD ViT-B/32 CoOp/DAC appendix experiment. It uses the Zhou DTD base/new class split, 16-shot base-class prompt training, a 100-epoch CoOp checkpoint, and five stratified novel-class reference/test splits (seeds 0–4; 20% reference). DAC is evaluated without novel-class labels. CoOp+LAPE and DAC+LAPE use labelled novel-class reference samples for post-hoc refinement.

The feature-cached CoOp trainer learns shared context tokens with frozen ViT-B/32 image and text encoders. It caches four augmented image views per base training image and samples among them during prompt optimization, rather than generating fresh augmentations every epoch. Post-hoc evaluation uses the trained checkpoint across the five novel-class splits.

## Run from this directory

Install a compatible PyTorch/torchvision build and `pip install -r requirements.txt`. Obtain the DTD images, `split_zhou_DescribableTextures.json` from the CoOp dataset instructions, and OpenCLIP ViT-B/32 `openai` weights. These assets are supplied separately.

```bash
python tools/build_dtd_manifest_from_zhou_split.py --split-json PATH_TO_ZHOU_SPLIT.json --image-dir PATH_TO_DTD_IMAGES --shots 16 --seed 1 --output work/dtd_manifest.csv
python tools/cache_dtd_open_clip.py --manifest work/dtd_manifest.csv --model-name ViT-B-32 --pretrained openai --base-train-views 4 --batch-size 8 --output work/dtd_feature_cache.npz
python tools/train_cached_coop.py --cache work/dtd_feature_cache.npz --output-dir work/coop_seed1 --model-name ViT-B-32 --pretrained openai --n-ctx 16 --epochs 100 --learning-rate 0.002 --patience 0 --seed 1
python tools/export_master_bundle.py --cache work/dtd_feature_cache.npz --coop-text-outputs work/coop_seed1/coop_text_outputs.npz --output work/dtd_cached_coop_master.npz
python tools/validate_master_bundle.py --bundle work/dtd_cached_coop_master.npz
python run_fast_rebuttal.py --master-bundle work/dtd_cached_coop_master.npz --output-dir outputs/dtd_coop_dac_lape --split-seeds 0,1,2,3,4 --reference-fraction 0.20 --model-seed cached_coop_seed1
```

`run_fast_rebuttal.py` writes DAC and LAPE-extension metrics by split and in aggregate. The paper reports percentage NLL gains from the aggregate means: `(anchor_mean - refined_mean) / anchor_mean`. Expected NLLs are CoOp 2.176879 → CoOp+LAPE 1.613748 (+25.87%) and DAC 1.778696 → DAC+LAPE 1.539453 (+13.45%). One prompt checkpoint is evaluated across five reference/test splits.

The exported master bundle holds model outputs for post-hoc evaluation. Run `python -m pytest -q` for unit tests; the tests use generated fixtures. The DTD dataset and OpenCLIP weights remain subject to their providers' terms.
