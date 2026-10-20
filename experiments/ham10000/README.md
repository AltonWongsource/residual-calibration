# HAM10000 image + clinical-metadata CFAF

This script reproduces the HAM10000 rows in the paper's multimodal CFAF table. It uses frozen ImageNet ResNet18 image features, a metadata MLP, VS-calibrated view predictions, and CFAF-LAPE on the joint representation. Samples are split by **lesion ID** so photographs of one lesion stay in the same split. Both binary tasks use seed 7 and a 60%/20%/20% train/calibration/test split.

Place `metadata.csv` and the JPEGs named `<isic_id>.jpg` under `data/ham10000/`. The CSV needs `isic_id`, `lesion_id`, `diagnosis_1`, `diagnosis_3`, `age_approx`, `sex`, `anatom_site_1`, `anatom_site_2`, `anatom_site_3`, `anatom_site_general`, and `anatom_site_special`. Images, metadata, and pretrained weights are not redistributed.

```bash
pip install torch torchvision numpy pandas scipy scikit-learn pillow
python run.py --task ham10000_melanoma_nevus --seed 7
python run.py --task ham10000_malignant_benign --seed 7
```

Each task writes a four-row detailed CSV and a JSON summary under `outputs/<task>/`. The reported methods are image only, metadata only, late fusion, and `CFAF-LAPE (concat/std/cosine, alpha<=0.5)`. A joint concat head supplies the CFAF anchor.

Expected NLL from the paper (late fusion → CFAF-LAPE): 0.3409 → 0.3194 for malignant-vs-benign, and 0.2553 → 0.2243 for melanoma-vs-nevus. The detailed CSV also reports accuracy, ECE, and ACE.
