# Letter Recognition / 26-class MLP

This experiment uses a 40/40/20 train/calibration/test split, train-only feature scaling, a `(128, 64)` MLP, and calibration-fitted hidden-feature scaling. It exports model outputs for the shared TS/VS+LAPE demo.

Download the UCI `letter-recognition.data` file. The exporter expects 20,000 rows and 26 classes.

From the repository root:

```bash
python experiments/letter_recognition/train_export.py --data path/to/letter-recognition.data
python demo/run_demo.py --npz experiments/letter_recognition/outputs/letter_model_outputs.npz --anchor VS --output demo/output_letter
```

The paper appendix reports NLL results for additional anchors, including Dir+LAPE. This entry point exports the model outputs for the VS example shown in [`examples/letter_recognition/`](../../examples/letter_recognition/README.md).
