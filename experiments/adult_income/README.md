# Adult Income / real OpenML tabular extension

This exporter uses OpenML dataset 1590 (48,842 records), a 50/25/25 stratified split, train-only preprocessing, and a `(64, 32)` MLP. It exports logits and hidden features for the shared TS/VS+LAPE demo.

```bash
python experiments/adult_income/train_export.py
python demo/run_demo.py --npz experiments/adult_income/outputs/adult_model_outputs.npz --anchor VS --output demo/output_adult
```

The paper appendix reports Adult NLL reductions of 9.25% for TS+LAPE and 6.97% for VS+LAPE relative to their anchors. The VS example is in [`examples/adult_income/`](../../examples/adult_income/README.md).
