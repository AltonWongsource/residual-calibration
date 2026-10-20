# Obesity Risk CPU reproduction track

Run these commands **from this directory** after installing `requirements.txt` and an appropriate CPU build of PyTorch and torchvision. Data, pretrained weights, caches, and result files are downloaded or created locally; none are included in this repository.

## Obesity Risk: 7-class MLP

The script fetches UCI dataset 544 via `ucimlrepo`, unless `--obesity-csv PATH` is supplied. These are separate train-time loss runs; each also runs the VS/LAPE post-hoc comparison and writes calibration/test prediction bundles.

```bash
python -m train_time_cpu.run --dataset obesity --loss ce --seed 1
python -m train_time_cpu.run --dataset obesity --loss mdca --seed 1
python -m train_time_cpu.run --dataset obesity --loss mbls --seed 1
```

Repeat with other seeds for an aggregate comparison. Outputs are under `outputs_cpu/train_time/obesity/`. The `--quick` flag runs two epochs for a smoke test; the commands above use the full defaults. Hyperparameter search and held-out selection follow the script defaults.

To view one exported model with the standalone anchor/LAPE demo, run from the repository root:

```bash
python demo/run_demo.py --calibration experiments/obesity_and_dtd/outputs_cpu/train_time/obesity/ce/seed_1/calibration.npz --test experiments/obesity_and_dtd/outputs_cpu/train_time/obesity/ce/seed_1/test.npz --anchor VS
```

## Checks

```bash
python -m pytest -q
```

The seven included unit tests cover losses, metrics, and post-hoc pipeline behavior. The DTD CoOp/DAC experiment is documented in [`../dtd_coop_dac/`](../dtd_coop_dac/README.md).
