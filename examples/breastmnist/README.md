# BreastMNIST example

The chart, HTML report, and animated GIF were generated with `demo/run_demo.py` and `demo/make_walkthrough.py` from BreastMNIST ResNet18 features and logits. This example shows the VS anchor comparison. The paper's Figure 2 includes additional methods and settings.

| Method | NLL ↓ | Accuracy ↑ |
| --- | ---: | ---: |
| Base | 0.4468 | 89.10% |
| VS | 0.3484 | 89.10% |
| VS+LAPE | 0.2976 | 89.10% |

To train and evaluate from the official dataset, follow [`experiments/breastmnist/README.md`](../../experiments/breastmnist/README.md). Results can vary with pretrained weights, training randomness, and implementation versions.
