# Letter Recognition results

This VS-anchor example uses the 20,000-row UCI Letter Recognition dataset with seed 42. The results come from the exporter and shared demo scripts.

| Method | NLL ↓ | Accuracy ↑ |
| --- | ---: | ---: |
| Base | 0.2380 | 94.30% |
| VS | 0.1734 | 94.225% |
| VS+LAPE | 0.1696 | 94.40% |

For data preparation and commands, see [`experiments/letter_recognition/README.md`](../../experiments/letter_recognition/README.md).
