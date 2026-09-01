# hatch-finder

## Tested environment

The project has been tested with the following versions:

- Python 3.11.0
- Pillow 12.3.0
- PyYAML 6.0.3
- Pydantic 2.13.4
- PyTorch 2.12.0.dev20260408+cu128
- Torchvision 0.27.0.dev20260407+cu128
- tqdm 4.70.0

The tested PyTorch and Torchvision packages are nightly builds for CUDA 12.8.

## Dataset structure

A dataset must contain separate `train` and `valid` splits. Each split has a
JSON Lines manifest and four image directories:

```text
dataset/
├── train/
│   ├── manifest.jsonl
│   ├── drawing/
│   ├── search_mask/
│   ├── hatch/
│   └── target/
└── valid/
    ├── manifest.jsonl
    ├── drawing/
    ├── search_mask/
    ├── hatch/
    └── target/
```

Each line in `manifest.jsonl` describes one example. Paths are relative to the
dataset root:

```json
{"drawing":"train/drawing/0001.png","search_mask":"train/search_mask/0001.png","hatch":"train/hatch/0001.png","target":"train/target/0001.png"}
```

- `drawing` is an RGB drawing.
- `search_mask` is a grayscale mask of the area in which to search.
- `hatch` is an RGB image of the hatch pattern to find.
- `target` is a grayscale ground-truth mask. White pixels mark matches; an
  all-black target represents a negative example.

Images belonging to one example may have arbitrary filenames, but `drawing`,
`search_mask`, and `target` must have the same spatial dimensions. The hatch
image may have a different size.

## Sample dataset

`examples/sample_dataset` contains 20 training and 10 validation examples. It
is intentionally small and is only suitable for checking the data pipeline and
training loop, not for training an accurate model. Each split contains an equal
number of positive and negative examples.

The minimal CPU configuration in `examples/smoke_test.yaml` uses a reduced
model and runs one epoch on this sample dataset. From the repository root:

```python
from pathlib import Path

from hatchfinder import Train, load_config

config = load_config(Path("examples/smoke_test.yaml"))
Train(config).train()
```

Training output is written to `runs/sample`.
