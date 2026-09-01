# hatch-finder

`hatchfinder` is an experimental PyTorch library for locating a supplied hatch
pattern inside a masked region of a drawing. The model produces a probability
heatmap with the same spatial dimensions as the drawing.

The project is currently in alpha. Its API, model format, and configuration may
change between releases.

## Installation

Install the package from PyPI:

```console
pip install hatchfinder
```

To work with the example dataset and configuration files, clone the repository
and install it in editable mode:

```console
git clone https://github.com/GreenMap-chan/hatch-finder.git
cd hatch-finder
pip install -e .
```

Python 3.11 or newer is required. PyTorch selects CPU or CUDA automatically
when `runtime.device` is set to `auto`.

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
Other PyTorch and Torchvision versions allowed by the package metadata have not
yet been verified.

## Inference

Create a model from a saved `.pt` file and pass paths or Pillow images to
`infer`:

```python
from pathlib import Path

from hatchfinder import HatchFinder

model = HatchFinder(
    load_model_path=Path("model.pt"),
    device="auto",
)

heatmap = model.infer(
    drawing=Path("drawing.png"),
    mask=Path("search_mask.png"),
    hatch=Path("hatch.png"),
    debug_path=Path("debug"),
    confidence=0.5,
)

print(heatmap.shape)  # [1, 1, height, width]
```

`drawing` and `hatch` are converted to RGB. `mask` is converted to grayscale
and binarized at `0.5`. The returned tensor contains probabilities in the
masked area and zeros outside it. When `debug_path` is provided, an overlay is
saved as `<drawing_name>_debug.png`.

## Training

Load a YAML configuration and start training:

```python
from pathlib import Path

from hatchfinder import Train, load_config

config = load_config(Path("examples/full_config.yaml"))
Train(config).train()
```

Relative paths in a configuration file are resolved relative to that YAML
file, not to the current working directory. Training writes the resolved
configuration, a log, `best.pt`, and `last.pt` to the configured output
directory.

Using `batch_size: 1` is strongly recommended when hatch images in the dataset
have different dimensions. Batches containing multiple examples pad smaller
hatch images to the largest height and width in the batch. The padding uses a
white background and may affect the hatch encoder. To retain a larger effective
batch without padding multiple hatch images together, keep `batch_size: 1` and
raise `gradient_accumulation_steps` instead. This is not a concern when all
hatch images in the dataset have the same dimensions, in which case a larger
batch size can be used.

`best.pt` and `last.pt` are resumable training checkpoints. To create a smaller
model-only file for inference:

```python
from pathlib import Path

from hatchfinder import convert_checkpoint_to_model

convert_checkpoint_to_model(
    Path("runs/full_sample/best.pt"),
    Path("runs/full_sample/model.pt"),
)
```

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

[`examples/sample_dataset`](https://github.com/GreenMap-chan/hatch-finder/tree/main/examples/sample_dataset)
contains 20 training and 10 validation examples. It is intentionally small and
is only suitable for checking the data pipeline and training loop, not for
training an accurate model. Each split contains an equal number of positive and
negative examples. The sample dataset is stored in the GitHub repository and is
not installed by `pip`.

The minimal CPU configuration in
[`examples/smoke_test.yaml`](https://github.com/GreenMap-chan/hatch-finder/blob/main/examples/smoke_test.yaml)
uses a reduced model and runs one epoch on this sample dataset. From the
repository root:

```python
from pathlib import Path

from hatchfinder import Train, load_config

config = load_config(Path("examples/smoke_test.yaml"))
Train(config).train()
```

Training output is written to `runs/sample`.

The full example configuration is available at
[`examples/full_config.yaml`](https://github.com/GreenMap-chan/hatch-finder/blob/main/examples/full_config.yaml).

## License

Licensed under the Apache License 2.0. See
[`LICENSE`](https://github.com/GreenMap-chan/hatch-finder/blob/main/LICENSE).
