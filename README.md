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

## Pretrained weights

Pretrained weights are available in the
[`GreenMap/hatch-finder-3.5m`](https://huggingface.co/GreenMap/hatch-finder-3.5m)
[`GreenMap/hatch-finder-11m`](https://huggingface.co/GreenMap/hatch-finder-11m)
model repository on Hugging Face. After downloading the `.pt` model file, pass
its local path to `HatchFinder` through `load_model_path`, as shown below.

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

By default, `infer()` uses the original image sizes. Set `imgsz` to resize a
tile while preserving its aspect ratio. An integer specifies its longest side.
A pair specifies a bounding box in **(height, width)** order. The drawing and
legend use the same scale factor, including when the tile is enlarged.

For a 1792 × 896 drawing (width × height):

| Call | Input to the model |
| --- | --- |
| `model.infer(drawing, mask, hatch, imgsz=896)` | 896 × 448 |
| `model.infer(drawing, mask, hatch, imgsz=896, rect=False)` | 896 × 896, with 224 white pixels above and below the drawing |

The default `rect=True` uses the fitted size. With `rect=False` and an explicit
`imgsz`, the drawing is centered on the requested canvas. The search mask uses
nearest-neighbor resizing and zero padding. The legend keeps its scaled size.
Both modes return the heatmap and debug overlay in the original drawing
coordinates, with heatmap values set to zero outside the original search mask.

### Reuse encoder features

Pass either or both keys when an input is reused across calls on the same model:

```python
heatmap = model.infer(
    drawing, mask, hatch,
    drawing_key=(sheet_id, tile_x, tile_y, mask_id),
    hatch_key=(legend_id, legend_scale),
    precision="fp16",
)
```

Use stable, hashable IDs. Change `drawing_key` whenever the tile pixels or
search mask change, and change `hatch_key` when the legend pixels change.
Include a document or revision ID when reusing names across different inputs.
The library uses these IDs directly and accounts for its own resizing, padding
and active inference precision. An omitted key runs that encoder on every call.

Each model retains features for its most recent keyed tile and up to 16 keyed
legends on the inference device. Legend entries are evicted in least-recently-used
order. Matching and pixel correlation still run for each pair.
Call `model.clear_inference_cache()` to release the features, or after editing
weights in place. Loading a state dict, entering training mode or moving the
model to another device or dtype clears both caches automatically.

## Cross-attention matching

Optional cross-attention compares drawing locations with individual pooled hatch
tokens. Enable it for selected matching levels in your YAML configuration:

```yaml
model:
  cross_attention:
    - level: 2
      num_heads: 8
      downsample: 1
      dropout: 0.0
      initial_gate: 0.01
```

Levels are zero-based and must have a positive `match_dims[level]`, divisible by
`num_heads`. The attention width is `match_dims[level]`; each head uses
`match_dims[level] / num_heads` features. Keys and values come from the
`hatch_pool_sizes[level]` square grid, projected per cell and augmented with
learned positional embeddings. Queries come from projected drawing features.
The local matcher compares queries and attended hatch context using their
concatenation, absolute difference, and product. Its output is added to the
existing matching map through a trainable gate initialized by `initial_gate`.

`downsample` reduces only the drawing query grid; the local matching map is
interpolated back before addition. The attention score dimensions are
`N_drawing × hatch_pool_sizes[level]²` per head. Multiple entries enable multiple
levels; each level may appear only once. `dropout` applies to attention weights.

## Training

For training on your own dataset, use
[`examples/recommended.yaml`](examples/recommended.yaml) as a recommended
starting point. It includes a four-level model, pixel correlation, and four
transformer blocks in each encoder. Tune the training settings for your data
and hardware; this preset is not a guarantee of optimal results.

The preset leaves dataset and output paths unset. Pass your own paths to
`Train`, or set `data.dataset` and `output.directory` in a local copy of the
YAML file. Paths passed directly to `Train` are relative to the current working
directory.

```python
from pathlib import Path

from hatchfinder import Train, load_config

if __name__ == "__main__":
    config = load_config(Path("examples/recommended.yaml"))
    Train(
        config,
        dataset_path=Path("dataset"),
        output_path=Path("runs/training"),
    ).train()
```

For the bundled sample dataset, use `examples/full_config.yaml`:

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

By default, `output.unique_run_directory: true` creates a new run directory
when `output.directory` already exists, adding `_0`, `_1`, and so on to the
directory name. Set it to `false` to reuse the directory for a new training
run. Resuming from `training.checkpoint_path` always uses the selected output
directory without adding a suffix.

Training uses BCE + soft Dice by default. To train with BCE + Tversky instead,
set `training.loss: bce_tversky`; `training.tversky_fp_weight` and
`training.tversky_fn_weight` default to `0.3` and `0.7` and must sum to one.
Validation and checkpoint selection continue to use BCE + soft Dice, so their
reported values remain comparable across training-loss choices. Existing
checkpoints without `training.loss` continue to use BCE + Dice.

`training.max_consecutive_skipped_updates` sets how many consecutive optimizer
updates can be skipped for a non-finite training loss or gradient norm. The
default `0` stops at the first failure. Set it to `1` to allow one skip and
stop on the second consecutive failure. A skip discards the accumulation group
and leaves the optimizer and learning-rate schedule at their previous step.
A successful update resets the counter. Checkpoints preserve the counter, and
you can set a new limit when resuming. The log identifies the batch and its
sample IDs for a loss failure, or all sample IDs in the group for a gradient
failure. Epoch summaries count skips and average loss over successful updates'
examples.

Set `data.imgsz` to the base resolution for both training and validation.
An integer sets the longest side. A YAML pair such as `[640, 896]` sets the
bounding height and width. The default `null` keeps the dataset resolution.
As in `infer()`, `data.rect: true` uses the fitted rectangle, and `false`
centers it on the requested canvas. The drawing and legend share the base
scale. Both search and target masks use nearest-neighbor resizing.

Training applies augmentations **after** this step. For example, a square tile
with `data.imgsz: 896` and a drawing resize factor of 1.05 reaches the model
at 941 × 941. Validation uses the base geometry directly (896 × 896 in this
example). On resume, omitted size settings come from the checkpoint. Explicit
changes to the base geometry start a fresh comparison of validation metrics.

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
JSON Lines manifest with paths to drawings, legends and targets. Datasets with
saved search windows also supply search masks, as shown below.

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
- `search_mask`, when supplied, is a grayscale mask of the area in which to search.
- `hatch` is an RGB image of the hatch pattern to find.
- `target` is a grayscale ground-truth mask. White pixels mark matches; an
  all-black target represents a negative example.

Images belonging to one example may have arbitrary filenames, but `drawing`,
`search_mask`, and `target` must have the same spatial dimensions. The hatch
image may have a different size.

### Generate search masks during training

The manifest supports two formats.

- **Saved window.** Supply `drawing`, `hatch`, `search_mask`, and `target`.
  The loader uses the saved window and its target, including when mask
  generation is enabled.
- **Generated window.** Supply `drawing`, `hatch`, and `target`, and set
  `data.search_mask_generation.enabled: true`. The target must mark every
  match across the **entire tile**. An empty target means the legend is absent
  everywhere on that tile. The loader generates a window and clips the target
  to it when it samples the training row.

The defaults for the optional mode are shown in
[the complete configuration](examples/full_config.yaml). With
`negative_fraction: 0.20` and `outside_window_share_of_negatives: 0.50`,
training draws about 80% positives, 10% pairs with an absent legend, and 10%
pairs with the legend visible only outside the window. The trainer logs the
observed proportions among generated windows per epoch. Rows with whole-tile
targets must supply each requested source pool.

Window dimensions satisfy `area_fraction_min/max` relative to the whole tile
and `aspect_ratio_min/max` in width divided by height. `min_margin_fraction`
applies independently to both axes. `min_positive_pixels` counts matching
pixels inside a positive window at the augmented drawing resolution, before
batch padding. Both negative types have zero matching pixels inside the window.
`full_tile_probability` selects the whole tile for positive and absent-legend
samples, overriding the rectangular size and margin limits for that draw.
When the base canvas adds padding, window fractions and margins use the
bounding box of the transformed drawing content. Generated rectangles stay
entirely inside that content, including after affine transforms. A full-tile
search covers all of the content and gives the padding zero weight.

Validation uses the stored window when supplied. For three-file validation
rows the loader generates a repeatable window using the augmentation seed and
manifest row index, keeping each original pair and its positive or absent type.
To evaluate outside-window negatives at fixed locations, include their saved
validation search masks.

On checkpoint resume, explicit window settings override saved settings. The
two training sampling fractions preserve the best metric and patience. Changing
settings that affect generated validation windows resets that comparison.

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

## Architecture experiments with trained weights

Set `training.load_new_model_config: true` to start a new experiment using either
`load_model_path` or `checkpoint_path` as the weight source (specify only one).
The source must include its model configuration; bare state dictionaries are not
supported in this mode. Paths in YAML are relative to that YAML file.

```yaml
training:
  checkpoint_path: ../runs/baseline/best.pt
  load_new_model_config: true
  learning_rate: 0.00003
  epochs: 30
model:
  pixel_correlation:
    enabled: true
data:
  dataset: ./sample_dataset
output:
  directory: ../runs/correlation_experiment
```

Omitted `model` fields inherit the source settings. Nested mappings merge by
field, while lists replace the complete source list: `transformers: []` removes
all drawing transformers. Omitted fields inside newly supplied list entries use
their standard defaults. Changes are validated after merging; when changing the
number of levels, update all corresponding per-level lists and attention levels.
Training, data, runtime and output settings come from the current configuration.

Compatible layers retain their trained weights; new or incompatible layers keep
their normal initialization, including biases. Decoder blocks are matched by
encoder level, and matcher output layers are matched separately from hidden
layers. Changing attention head counts resets the affected attention, including
its output projection. Changing stride, downsample or dropout retains compatible
weights. Existing learned gates are retained; `initial_gate` applies to new
branches. A new transformer or cross-attention branch can use `initial_gate: 0.0`
to initially preserve the original predictions. Adding blocks inside an existing
transformer does not provide that guarantee.

The run prints and logs the configuration differences, transferred/initialized
layers, unused source layers, and transferred parameter count. Optimizer,
scheduler, epoch count and early stopping start fresh. A unique output directory
is always used, even when `unique_run_directory` is false. The resolved config
records the source path, and new checkpoints contain the full resulting model
configuration, so inference and ordinary resume do not require the source file.
To resume this experiment normally, pass its new checkpoint with
`load_new_model_config: false`. The default `false` preserves ordinary loading
and resume behavior. Initial validation runs before training as usual.

## License

Licensed under the Apache License 2.0. See
[`LICENSE`](https://github.com/GreenMap-chan/hatch-finder/blob/main/LICENSE).
