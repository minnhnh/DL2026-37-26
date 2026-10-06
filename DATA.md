# Dataset: Open Images V7 continual subset

## Source

- Dataset: Open Images V7
- Official page: <https://storage.googleapis.com/openimages/web/download_v7.html>
- Version: V7, accessed through the FiftyOne dataset zoo (`open-images-v7`)
- Source splits used: official `validation` and `test`
- Target box classes: Dog, Cat, Car, Person, Building
- Intended task: single-label class-incremental image classification

Open Images images are distributed under their source licenses, commonly
Creative Commons Attribution. Preserve the Open Images image ID and follow the
official attribution requirements when publishing example images.

## Processed dataset download

The processed crops used in every reported experiment (2,500 JPEG files,
about 377 MB) are available at:

- Download: <https://drive.google.com/file/d/1MEBVCbjy7x3ySbqFA9-fqIKRAuTG8OAl/view?usp=sharing>
  (`DL2026-37-26_open_images_v7_crops.zip`, 372 MiB)

The archive contains the 2,500 crops, the three split manifests, and a short
`README.txt` with source and license notes. Extract it at the repository root
so that the files appear under `data/processed_clean/`. The committed manifests in `data/manifests/` already
point to these paths, so no further step is needed before training.

## Data split

| Split | Per class | Total | Manifest |
|---|---:|---:|---|
| Train | 400 | 2,000 | `data/manifests/train.csv` |
| Validation | 50 | 250 | `data/manifests/val.csv` |
| Test | 50 | 250 | `data/manifests/test.csv` |

The official validation and test images are pooled, then repartitioned into the
course train/validation/test sets by original image ID with seed 42
(`scripts/partition_manifest.py`). This avoids the high-memory metadata pass
required by the nine-million-image Open Images train split. Exact and
perceptual duplicate checks are run before partitioning. No original image ID
may occur in multiple course splits.

The `source_split` column in each manifest holds the course split. The folder
names under `data/processed_clean/` (`validation/`, `test/`) are the official
Open Images source splits, not the course splits, so always select samples
through the manifests rather than through the folders.

Each manifest row contains:

```text
sample_id,original_image_id,filepath,label,label_id,source_split,bbox,sha256,crop_width,crop_height
```

Label IDs follow the class-incremental order: dog 0, cat 1, car 2, person 3,
building 4. The stream is Dog + Cat at Stage 0, then Car, Person and Building
at Stages 1, 2 and 3.

## Preprocessing

### Crop construction

The project downloads only images with target-class detection annotations. It
keeps at most one target crop per original source image, rejects group boxes,
depictions, inside views, very small boxes (box area below 2% of the
image, or crop shorter side below 80 px), and corrupt files, and applies 8% context padding
around each box. Person additionally uses Man/Woman/Boy/Girl annotations and
rejects occluded or truncated boxes. Building additionally uses House/Office
building/Skyscraper/Tower/Castle annotations and rejects occluded boxes. Each
class is balanced to 500 crops.

### Training-time transforms

Defined in `src/continual_dl/data/transforms.py`:

- Train: `RandomResizedCrop(224, scale=(0.7, 1.0))`, random horizontal flip,
  `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)`,
  ImageNet mean/std normalization.
- Validation and test: resize the shorter side to 256, center crop 224,
  ImageNet mean/std normalization.

## Reproduction scripts

Install the data extras first (`pip install -r requirements-data.txt`), then
run from the repository root:

```powershell
# 1. Download source images and annotations through FiftyOne
python scripts/download_openimages.py --splits validation test --sampling-mode balanced --max-samples 1200

# 2. Build balanced crops, check duplicates, and partition by image ID
python scripts/build_crops.py --raw-manifest data/raw/manifests/validation.jsonl data/raw/manifests/test.jsonl --output-root data/processed_clean --output-manifest data/manifests/pool_clean.csv --target-per-class 500
python scripts/deduplicate.py data/manifests/pool_clean.csv
python scripts/partition_manifest.py --pool-manifest data/manifests/pool_clean.csv

# 3. Audit and verify the final splits
python scripts/audit_dataset.py data/manifests/train.csv data/manifests/val.csv data/manifests/test.csv --output-dir outputs/dataset_audit_clean
python scripts/verify_dataset.py data/manifests/train.csv data/manifests/val.csv data/manifests/test.csv
```

`verify_dataset.py` fails on missing files, label-mapping drift, class
imbalance (with `--expected-per-class`), and source-image or exact-crop
leakage between splits.

A fresh FiftyOne download is not guaranteed to return exactly the same source
images. To reproduce the reported numbers, use the processed archive above
together with the committed manifests.

## Known limitations

- Object crops are easier than full-scene image classification.
- Bounding-box context and crop scale differ between object categories.
- Some valid Person crops contain only a clearly recognizable upper/lower body,
  and some Building crops depict entrances or interiors rather than facades.
- `Person` raises privacy and representation concerns; only public research
  images and the released Open Images annotations are used.
- ImageNet-pretrained backbones may already encode the five target concepts.
- A single `Building` prototype may be weak because the visual class is highly
  multimodal.

## Required manual audit

Before training, inspect every generated contact sheet under
`outputs/dataset_audit_clean/`. Remove mislabeled, unsafe, ambiguous, or low-quality
crops and restore class balance before producing final manifests.
