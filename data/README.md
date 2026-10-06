# Dataset directory

The manifests in `data/manifests/` are committed. Image files and raw
downloads (`data/raw/`, `data/processed_clean/`, FiftyOne caches) are not
committed; download the processed archive or rebuild it as described in
[`DATA.md`](../DATA.md).

Final manifests:

- `data/manifests/train.csv`
- `data/manifests/val.csv`
- `data/manifests/test.csv`

Each CSV contains:

```text
sample_id,original_image_id,filepath,label,label_id,source_split,bbox,sha256,crop_width,crop_height
```
