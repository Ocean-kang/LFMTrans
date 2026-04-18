# LFMTrans

LFMTrans is a feature-alignment repository for mapping text features into a vision feature space and solving correspondence with functional maps. The repo currently supports:

- standard feature-bank training (`fmap.type=train`)
- one-photo / few-photo feature extraction and training (`fmap.type=train_one_photo`)
- anchor-based alignment (`fmap.type=anchor`)
- IP + L2 graph evaluation (`fmap.type=Ip_and_L2`)
- segmentation-style evaluation with a trained projector (`eval.enabled=1`)

## 1. Environment

```bash
conda env create -f conda.yaml
conda activate lfmtrans
```

## 2. Repository layout

```text
configs/LFMTrans_cfg.yaml          # main config
main.py                            # training / FM evaluation entry
extract_one_photo_features.py      # one-photo feature extraction entry
model/                             # projector + DINOv2 wrapper
src/                               # trainer / mapping logic
utils/load_feature.py              # offline feature loading
utils/load_feature_one_photo.py    # one-photo feature extraction/loading
```

## 3. Required data and pretrained assets

Set the dataset roots in `configs/LFMTrans_cfg.yaml`:

- `dataset.root_dir_mscoco`
- `dataset.root_dir_ade20k`
- `dataset.root_dir_pc`
- `dataset.root_dir_voc`

Put pre-extracted offline features under `paths.feature_root` (default: `./feature`).

Put the local LLaMA checkpoint under `paths.llama_root` (default: `./Meta-Llama-3-8B-Instruct`).

## 4. Training modes

### 4.1 Standard projector training

```bash
python main.py -F train_log/coco_train \
  with fmap.type=train \
       train.dataset=cocostuff \
       train.text_model=llama_unmean \
       train.type=patch_unmean \
       validation.text_model=llama \
       validation.type=patch
```

### 4.2 One-photo feature extraction

```bash
python extract_one_photo_features.py -F train_log/one_photo_extract \
  with train.dataset=pc59 \
       train_one_photo.split=train \
       train_one_photo.num_images=3 \
       train_one_photo.target_num_classes=10 \
       train_one_photo.bank_size=12
```

The extracted feature file will be saved inside the Sacred log directory, for example:

```text
train_log/one_photo_extract/<run_id>/features/one_photo/*.pkl
```

### 4.3 Train with extracted one-photo features

```bash
python main.py -F train_log/one_photo_train \
  with fmap.type=train_one_photo \
       train.dataset=pc59 \
       train.text_model=llama_unmean \
       train.type=patch_unmean \
       validation.text_model=llama \
       validation.type=patch \
       train_one_photo.feature_path=/absolute/path/to/your_feature.pkl
```

## 5. Other modes

### 5.1 Anchor mode

```bash
python main.py -F train_log/anchor \
  with fmap.type=anchor
```

### 5.2 IP + L2 evaluation

```bash
python main.py -F train_log/ip_l2_eval \
  with fmap.type=Ip_and_L2 \
       projector.checkpoint_path=/absolute/path/to/projector.pt
```

### 5.3 Segmentation-style evaluation with a trained projector

```bash
python main.py -F train_log/seg_eval \
  with eval.enabled=1 \
       eval.method=LFMTrans \
       eval.checkpoint_path=/absolute/path/to/projector.pt
```

## 6. Output structure

When you run with Sacred file storage (`-F train_log/<exp_name>`), every run writes into its own log directory:

```text
train_log/<exp_name>/<run_id>/
├── checkpoints/                  # trained projector checkpoints
├── features/one_photo/           # one-photo extracted features
├── features/one_photo/debug/     # optional debug outputs
└── resolved_config.json          # config snapshot after runtime path resolution
```

## 7. Important config fields

- `train.dataset`: dataset used for training
- `train.text_model`: text feature family (`llama`, `llama_unmean`, `mpnet`, `mpnet_unmean`)
- `train.type`: vision feature family (`patch`, `patch_unmean`)
- `validation.datasets`: datasets used in retrieval-style validation
- `projector_train.*`: projector optimization hyperparameters
- `projector_train.mask`: `1` for original training, `2` for cluster training
- `train_one_photo.*`: one-photo sampling / extraction parameters
- `paths.feature_root`: offline feature directory
- `paths.log_root`: default fallback log root when `-F` is not used

## 8. Notes

- `main.py` uses the first validation dataset to infer projector input/output dimensions during evaluation.
- Segmentation evaluation only supports semantic-segmentation datasets (`cocostuff`, `150`, `847`, `pc59`, `pc459`, `voc20`, `voc20b`). Non-segmentation datasets in `validation.datasets` are skipped in that path.
- For reproducibility, seeds are set in both `main.py` and `extract_one_photo_features.py`.
