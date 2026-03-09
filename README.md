# Echo2ECG: Enhancing ECG Representations with Cardiac Morphology from Multi-View Echos

This is the official PyTorch implementation of Echo2ECG.

## 📋 Outline

1. [Getting Started](#getting-started)
2. [Setup](#setup)
3. [Data Preparation](#data-preparation)
4. [Training and Evaluation](#training-and-evaluation)

## 🚀 Getting Started

1. **Setup**: Complete the installation steps in the "Setup" section.

2. **Models**: Download OTiS and EchoPrime model weights and place them in the `model_weights` directory.

3. **Data**: Prepare your datasets as described in the "Data Preparation" section.

4. **Train/Eval**: Run pre-training and kNN evaluation scripts according to your specific needs.

## 🛠 Setup

```bash
# Create and activate environment
conda create -n echo2ecg python=3.12
conda activate echo2ecg

# Install dependencies
pip install -r requirements.txt

# Install the current package
pip install -e .
```

## 📊 Data Preparation

### 1) ECG preprocessing

```bash
python ecg/data_processing/processing.py --input_dir INPUT_DIR --output_dir OUTPUT_DIR
```
where:
- `INPUT_DIR`: path to the directory containing ECGs to process
- `OUTPUT_DIR`: path to the directory saving the processed ECGs

### 2) Echo embeddings for multimodal pre-training

```bash
python echo/data_processing/generate_embeddings.py --echo_encoder_path ECHO_ENCODER_PATH --input_dir INPUT_DIR --output_dir OUTPUT_DIR
```
where:
- `ECHO_ENCODER_PATH`: path to the EchoPrime model
- `INPUT_DIR`: path to the directory containing Echos (`.avi`) to process
- `OUTPUT_DIR`: path to the directory saving the generated Echo embeddings (e.g., `echoprime_echo_embeddings_unnorm.pt`)

### 3) Paired ECG-Echo data for multimodal pre-training

Paired data should be prepared as a `.csv` file. Each row should provide at least:

- `ecg_study_id`
- `echo_study_id`
- `ecg_filepath` (the full path to a `.pt` ECG tensor)
- `echo_filepath` (the full path to an `.avi` Echo) [this is not necessary if only pre-computed Echo embeddings are used for pre-training]
- `echo_embed_idx` (index of the Echo in the pre-computed Echo embeddings)

### 4) Downstream ECG data

Downstream data loaders require two files per split: one ECG file (`data_<train/val/test>`) and one label file (`labels_<train/val/test>`).

For `data_<train/val/test>`, you can provide either:
- a `.pt` file containing preprocessed ECGs in the format `[('ecg', torch.Tensor), ...]` with length `num_samples`, where each tensor has shape `(num_leads, num_timesteps)`, or
- a `.csv` file with a `filename` column pointing to ECG `.pt` files.

`labels_<train/val/test>` must be a `.pt` tensor of shape `(num_samples, num_classes)` with one-hot encoded labels.


## 🏋🏻‍♀️ Training and Evaluation

### A) Multimodal pre-training (CLIP)

```bash
# base config: configs/base.yaml
# hyperparameters as used in the paper
python run.py --config-name=base \
    dataset=multimodal/dataset_clip \
    model=multimodal/model_clip \
    train=multimodal/pretrain_clip \
    max_epochs=50 \
    dataset.batch_size=256 \
    dataset.accum_iter=1 \
    dataset.echo.use_precomputed_embeds=true \
    model.echo.view_aggregation.use=true \
    model.echo.view_aggregation.strategy=att \
    model.echo.view_aggregation.num_layers=1 \
    model.echo.view_aggregation.proj_embed_dim=1024 \
    model.alignment.proj_embed_dim=512 \
    train.encoder.ecg.checkpoint_path=<path-to-otis-model> \
    train.encoder.ecg.freeze_first_n_layers=0 \
    train.encoder.echo.freeze_first_n_layers=16 \
    train.params.lr_ecg_encoder=5e-4 \
    train.params.lr=5e-4 \
    train.params.weight_decay_ecg_encoder=1e-7 \
    train.params.weight_decay=1e-7 \
    train.params.layer_decay=0.75 \
    train.params.scheduler.warmup_cosine.warmup_steps=2 \
    train.clip_loss.temperature=0.5
```
Important configs:
- Update `home_dir` in `configs/base.yaml` (recommended: path to this repo)
- Adjust filepaths to the data in `configs/dataset_clip.yaml`
- Using ECG <-> Multi-view Echo alignment: `model.echo.view_aggregation.use=true`
- Using ECG <-> Single-view Echo alignment: `model.echo.view_aggregation.use=false`
- Using pre-computed echo embeddings: `dataset.echo.use_precomputed_embeds=true`
- Initializing ECG encoder with OTiS weights: `train.encoder.ecg.checkpoint_path=<path-to-otis-model>`
- Fully freezing Echo encoder layers: `train.encoder.echo.freeze_first_n_layers=16`

Outputs under `checkpoints/`, including:
- `ecg_multimodal.ckpt`
- `echo_multimodal.ckpt`
- `token_aggregator.ckpt`
- `ecg_alignment_global_token.ckpt`
- `echo_alignment_global_token.ckpt`
- optionally `echo_projection_for_agg.ckpt`, `echo_view_aggregator.ckpt`


### B) Downstream ECG: kNN evaluation

```bash
# base config: configs/base_ecg_linearprobe.yaml
python run.py --config-name=base_ecg_linearprobe \
    downstream_task_ecg=<downstream-task> \
    downstream_task_ecg.time_steps=1008 \
    downstream_task_ecg.apply_augmentations=false \
    ecg_encoder_checkpoint_path=<path-to-ecg-model> \
    token_aggregator_path=<path-to-token-aggregator> \
    ecg_alignment_path=null
```
Important configs:
- Update `home_dir` in `configs/base_ecg_linearprobe.yaml` (recommended: path to this repo)
- Adjust filepaths to the data in `configs/downstream_task_ecg/<downstream-task>.yaml`
- After multimodal pre-training, `ecg_encoder_checkpoint_path` should be the path to `ecg_multimodal.ckpt` and `token_aggregator_path` should be the path to `token_aggregator.ckpt`
- Run evaluation on val set: `validate=true`
- Run evaluation on test set: `test=true`

To add a new downstream ECG task, add a new config file `<downstream-task>.yaml `in the `configs/downstream_task_ecg` folder and pass it via `downstream_task_ecg=<downstream-task>`.

### C) ECG->Echo Retrieval

Coming soon