# Training Log

**Project**: 3D Breast MRI Subtraction Synthesis with ControlNet

---

## Stage 1: ControlNet Alignment (Epochs 1-50)

**Status**: Ready to start

**Configuration**:
- U-Net: Fully frozen
- ControlNet LR: 1e-4
- LR Schedule: Warmup (5%) + Cosine Annealing
- Validation: Every epoch
- Loss: Weighted MSE (ROI: 5.0, Background: 1.0)

**Expected Output**: `models/breast_controlnet_stage1_best.pt`

---

## Training Commands

```bash
# Stage 1
sbatch scripts/submit_stage1.sh

# Manual start
conda activate breast_gen
python -m scripts.train_controlnet \
    --env_config_path configs/environment_maisi_controlnet_train_rflow-mr_breast_stage1.json \
    --model_config_path configs/config_maisi_controlnet_train_stage1.json \
    --model_def_path configs/config_network_rflow.json \
    --num_gpus 1
```

---

## Monitoring

```bash
# Real-time logs
tail -f outputs/logs/breast_controlnet_stage1/train.log

# TensorBoard
tensorboard --logdir outputs/tfevent/

# Check validation results
grep "Validation COMPLETE" outputs/logs/breast_controlnet_stage1/train.log
```

---

## Dataset Statistics

| Split | Samples | Patients | With Tumor | No Tumor |
|-------|---------|----------|------------|----------|
| Train | 1,553 (80%) | 1,203 | 77.4% | 22.6% |
| Val    | 390 (20%)  | 301   | 77.2% | 22.8% |

---

## Checkpoints

| Stage | Epoch | Best Loss | Status |
|-------|-------|-----------|--------|
| 1 | - | - | Ready to start |
