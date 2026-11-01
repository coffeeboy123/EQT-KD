# EQT-KD

This repository contains the code for EQT-KD, a lightweight infrared person re-identification project based on quantization-aware training and knowledge distillation.

## Dataset

The datasets used in this project are low-resolution infrared person re-identification datasets.

### Low-Resolution SYSU Dataset

Download link:

https://drive.google.com/drive/folders/10z9OGHbDUPH8t5mGl_vaMmy_1CKcGIsK?usp=drive_link

### Low-Resolution RegDB Dataset

Download link:

https://drive.google.com/drive/folders/1yXsAhncgnUsYKSMw5FF6XSNyOmzey-XW?usp=drive_link

## Pretrained Weight

Download link:

https://drive.google.com/drive/folders/1gAPjVMJAg2XUBevBbRS_mbr1bgT93myX?usp=drive_link

## Reproducibility

### Data preparation

The low-resolution (LR) IR images follow the protocol of our previous study: each high-resolution image is resized to 256 × 256 pixels with Lanczos interpolation, blurred with a Gaussian filter (σ = 3), downscaled by a factor of four to 64 × 64 pixels, and corrupted with additive Gaussian noise (σ = 10 on the 8-bit intensity scale).
The degradation is applied once, offline, with fixed parameters to all training, validation, query, and gallery images. At training and test time, the LR images are resized to 384 × 128 pixels with bilinear interpolation.

Each dataset is split into two folds by identity (training and test identities are disjoint). In each fold, 20% of the images of each training identity form the validation set, and one query image and two gallery images are randomly selected for each test identity.

### Training

```bash
bash scripts/train_eqtkd.sh <data root> <teacher checkpoint dir> [seed]
```

Main settings: Adam (learning rate 6 × 10⁻⁴, weight decay 5 × 10⁻⁴) with cosine annealing for 300 epochs, batches of 6 identities × 2 images, loss weights λ_ce = 0.5, λ_ms = 0.5, and λ_kd = 0.25, label smoothing ε = 0.1, KD temperature T = 1.5, KLW scaling δ = 0.01, and MS loss with margin 0.8, similarity threshold 0.5, and scaling factors 2 and 40.
QAT uses the eager-mode PyTorch quantization API with the default QNNPACK QAT configuration. The backbones are initialized with the ImageNet-pretrained OSNet weights (β = 1.0 for the teacher and β = 0.5 for the student).

Random seeds: 251 for all results in the paper; 1 and 2 were additionally used for the repeated runs.

Checkpoint selection: the test set is evaluated during training, and the checkpoint with the highest Rank-1 accuracy (mAP as a tie-breaker) is reported. The final (300th-epoch) checkpoint is also reported in the paper as a reference without test-based selection.

### Evaluation

INT8 results are obtained by converting the QAT model with `torch.ao.quantization.convert` and evaluating it on the CPU (QNNPACK).

```bash
# per-query AP and CMC of a stored checkpoint
python tools/eval_perquery.py qat <checkpoint> <out.npz> -- --datadir <fold dir> --data_test SYSU --model_student LMBN_n_osnet_x0_5_student_scaling --feats_student 256 --height 384 --width 128
# query-level paired bootstrap CIs (10,000 resamples) and Wilcoxon signed-rank test between two models
python tools/paired_bootstrap.py <A_foldA.npz> <A_foldB.npz> <B_foldA.npz> <B_foldB.npz>
# FP32 / fake-quant / INT8 evaluation of the final QAT weights (qat_final_state.pth saved by main.py)
python tools/eval_fp32_fq_int8.py <qat_final_state.pth> <out prefix> -- --datadir <fold dir> --data_test SYSU --model_student LMBN_n_osnet_x0_5_student_scaling --model_teacher LMBN_n_fusion_3 --feats_student 256 --feats_teacher 512 --height 384 --width 128
```
