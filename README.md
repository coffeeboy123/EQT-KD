# EQT-KD

This repository contains the code for EQT-KD, a lightweight infrared person re-identification project based on quantization-aware training and knowledge distillation.

## Dataset

The experiments use DBPerson-Recog-DB1, SYSU-MM01, and the NIR subset of RGBNT201. The datasets are not redistributed in this repository; please obtain them from their providers under their respective terms and generate the low-resolution images as described in [Data preparation](#data-preparation).

## Pretrained Weight

Download link:

https://drive.google.com/drive/folders/1gAPjVMJAg2XUBevBbRS_mbr1bgT93myX?usp=drive_link

## Reproducibility

### Data preparation

The low-resolution (LR) IR images follow the protocol of our previous study: each high-resolution image is resized to 256 × 256 pixels with Lanczos interpolation, blurred with a Gaussian filter (σ = 3), downscaled by a factor of four to 64 × 64 pixels, and corrupted with additive Gaussian noise (σ = 10 on the 8-bit intensity scale).
The degradation is applied once, offline, with fixed parameters to all training, validation, query, and gallery images. At training and test time, the LR images are resized to 384 × 128 pixels with bilinear interpolation.

Each dataset is split into two folds by identity (training and test identities are disjoint). In each fold, about 20% of the images of each training identity form the validation set (2 of 10 images per identity for DBPerson-Recog-DB1, the rounded-up 20% for RGBNT201, and 6 images per identity for SYSU-MM01), and one query image and two gallery images are randomly selected once for each test identity. The query images carry a camera label different from the gallery images, so the camera filter of the evaluator removes no gallery image and all queries are evaluated (247 per fold for SYSU-MM01, 206 for DBPerson-Recog-DB1, 86/85 for RGBNT201).

Datasets used:
- DBPerson-Recog-DB1 (also distributed as RegDB; `Reg` in the code): thermal images only. The loader parses the first number in a file name (`<...>_<number>_<identity>.bmp`) as the camera label and the second as the identity; within each identity, the query and gallery images have different first numbers, so the camera filter removes no gallery image.
- SYSU-MM01: IR images of cameras 3 and 6 (494 identities in our copy, split into two folds of 247 identities); the official cross-modality identity lists are not used.
- RGBNT201: NIR images (`NI`) of the 141 identities of the official training set (`train_141`) and the 30 identities of the official test set (`test`), i.e., 171 identities, split into folds of 85 and 86 identities. The 30 official validation identities (in `train_171` but not in `train_141`) are not used.

Expected folder structure of one fold (loaders `SYSU`, `Reg`, `RGBNT201`):
```
<data root>/ReIDataset_SYSU_A/SYSU/{bounding_box_train, bounding_box_validation, query, bounding_box_test}
```

### Training

```bash
bash scripts/train_eqtkd.sh <data root> <teacher checkpoint dir> [seed]
```

Main settings: Adam (learning rate 6 × 10⁻⁴, weight decay 5 × 10⁻⁴) with cosine annealing for 300 epochs, batches of 6 identities × 2 images, loss weights λ_ce = 0.5, λ_ms = 0.5, and λ_kd = 0.25, label smoothing ε = 0.1, KD temperature T = 1.5, KLW scaling δ = 0.01, and MS loss with margin 0.8, similarity threshold 0.5, and scaling factors 2 and 40.
QAT uses the eager-mode PyTorch quantization API with the default QNNPACK QAT configuration. The backbones are initialized with the ImageNet-pretrained OSNet weights released by the authors of OSNet (β = 1.0 for the teacher and β = 0.5 for the student), which are loaded from `~/.cache/torch/checkpoints/osnet_x1_0_imagenet.pth` and `osnet_x0_5_imagenet.pth`.

Random seeds: 251 for all results in the paper; 1 and 2 were additionally used for the repeated runs.

The student model of the paper is `LMBN_n_osnet_x0_5_student_scaling` (KLW: bias-free linear layer C→1, ω = 1 + 0.01·w, Eq. 8); it is the default of `--model_student`. Other student variants in `model/` (e.g., `LMBN_n_osnet_x0_5_student` with a sigmoid-based weight) were not used for the reported results. The rule-based fallback weight in `loss/kd_logic_loss_qaware.py` is not used when the student supplies learned weights, as in all reported results.
The teacher is `LMBN_n_fusion_3`; the student of each fold must be trained with the teacher checkpoint of the same fold (`--teacher_pretrain`), because the classifier size depends on the training identities of the fold. `train_ablation.bat` is a historical script for the ablations without KD and uses machine-specific paths.

Known behaviors of the code as used for the paper (kept unchanged for reproducibility):
- Validation uses the training transform and random (not identity-balanced) sampling, and is used only to monitor the loss/accuracy curves.
- During the per-epoch validation, the fake-quantization observers of the QAT student are not disabled, so the validation images (training identities) also update the activation ranges. Test results are computed on a converted copy. `tools/eval_fp32_fq_int8.py` disables the observers.
- The seed is set after the models and data manager are constructed, so runs are not bit-wise repeatable; the paper reports the variation over three seeds.

Checkpoint selection: the test set is evaluated during training, and the checkpoint with the highest Rank-1 accuracy (mAP as a tie-breaker) is reported. The final (300th-epoch) checkpoint is also reported in the paper as a reference without test-based selection.

### Evaluation

INT8 results are obtained by converting the QAT model with `torch.ao.quantization.convert` and evaluating it on the CPU (QNNPACK). For all models, the features of each image and its horizontally flipped version are summed; for the student, the eight 256-dimensional BNNeck descriptors are each L2-normalized and concatenated (2048 dimensions), and the distance is one minus the inner product. Re-ranking is not used, and the two folds are averaged with equal weight.
The inference times in the paper were measured separately (TensorRT `trtexec` on the desktop GPU and Jetson Orin Nano, ONNX on the mobile NPU) and are not produced by this evaluator. The parameter count (2.35 M on SYSU-MM01) includes the training-only classifier heads and KLW; the embedding network has about 1.83 M parameters.

```bash
# per-query AP and CMC of a stored checkpoint
python tools/eval_perquery.py qat <checkpoint> <out.npz> -- --datadir <fold dir> --data_test SYSU --model_student LMBN_n_osnet_x0_5_student_scaling --feats_student 256 --height 384 --width 128
# query-level paired bootstrap CIs (10,000 resamples) and Wilcoxon signed-rank test between two models
python tools/paired_bootstrap.py <A_foldA.npz> <A_foldB.npz> <B_foldA.npz> <B_foldB.npz>
# FP32 / fake-quant / INT8 evaluation of the final QAT weights (qat_final_state.pth saved by main.py)
python tools/eval_fp32_fq_int8.py <qat_final_state.pth> <out prefix> -- --datadir <fold dir> --data_test SYSU --model_student LMBN_n_osnet_x0_5_student_scaling --model_teacher LMBN_n_fusion_3 --feats_student 256 --feats_teacher 512 --height 384 --width 128
```
