#!/bin/bash
# Train the EQT-KD student (Extreme LightMBN, INT8 QAT) on both folds with the settings of the paper.
# usage: bash scripts/train_eqtkd.sh <data root> <teacher checkpoint dir> [seed]
#   <data root>/ReIDataset_SYSU_{A,B}         : low-resolution SYSU-MM01 folds
#   <teacher checkpoint dir>/LMBN_n_fusion_3_SYSU_{A,B}_6_2_0.0006_12_300_model-best.pth : Enhanced LightMBN teachers
# The paper uses seed 251; the repeated runs in Section 5.1 additionally use seeds 1 and 2.
DATA=$1
TEACHER=$2
SEED=${3:-251}
for F in A B; do
  python main.py --datadir $DATA/ReIDataset_SYSU_$F \
    --data_train SYSU --data_validation SYSU --data_test SYSU \
    --model_student LMBN_n_osnet_x0_5_student_scaling --model_teacher LMBN_n_fusion_3 \
    --teacher_pretrain $TEACHER/LMBN_n_fusion_3_SYSU_${F}_6_2_0.0006_12_300_model-best.pth \
    --batchid 6 --batchimage 2 --batchtest 12 --seed $SEED --epochs 300 \
    --loss "0.5*CrossEntropy+0.5*MSLoss+0.25*KL_Logic_Loss_QAware" --kl_temp 1.5 --margin 0.8 \
    --nGPU 1 --lr 6e-4 --optimizer ADAM --random_erasing --feats_student 256 --feats_teacher 512 \
    --if_labelsmooth --w_cosine_annealing --height 384 --width 128 --save eqtkd_seed$SEED
done
