call "C:\Users\Howon_LEE\miniconda3\Scripts\activate.bat"
call conda activate howond

call cd /d D:\LightMBN-master-experiment-val\LightMBN-master-experiment-val

echo ------------------------- 학습: 1 시작 -------------------------

call python main.py ^
  --datadir C:/Users/Howon_LEE/Desktop/Hard_Data_validation/ReIDataset_Reg_A ^
  --data_train REGDB ^
  --data_test REGDB ^
  --data_validation REGDB ^
  --model LMBN_n_par_3_partweightgate_no_residual ^
  --batchid 6 ^
  --batchimage 2 ^
  --batchtest 12 ^
  --seed 251 ^
  --test_every 1 ^
  --epochs 300 ^
  --loss 0.5*CrossEntropy+0.5*MSLoss ^
  --margin 0.8 ^
  --nGPU 1 ^
  --lr 6e-4 ^
  --optimizer ADAM ^
  --random_erasing ^
  --feats 512 ^
  --save log ^
  --if_labelsmooth ^
  --w_cosine_annealing ^
  --height 384 ^
  --width 128



echo ------------------------- 학습: 1 완료 -------------------------
timeout /t 5 >nul
echo ------------------------- 학습: 2 시작 -------------------------

call python main.py ^
  --datadir C:/Users/Howon_LEE/Desktop/Hard_Data_validation/ReIDataset_Reg_B ^
  --data_train REGDB ^
  --data_test REGDB ^
  --data_validation REGDB ^
  --model LMBN_n_par_3_partweightgate_no_residual ^
  --batchid 6 ^
  --batchimage 2 ^
  --batchtest 12 ^
  --seed 251 ^
  --test_every 1 ^
  --epochs 300 ^
  --loss 0.5*CrossEntropy+0.5*MSLoss ^
  --margin 0.8 ^
  --nGPU 1 ^
  --lr 6e-4 ^
  --optimizer ADAM ^
  --random_erasing ^
  --feats 512 ^
  --save log ^
  --if_labelsmooth ^
  --w_cosine_annealing ^
  --height 384 ^
  --width 128

echo ------------------------- 학습: 2 완료 -------------------------
timeout /t 5 >nul


ECHO 모든 스크립트를 성공적으로 실행했습니다.

:END
PAUSE
