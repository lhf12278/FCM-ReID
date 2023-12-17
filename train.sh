# VIT-S
# '--resume' is the model parameters pre-trained on the baseline after 50 epochs
CUDA_VISIBLE_DEVICES=0 python train.py -b 256 -a vit_small -d market1501 --iters 200 --eps 0.6 --self-norm \
--use-hard --hw-ratio 2 --num-instances 8 --temperature 1.0 --layers 1 --k 16 --w 1.0 \
-pp /home/a/data/hqs/reid_data/vit_small_cfs_lup.pth --logs-dir ./log_market1 \
--resume /home/a/data/hqs_2023_v2_2/code1/log_base/market1/model_best.pth.tar

CUDA_VISIBLE_DEVICES=0 python train.py -b 256 -a vit_small -d msmt17 --iters 200 --eps 0.7 --self-norm \
--use-hard --hw-ratio 2 --num-instances 8 --temperature 1.0 --layers 1 --k 32 --w 0.0 \
-pp /home/a/data/hqs/reid_data/vit_small_cfs_lup.pth --logs-dir ./log_msmt2 \
--resume /home/a/data/hqs_2023_v2_2/code1/log_base/msmt1/model_best.pth.tar

CUDA_VISIBLE_DEVICES=1 python train.py -b 256 -a vit_small -d duke --iters 200 --eps 0.8 --self-norm \
--use-hard --hw-ratio 2 --num-instances 8 --temperature 0.05 --layers 1 --k 8 --w 0.0 \
-pp /home/a/data/hqs/reid_data/vit_small_cfs_lup.pth --logs-dir ./log_duke1 \
--resume /home/a/data/hqs_2023_v2_2/code1/FCM_v1/log_base/log_duke1/checkpoint50.pth.tar





