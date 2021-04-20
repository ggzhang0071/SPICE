 python -m pdb  tools/train_moco.py

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_self_v2.py

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/local_consistency.py


#python -m pdb tools/eval_self.py --config-file configs/stl10/eval.py --weight PATH/TO/MODEL --all 1


CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/eval_semi.py --load_path PATH/TO/MODEL --net WideResNet --widen_factor 2 --data_dir PATH/TO/DATA --dataset cifar10 --all 1 
