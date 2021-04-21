timestamp=`date +%Y%m%d%H%M%S`
rm Logs/*.log
 
python   tools/train_moco.py  --data_type kangqiang_segment_results  --data '/git/segment_images'  --batch-size 512 \
--epochs 1000 --save_folder "/git/results/kangqiang" 2>&1 |tee Logs/$timestamp.log

python tools/pre_compute_embedding.py

#python tools/train_self_v2.py  

 #python tools/local_consistency.py


#python ./tools/train_semi.py --unlabeled 1 --num_classes 10 --num_workers 4 --dist-url tcp://localhost:10001 --label_file ./results/stl10/eval/labels_reliable_0.983136_6760.npy --save_dir ./results/stl10/spice_semi --save_name 098_6760 --batch_size 64  --net WideResNet_stl10 --data_dir ./datasets/stl10 --dataset stl10


#python tools/eval_self.py --config-file configs/stl10/eval.py --weight PATH/TO/MODEL --all 1


#CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/eval_semi.py --load_path PATH/TO/MODEL --net WideResNet --widen_factor 2 --data_dir PATH/TO/DATA --dataset cifar10 --all 1 
