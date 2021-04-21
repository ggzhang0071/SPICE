#!/bin/bash
#img="pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel"
img="spice:0.1"
nvidia-docker run --privileged=true  --workdir /git --name "spice"  -e DISPLAY --ipc=host -d --rm  -p 8098:8888    \
-v /disk/zhanggege/nfs_12/SPICE/:/git/SPICE \
-v /mnt/di/:/git/dataSet \
-v /disk/zhanggege/nfs_12/PaDiM-Anomaly-Detection-Localization-master/kangqiang_result/segment_image_result_wide_resnet50_2:/git/segment_images \
-v /disk/zhanggege/nfs_12/results/self_learning_results/:/git/results \
-v /disk/zhanggege/nfs_12/datasets/:/git/datasets \
$img sleep infinity
