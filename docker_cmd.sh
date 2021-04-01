img="pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel"

nvidia-docker run --privileged=true  --workdir /git --name "self-learning"  -e DISPLAY --ipc=host -d --rm  -p 8090:8888    \
-v /mnt/di/:/git/dataSet \
-v /disk/zhanggege/SPICE/:/git/SPICE \
$img sleep infinity

