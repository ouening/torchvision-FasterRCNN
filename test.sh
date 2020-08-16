#!/bin/bash

python3 train.py --data-path /media/gaoya/disk/Datasets/ObjectDetection/TerahertzImages/YOLOAugumented/VOC2020/CocoFormat \
--dataset coco \
--num-classes  8 \
--batch-size 4 \
--test-only \
--resume checkpoints/terahertz-8cls-model_29.pth
