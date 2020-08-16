#!/bin/bash

python3 train.py --data-path /media/gaoya/disk/Datasets/ObjectDetection/输电线路缺陷/CotterPin2classes/CocoFormat \
--dataset coco \
--num-classes  8 \
--batch-size 4 \
--epochs 50
