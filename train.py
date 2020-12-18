r"""PyTorch Detection Training.

To run in a multi-gpu environment, use the distributed launcher::

    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU

The default hyperparameters are tuned for training on 8 gpus and 2 images per gpu.
    --lr 0.02 --batch-size 2 --world-size 8
If you use different number of gpus, the learning rate should be changed to 0.02/8*$NGPU.

On top of that, for training Faster/Mask R-CNN, the default hyperparameters are
    --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3

Also, if you train Keypoint R-CNN, the default hyperparameters are
    --epochs 46 --lr-steps 36 43 --aspect-ratio-group-factor 3
Because the number of images is smaller in the person keypoint subset of COCO,
the number of epochs should be adapted so that we have the same number of iterations.
"""
import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, AnchorGenerator

from coco_utils import get_coco, get_coco_kp

from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from engine import train_one_epoch, evaluate

import utils
import transforms as T


def get_dataset(name, image_set, transform, data_path,num_classes):
    paths = {
        "coco": (data_path, get_coco, int(num_classes)+1), # 修改自定义数据集类别数量：num_classes+1（背景 ）
        "coco_kp": (data_path, get_coco_kp, 2)
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

'''
anchor box优化:https://github.com/martinzlocha/anchor-optimization/
python3 optimize_anchors.py --seed 2020  --objective focal 
/media/gaoya/disk/Documents/zhouwenqing/cotterpin_detect/train_GAN_twoclass/csv_annotations.csv --no-resize
先用脚本转换为csv格式，再计算ratio和scale
Final best anchor configuration
Ratios: [0.789, 1.0, 1.267]
Scales: [0.632, 0.801, 1.0]
'''
def get_mobilnet_v2_model_FRCNN(num_classes):
    '''
	backbone is mobilenet_v2
	'''
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                aspect_ratios=((0.5, 1.0, 2.0),))
    # anchor_generator = AnchorGenerator(sizes=((8, 16, 32, 64, 128, 256, 512),),
	# 							aspect_ratios=((0.5, 1.0, 1,5, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
													output_size=7,
													sampling_ratio=2)
    model = torchvision.models.detection.faster_rcnn.FasterRCNN(backbone,
					num_classes,
					rpn_anchor_generator=anchor_generator,
					box_roi_pool=roi_pooler
                    )

    return  model

def get_squeezenet1_0_model_FRCNN(num_classes):
    '''
	backbone is efficientnet
	'''
    backbone = torchvision.models.squeezenet1_0(pretrained=True).features
    # backbone = EfficientNet.from_pretrained('efficientnet-b7') 
    backbone.out_channels = 512
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                aspect_ratios=((0.5, 1.0, 2.0),))
    # anchor_generator = AnchorGenerator(sizes=((8, 16, 32, 64, 128, 256, 512),),
	# 							aspect_ratios=((0.5, 1.0, 1,5, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
													output_size=7,
													sampling_ratio=2)
    model = torchvision.models.detection.faster_rcnn.FasterRCNN(backbone,
					num_classes,
					rpn_anchor_generator=anchor_generator,
					box_roi_pool=roi_pooler)

    return  model

def get_alexnet_model_FRCNN(num_classes):
    '''
	backbone is efficientnet
	'''
    backbone = torchvision.models.alexnet(pretrained=True).features
    # backbone = EfficientNet.from_pretrained('efficientnet-b7') 
    backbone.out_channels = 256
    # anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                # aspect_ratios=((0.5, 1.0, 2.0),))
    anchor_generator = AnchorGenerator(sizes=((8, 16, 32, 64, 128, 256, 512),),
								aspect_ratios=((0.5, 1.0, 1,5, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
													output_size=7,
													sampling_ratio=2)
    model = torchvision.models.detection.faster_rcnn.FasterRCNN(backbone,
					num_classes,
					rpn_anchor_generator=anchor_generator,
					box_roi_pool=roi_pooler)

    return  model

def get_mnasnet0_5_model_FRCNN(num_classes):
    '''
	backbone is efficientnet
	'''
    backbone = torchvision.models.mnasnet0_5(pretrained=True).layers
    # backbone = EfficientNet.from_pretrained('efficientnet-b7') 
    backbone.out_channels = 1280
    # anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                # aspect_ratios=((0.5, 1.0, 2.0),))
    anchor_generator = AnchorGenerator(sizes=((8, 16, 32, 64, 128),),
								aspect_ratios=((0.5, 1.0, 1,5, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
													output_size=5,
													sampling_ratio=2)
    model = torchvision.models.detection.faster_rcnn.FasterRCNN(backbone,
					num_classes,
					rpn_anchor_generator=anchor_generator,
					box_roi_pool=roi_pooler)

    return  model

class BoxHead(nn.Module):
            def __init__(self, vgg):
                super(BoxHead, self).__init__()
                self.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])
                self.in_features = 4096 # feature out from mlp
            def forward(self, x):
                x = x.flatten(start_dim=1)
                x = self.classifier(x)
                return x

def get_vgg16_model_FRCNN(num_classes):
	'''
	backbone is vgg16
	'''
    # modified from this issue page https://github.com/pytorch/vision/issues/1116
	vgg = torchvision.models.vgg16(pretrained=True)
	backbone = vgg.features[:-1]
	for layer in backbone[:10]:
		for p in layer.parameters():
			p.requires_grad = False
	backbone.out_channels = 512
	# anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                        #   aspect_ratios=((0.5, 1.0, 2.0),))

	anchor_generator = AnchorGenerator(sizes=((2,4,8,16,32, 64, 128, 256,512),),
                                          aspect_ratios=((0.5, 1.0, 1.5, 2.0),))

	roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                       output_size=7,
                                                       sampling_ratio=2)


	box_head = BoxHead(vgg)
	in_features = box_head.in_features
       
	model = torchvision.models.detection.faster_rcnn.FasterRCNN(
                       backbone, # num_classes, #num_classes should be None when box_predictor is specified
                       rpn_anchor_generator = anchor_generator,
                       box_roi_pool = roi_pooler,
                       box_head = box_head,
                       box_predictor = FastRCNNPredictor(in_features, num_classes))
       
	return model

def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # Data loading code
    print("Loading data")

    dataset, num_classes = get_dataset(args.dataset, "trainval", 
                        get_transform(train=True), args.data_path, args.num_classes)
    dataset_test, _ = get_dataset(args.dataset, "test", 
                        get_transform(train=False), args.data_path,args.num_classes)

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, args.batch_size, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    print("Creating model")
    # model = torchvision.models.detection.__dict__[args.model](num_classes=num_classes,
    #                                                           pretrained=args.pretrained)
    if args.model == 'resnet50':
        model = torchvision.models.detection.__dict__['fasterrcnn_resnet50_fpn'](num_classes=num_classes,
                                        pretrained=args.pretrained) # 默认backbone为fasterrcnn_resnet50_fpn
    elif args.model == 'mobilenet_v2':
        model = get_mobilnet_v2_model_FRCNN(num_classes) # backbone为mobilenet_v2
    elif args.model == 'vgg16':
        model = get_vgg16_model_FRCNN(num_classes)  # backbone为vgg16
    elif args.model =='squeezenet1_0':
        model = get_squeezenet1_0_model_FRCNN(num_classes)
    elif args.model =='alexnet':
        model = get_alexnet_model_FRCNN(num_classes)
    elif args.model =='mnasnet0_5':
        model = get_mnasnet0_5_model_FRCNN(num_classes)

    model.to(device)
    print(model)
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        evaluate(model, data_loader_test, device=device)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq)
        lr_scheduler.step()
        if args.output_dir:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args,
                'epoch': epoch},
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))

        # evaluate after every epoch
        evaluate(model, data_loader_test, device=device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--data-path', default=r"D:\simpleod\YOLODataset\CocoDataset", 
                        help='dataset path')
    parser.add_argument('--dataset', default='coco', help='dataset')
    parser.add_argument('--num-classes', default=3, required=False, help='number of classes in dataset')
    parser.add_argument('--model', default='mobilenet_v2', 
                        help='backbone model of fasterrcnn, options are: \
                        resnet50,vgg16,mobilenet_v2,squeezenet1_0,alexnet,mnasnet0_5')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', default=0.02, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                        'on 8 gpus and 2 images_per_gpu')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[16, 22], nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=50, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='checkpoints', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    if args.output_dir:
        utils.mkdir(args.output_dir)

    main(args)
