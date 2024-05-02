# demo model
# Copyright (c) Owen Xing @ owenxing1994@gmail.com
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

import torchvision
from torch.utils.data import DataLoader


import argparse

from pathlib import Path
import logging
import os
import shutil
import random
import numpy as np
from PIL import Image
import cv2
import math

from torchvision.ops.boxes import box_area
import torch.nn.functional as F
import torchvision
import torch.distributed as dist
from packaging import version
if version.parse(torchvision.__version__) < version.parse('0.7'):
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size

from scipy.optimize import linear_sum_assignment

from owen_data import build, collate_fn
from owen_model import build_backbone, build_transformer, DETR
# from owen_criterion import build_matcher, SetCriterion, reduce_dict
from det_loss import build_matcher, SetCriterion, reduce_dict


def main(args):
     # create logger
    if not os.path.exists("./log/"+args.name):
        os.makedirs("./log/"+args.name)
    else:
        # stop the program if the log folder already exists
        raise Exception(f"Log folder {args.name} already exists. Please use another name.")
    log_filename = f'./log/{args.name}/training_log.log'
    logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    logging.info(f"Create logger at {log_filename}")
    logging.info(f"Choose arguments: {args}")

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Set device to {device}")

    # fix the seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    logging.info(f"Fix the seed {seed}")

    # build dataset
    dataset_train = build(image_set='train', args=args)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=0, collate_fn=collate_fn, shuffle=False)
    logging.info(f"Build train dataset with {len(dataset_train)} images")

    # build backbone
    backbone = build_backbone(args)
    # build transformer
    transformer = build_transformer(args)
    model = DETR(
        backbone,
        transformer,
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    ).to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Build model. Number of parameters: {n_parameters}")

    # resume training
    if args.resume:
        checkpoint = torch.load(args.resume,  map_location=torch.device('cuda:0') if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(checkpoint['model'])
        logging.info(f"Load model from {args.resume}")

    # build criterion
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    criterion_det = SetCriterion(args.num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion_det.to(device)

    # build optimizer and lr scheduler
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    logging.info(f"Build optimizer with lr: {args.lr}, lr_backbone: {args.lr_backbone}, weight_decay: {args.weight_decay}, lr_drop: {args.lr_drop}")

    logging.info("Start training ...")
    for epoch in range(1, args.epochs+1):
        acc_det = 0
        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            model.train()
            criterion_det.train()

            model.zero_grad()
            optimizer.zero_grad()

            # forward prop
            outputs = model(images)
            
            # calculate loss
            
            loss_dict = criterion_det(outputs, targets)
            weight_dict = criterion_det.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_dict(loss_dict)
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                        for k, v in loss_dict_reduced.items()}
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
            loss_value = losses_reduced_scaled.item()
            acc_det += loss_value

            if args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
            optimizer.step()
        
        lr_scheduler.step()
        
        acc_det /= len(train_loader)
        logging.info(f"Epoch {epoch}/{args.epochs} - loss: {acc_det:2f}")

        if epoch % args.save_freq == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args
            }
            torch.save(checkpoint, f'./log/{args.name}/checkpoint_{epoch}.pth')
        
             

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script')

    # train
    parser.add_argument('--name', type=str, help='The name of the experiment')
    parser.add_argument('--coco_path', help='path to COCO dataset')
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument("--num_classes", type=int, required=True , help='Number of classes')
    parser.add_argument("--resume", type=str, help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='The random seed')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument("--save_freq", type=int, default=5, help="save every few epochs.")

    # detection
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--pre_norm', action='store_true')

    #Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    
    args = parser.parse_args()

    main(args)