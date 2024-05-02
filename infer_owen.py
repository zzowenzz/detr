import torch
from torch.utils.data import DataLoader

import argparse
from pathlib import Path

from owen_data import build, collate_fn
from owen_model import build_backbone, build_transformer, DETR


def save_detections_in_yolo_format(class_indices_to_keep, label_indices_to_keep, label_path):
    """
    Save the detections in YOLO format in batch size 1.
    Parameters:
    - class_indices_to_keep: 1 d tensor of class indices to keep
    - label_indices_to_keep: 2 d tensor of label indices to keep
    - label_path: path to the label file
    """
    with open(label_path, 'w') as file:
        for class_idx, label in zip(class_indices_to_keep, label_indices_to_keep):
            cx, cy, w, h = label
            file.write(f"{class_idx} {cx} {cy} {w} {h}\n")

def main(args):
    # create output directories
    if not (Path(args.model).parent / ("images_"+ Path(args.model).stem )).exists(): 
        (Path(args.model).parent / ("images_"+ Path(args.model).stem )).mkdir(parents=True)
    if not (Path(args.model).parent / ("labels_"+ Path(args.model).stem )).exists():
        (Path(args.model).parent / ("labels_"+ Path(args.model).stem )).mkdir(parents=True)

    # build dataset
    dataset_train = build(image_set='train', args=args)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=0, collate_fn=collate_fn, shuffle=True)
    print(f"Build val dataset with {len(dataset_train)} images")

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create model
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

    # load pre-trained weights
    checkpoint = torch.load(args.model, map_location=torch.device('cuda:0') if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(checkpoint['model'])

    print("Start inference...")
    with torch.no_grad():
        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            model.eval()

            # forward pass
            pred = model(images)

            probas = pred['pred_logits'].softmax(-1)[0, :, :-1]
            max_probs, class_indices = probas.max(-1)
            keep = max_probs > args.conf_thres
            probas_to_keep = probas[keep]
            class_indices_to_keep = class_indices[keep]
            label_indices_to_keep = pred['pred_boxes'][0,:,:][keep]

            # import pdb; pdb.set_trace()
            # save_detections_in_yolo_format(class_indices_to_keep, label_indices_to_keep, Path(args.model).parent / ("labels_"+ Path(args.model).stem ) / (str(Path(img_name[0]).stem) + ".txt"))
            save_detections_in_yolo_format(class_indices_to_keep, label_indices_to_keep, Path(args.model).parent / ("labels_"+ Path(args.model).stem ) / (str(targets[0]["image_id"].item()) + ".txt"))
            # import pdb; pdb.set_trace()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a baseline model')

    # model
    parser.add_argument("--model", type=str, default="", help="path to the model.")

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
    parser.add_argument('--masks', action='store_true',help="Train segmentation head if the flag is provided")

    # dataset
    parser.add_argument('--coco_path', help='path to COCO dataset')

    # inference
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--num_classes", type=int, required=True , help='Number of classes')
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument("--conf_thres", type=float, default=0.49, help="confidence threshold")
    
    # others
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    
    args = parser.parse_args()
    main(args)