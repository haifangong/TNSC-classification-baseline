import argparse
import os

import torch
from resnest.torch import resnest50
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.resnet import resnet34, resnet18, resnet50, resnet101
from tqdm import tqdm

from dataloaders import custom_transforms as trforms
from dataloaders import tnsc_dataset


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default='0')

    # Model setting
    parser.add_argument('-backbone', type=str, default='resnest50')
    parser.add_argument('-input_size', type=int, default=224)

    # Training setting
    parser.add_argument('-seed', type=int, default=1234)
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-nepochs', type=int, default=100)
    parser.add_argument('-resume_epoch', type=int, default=0)
    parser.add_argument('-train_fold', type=str, default='CROP')
    parser.add_argument('-run_id', type=int, default=-1)

    # Optimizer setting
    parser.add_argument('-lr', type=float, default=1e-2)
    parser.add_argument('-weight_decay', type=float, default=5e-4)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-update_lr_every', type=int, default=20)

    parser.add_argument('-save_every', type=int, default=2)
    parser.add_argument('-log_every', type=int, default=100)

    return parser.parse_args()


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.backbone == 'resnet18':
        backbone = resnet18(nInputChannels=3, os=32, pretrained=True)
    elif args.backbone == 'resnet34':
        backbone = resnet34(num_classes=2, pretrained=False)
    elif args.backbone == 'resnet50':
        backbone = resnet50(nInputChannels=3, os=32, pretrained=True)
    elif args.backbone == 'resnet101':
        backbone = resnet101(nInputChannels=3, os=32, pretrained=True)
    elif args.backbone == 'resnest50':
        backbone = resnest50(num_classes=2, pretrained=False)
    else:
        raise NotImplementedError

    backbone.load_state_dict(torch.load('/home/duadua/TNSC/classifier/run/CROP-resnest50-Tricks/run_1/models/best_backbone_e61.pth', map_location=lambda storage, loc: storage))
    torch.cuda.set_device(device=0)
    backbone.cuda()

    composed_transforms_ts = transforms.Compose([
        trforms.FixedResizeI(size=(args.input_size, args.input_size)),
        trforms.NormalizeI(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        trforms.ToTensorI()])

    testset = tnsc_dataset.TNSCDataset(mode='test', transform=composed_transforms_ts, return_size=False)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
    backbone.eval()

    f = open('classification.csv', 'w')
    f.write('ID,CATE\n')
    for sample_batched in tqdm(testloader):
        img = sample_batched['image'].cuda()
        feats = backbone.forward(img)
        pred = torch.argmax(feats, dim=1, keepdim=False)
        f.write(sample_batched['label_name'][0]+','+str(pred.item())+'\n')


if __name__ == '__main__':
    args = get_arguments()
    main(args)
