import socket
import argparse
from datetime import datetime
import time
import os
import glob

import torch
from torch import nn
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from tensorboardX import SummaryWriter

from dataloaders import tnsc_dataset
from dataloaders import custom_transforms as trforms

# from resnest.torch import resnest50
from torchvision.models.resnet import resnet34, resnet18, resnet50, resnet101
import utils

from model.resnest50 import resnest50


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default='0')

    # Model setting
    parser.add_argument('-backbone', type=str, default='resnest50')
    parser.add_argument('-input_size', type=int, default=224)
    parser.add_argument('-pretrain', type=str, default='../pre_train/resnest50-528c19ca.pth')

    # Training setting
    parser.add_argument('-seed', type=int, default=1234)
    parser.add_argument('-classes', type=int, default=2)
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-nepochs', type=int, default=50)
    parser.add_argument('-resume_epoch', type=int, default=0)
    parser.add_argument('-train_fold', type=str, default='five-fold')
    parser.add_argument('-run_id', type=int, default=-1)

    # Optimizer setting
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('-weight_decay', type=float, default=5e-4)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-update_lr_every', type=int, default=20)

    parser.add_argument('-save_every', type=int, default=10)
    parser.add_argument('-log_every', type=int, default=25)

    parser.add_argument('-fold', type=int, default=4)


    # tricks
    parser.add_argument('--mixup', action='store_true', default=True,
                        help='whether train the model with mix-up. default is false.')
    parser.add_argument('--mixup_alpha', type=float, default=0.2,
                        help='beta distribution parameter for mixup sampling, default is 0.2.')
    parser.add_argument('--mixup-off-epoch', type=int, default=0,
                        help='how many last epochs to train without mixup, default is 0.')
    parser.add_argument('--label_smooth', action='store_true', default=True,
                        help='use label smoothing or not in training. default is false.')

    parser.add_argument('--warmup', action='store_true', default=True,
                        help='whether train the model with warmup. default is false.')
    parser.add_argument('--warmup_epoch', action='store_true', default=5,
                        help='when to train the model with warmup. default is 5.')

    return parser.parse_args()


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.backbone == 'resnet18':
        backbone = resnet18(nInputChannels=3, os=32, pretrained=True)
    elif args.backbone == 'resnet34':
        backbone = resnet34(num_classes=args.classes, pretrained=False)
    elif args.backbone == 'resnet50':
        backbone = resnet50(nInputChannels=3, os=32, pretrained=True)
    elif args.backbone == 'resnet101':
        backbone = resnet101(nInputChannels=3, os=32, pretrained=True)
    elif args.backbone == 'resnest50':
        backbone = resnest50(num_classes=args.classes)
    else:
        raise NotImplementedError

    backbone = utils.load_pretrain_model(backbone, torch.load(args.pretrain))
    save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    if args.resume_epoch != 0:
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', args.train_fold, 'run_*')))
        run_id = int(runs[-1].split('_')[-1]) if runs else 0
    else:
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', args.train_fold, 'run_*')))
        run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

    if args.run_id >= 0:
        run_id = args.run_id

    save_dir = os.path.join(save_dir_root, 'run', args.train_fold, 'run_' + str(run_id))
    log_dir = os.path.join(save_dir, datetime.now().strftime('%b%d_%H-%M-%M%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    logger = open(os.path.join(save_dir, 'log.txt'), 'w')
    logger.write('optim: SGD \nlr=%.4f\nweight_decay=%.4f\nmomentum=%.4f\nupdate_lr_every=%d\nseed=%d\n' %
                 (args.lr, args.weight_decay, args.momentum, args.update_lr_every, args.seed))

    if not os.path.exists(os.path.join(save_dir, 'models')):
        os.makedirs(os.path.join(save_dir, 'models'))

    if args.resume_epoch == 0:
        print('Training from scratch...')
    else:
        backbone_resume_path = os.path.join(save_dir, 'models', 'backbone_epoch-' + str(args.resume_epoch - 1) + '.pth')
        print('Initializing weights from: {}, epoch: {}...'.format(save_dir, args.resume_epoch))
        backbone.load_state_dict(torch.load(backbone_resume_path, map_location=lambda storage, loc: storage))

    torch.cuda.set_device(device=0)
    backbone.cuda()
    params = utils.split_weights(backbone)

    backbone_optim = optim.SGD(
        # params,
        backbone.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True
    )

    composed_transforms_tr = transforms.Compose([
        trforms.FixedResize(size=(args.input_size + 8, args.input_size + 8)),
        trforms.RandomCrop(size=(args.input_size, args.input_size)),
        trforms.RandomHorizontalFlip(),
        # trforms.RandomRotate(degree=15),
        # trforms.RandomRotateOrthogonal(),
        trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        trforms.ToTensor()])

    composed_transforms_ts = transforms.Compose([
        trforms.FixedResize(size=(args.input_size, args.input_size)),
        trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        trforms.ToTensor()])

    trainset = tnsc_dataset.TNSCDataset(mode='train', fold=args.fold, transform=composed_transforms_tr, return_size=False)
    valset = tnsc_dataset.TNSCDataset(mode='val', fold=args.fold, transform=composed_transforms_ts, return_size=False)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=2)

    train_scheduler = optim.lr_scheduler.CosineAnnealingLR(backbone_optim, args.nepochs - args.warmup_epoch)
    warmup_scheduler = utils.WarmUpLR(backbone_optim, len(trainloader) * args.warmup_epoch)


    num_iter_tr = len(trainloader)
    nitrs = args.resume_epoch * num_iter_tr
    nsamples = args.batch_size * nitrs
    print('each_epoch_num_iter: %d' % (num_iter_tr))

    # lossF = utils.FocalLossV1()

    global_step = 0
    best_acc = 0

    recent_losses = []
    start_t = time.time()
    print('Training Network')


    for epoch in range(args.resume_epoch, args.nepochs):
        # if epoch <= args.warmup_epoch:
        #     train_scheduler.step(epoch)
        backbone.train()
        epoch_losses = []

        for ii, sample_batched in enumerate(trainloader):
            # if epoch <= args.warmup_epoch:
            #     warmup_scheduler.step()
            img, label = sample_batched['image'], sample_batched['label']
            img, label = img.cuda(), label.cuda()
            global_step += args.batch_size

            if args.mixup:
                inputs, targets_a, targets_b, lam = utils.mixup_data(img, label, args.mixup_alpha, use_cuda=True)
                loss_func = utils.mixup_criterion(targets_a, targets_b, lam)
                outputs = backbone.forward(inputs)
                if args.label_smooth:
                    criterion = utils.LabelSmoothingCrossEntropy()
                else:
                    # criterion = utils.FocalLossV1()
                    criterion = nn.CrossEntropyLoss()
                loss = loss_func(criterion, outputs)

            else:
                feats = backbone.forward(img)
                # loss = lossF(feats, label)
                loss = utils.CELoss(logit=feats, target=label, reduction='mean')

            trainloss = loss.item()
            epoch_losses.append(trainloss)
            if len(recent_losses) < args.log_every:
                recent_losses.append(trainloss)
            else:
                recent_losses[nitrs % len(recent_losses)] = trainloss
            backbone_optim.zero_grad()
            loss.backward()
            backbone_optim.step()

            nitrs += 1
            nsamples += args.batch_size

            if nitrs % args.log_every == 0:
                meanloss = sum(recent_losses) / len(recent_losses)

                print('epoch: %d ii: %d trainloss: %.2f timecost:%.2f secs' % (
                    epoch, ii, meanloss, time.time() - start_t))
                writer.add_scalar('data/trainloss', meanloss, nsamples)

            if ii % (num_iter_tr // 10) == 0:
                grid_image = make_grid(img[:3].clone().cpu().data, 3, normalize=True)
                writer.add_image('Image', grid_image, global_step)

        # validation
        backbone.eval()
        acc = 0.0

        for ii, sample_batched in enumerate(valloader):
            img, label = sample_batched['image'], sample_batched['label']

            img, label = img.cuda(), label.cuda()
            feats = backbone.forward(img)

            if torch.argmax(feats, dim=1, keepdim=False) == label:
                acc += 1
        acc /= len(valset)

        if acc > best_acc:
            best_acc = acc
            backbone_save_path = os.path.join(save_dir, 'models', 'best_backbone_e' + str(epoch) + '.pth')
            torch.save(backbone.state_dict(), backbone_save_path)
            print("Save best backbone at {}\n".format(backbone_save_path))

        print('Validation:')
        print('epoch: %d, images: %d acc: %.4f' % (epoch, len(valset), acc))
        logger.write('epoch: %d, images: %d acc: %.4f' % (epoch, len(valset), acc))

        writer.add_scalar('data/valid_acc', acc, nsamples)

        if epoch % args.save_every == args.save_every - 1:
            backbone_save_path = os.path.join(save_dir, 'models', 'backbone_epoch-' + str(epoch) + '.pth')
            torch.save(backbone.state_dict(), backbone_save_path)
            print("Save backbone at {}\n".format(backbone_save_path))

        if epoch % args.update_lr_every == args.update_lr_every - 1:
            lr_ = utils.lr_poly(args.lr, epoch, args.nepochs, 0.9)
            print('(poly lr policy) learning rate: ', lr_)
            backbone_optim = optim.SGD(
                backbone.parameters(),
                lr=lr_,
                momentum=args.momentum,
                weight_decay=args.weight_decay
            )


if __name__ == '__main__':
    args = get_arguments()
    main(args)
