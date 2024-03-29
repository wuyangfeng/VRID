from __future__ import print_function, absolute_import
import os
import sys
import time
import datetime
import argparse
import scipy.io as sio
import os.path as osp
import numpy as np
import IPython

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

# Using tensorboardX
# from tensorboardX import SummaryWriter

import data_manager
from dataset_loader import ImageDataset
import transforms as T
import models
from evaluate_rerank import rerank_main
from random_erasing import RandomErasing
from losses import CrossEntropyLabelSmooth, TripletLoss, DeepSupervision
from utils import AverageMeter, Logger, save_checkpoint
from eval_metrics import evaluate
from samplers import RandomIdentitySampler
from optimizers import init_optim

parser = argparse.ArgumentParser(description='Train image model with cross entropy loss and hard triplet loss')
# Datasets
parser.add_argument('--root', type=str, default='data', help="root path to data directory")
parser.add_argument('-d', '--dataset', type=str, default='market1501',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=256,
                    help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=128,
                    help="width of an image (default: 128)")
parser.add_argument('--split-id', type=int, default=0, help="split index")
# CUHK03-specific setting
parser.add_argument('--cuhk03-labeled', action='store_true',
                    help="whether to use labeled images, if false, detected images are used (default: False)")
parser.add_argument('--cuhk03-classic-split', action='store_true',
                    help="whether to use classic split by Li et al. CVPR'14 (default: False)")
parser.add_argument('--use-metric-cuhk03', action='store_true',
                    help="whether to use cuhk03-metric (default: False)")
# Optimization options
parser.add_argument('--optim', type=str, default='adam', help="optimization algorithm (see optimizers.py)")
parser.add_argument('--max-epoch', default=180, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--train-batch', default=32, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=32, type=int, help="test batch size")
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    help="initial learning rate")
parser.add_argument('--stepsize', default=60, type=int,
                    help="stepsize to decay learning rate (>0 means this is enabled)")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")
parser.add_argument('--margin', type=float, default=0.3, help="margin for triplet loss")
parser.add_argument('--num-instances', type=int, default=4,
                    help="number of instances per identity")
parser.add_argument('--htri-only', action='store_true', default=False,
                    help="if this is True, only htri loss is used in training")
# Architecture
parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.get_names())
# Miscs
parser.add_argument('--print-freq', type=int, default=10, help="print frequency")
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--eval-step', type=int, default=-1,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--start-eval', type=int, default=0, help="start to evaluate after specific epoch")
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--use-cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--warmup', action='store_true', default=False, help='use warmup')
parser.add_argument('--loss', type=str, default='xent')
parser.add_argument('--probability', type=float, default=0.5, help='')
parser.add_argument('--random-erasing',action='store_true',help='Use Random Erasing')
parser.add_argument('--save-fea',action='store_true',help='Save features')
parser.add_argument('--l2-reg',action='store_true',help='L2 Softmax')

args = parser.parse_args()

def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))

    # tensorboardX
    # writer = SummaryWriter(log_dir=osp.join(args.save_dir,'summary'))

    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_img_dataset(
        root=args.root, name=args.dataset, split_id=args.split_id,
        cuhk03_labeled=args.cuhk03_labeled, cuhk03_classic_split=args.cuhk03_classic_split,
    )

    transform_train = T.Compose([
        T.Random2DTranslation(args.height, args.width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    if args.random_erasing:
        transform_train = T.Compose([
            T.Random2DTranslation(args.height, args.width),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomErasing(probability=args.probability, mean=[0.0, 0.0, 0.0]),
        ])
        

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pin_memory = True if use_gpu else False

    if args.loss == 'xent,htri':
        trainloader = DataLoader(
            ImageDataset(dataset.train, transform=transform_train),
            sampler=RandomIdentitySampler(dataset.train, num_instances=args.num_instances),
            batch_size=args.train_batch, num_workers=args.workers,
            pin_memory=pin_memory, drop_last=True,
        )
    elif args.loss == 'xent':
        trainloader = DataLoader(
            ImageDataset(dataset.train, transform=transform_train),
            batch_size=args.train_batch, shuffle=True, num_workers=args.workers,
            pin_memory=pin_memory, drop_last=True,
        )

    queryloader = DataLoader(
        ImageDataset(dataset.query, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    galleryloader = DataLoader(
        ImageDataset(dataset.gallery, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids, loss=args.loss)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

    criterion_xent = CrossEntropyLabelSmooth(num_classes=dataset.num_train_pids, use_gpu=use_gpu)
    criterion_htri = TripletLoss(margin=args.margin)
    
    optimizer = init_optim(args.optim, model.parameters(), args.lr, args.weight_decay)
    if args.stepsize > 0:
        if not args.warmup:
            scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
    start_epoch = args.start_epoch

    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if args.evaluate:
        print("Evaluate only")
        test(model, queryloader, galleryloader, use_gpu)
        return
    def adjust_lr(optimizer, ep):
        if ep < 20:
            lr = 1e-4 * (ep + 1) / 2
        elif ep < 80:
            #lr = 1e-3 * len(args.gpu_devices)
            lr = 1e-3
        elif ep < 180:
            #lr = 1e-4 * len(args.gpu_devices)
            lr = 1e-4
        elif ep < 300:
            #lr = 1e-5 * len(args.gpu_devices)
            lr = 1e-5
        elif ep < 320:
            #lr = 1e-5 * 0.1 ** ((ep - 320) / 80) * len(args.gpu_devices)
            lr = 1e-5 * 0.1 ** ((ep - 320) / 80)
        elif ep < 400:
            lr = 1e-6
        elif ep < 480:
            #lr = 1e-4 * len(args.gpu_devices)
            lr = 1e-4
        else:
            #lr = 1e-5 * len(args.gpu_devices)
            lr = 1e-5
        for p in optimizer.param_groups:
            p['lr'] = lr
    
    length = len(trainloader)
    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    #best_rerank1 = -np.inf
    #best_rerankepoch = 0
    print("==> Start training")

    for epoch in range(start_epoch, args.max_epoch):
        start_train_time = time.time()
        if args.stepsize > 0:
            if args.warmup:
                adjust_lr(optimizer, epoch + 1)
            else:
                scheduler.step()
        train(epoch, model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu=use_gpu, summary=None, length=length)
        train_time += round(time.time() - start_train_time)
        
        if (epoch+1) > args.start_eval and args.eval_step > 0 and (epoch+1) % args.eval_step == 0 or (epoch+1) == args.max_epoch:
            print("==> Test")
            rank1 = test(epoch, model, queryloader, galleryloader, use_gpu=True, summary=None)
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1
            ####### Best Rerank
            #is_rerankbest = rerank1 > best_rerank1
            #if is_rerankbest:
            #    best_rerank1 = rerank1
            #    best_rerankepoch = epoch + 1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            save_checkpoint({
                'state_dict': state_dict,
                'rank1': rank1,
                'epoch': epoch,
            }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch+1) + '.pth.tar'))

    writer.close()
    print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))
    #print("==> Best Rerank-1 {:.1%}, achieved at epoch {}".format(best_rerank1, best_rerankepoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))

def train(epoch, model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu=True, summary=None, length=0):
    losses = AverageMeter()
    xentlosses = AverageMeter()
    htrilosses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()

    end = time.time()
    for batch_idx, (imgs, pids, _) in enumerate(trainloader):
        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()

        # measure data loading time
        data_time.update(time.time() - end)
        
        if args.loss == 'xent,htri':
            outputs, features = model(imgs)
        elif args.loss == 'xent':
            outputs = model(imgs)
        # use l2-softmax    
        if args.l2_reg:
            # L2 norm
            outputs = outputs/torch.norm(outputs, dim=1, keepdim=True)
            features = features/torch.norm(features, dim=1, keepdim=True)
            # scale
            outputs = outputs * 32
            # features = features * 64
        if args.htri_only:
            if isinstance(features, tuple):
                loss = DeepSupervision(criterion_htri, features, pids)
            else:
                loss = criterion_htri(features, pids)
        elif args.loss == 'xent,htri':
            if isinstance(outputs, tuple):
                xent_loss = DeepSupervision(criterion_xent, outputs, pids)
            else:
                xent_loss = criterion_xent(outputs, pids)
            
            if isinstance(features, tuple):
                htri_loss = DeepSupervision(criterion_htri, features, pids)
            else:
                htri_loss = criterion_htri(features, pids)
        else:
            if isinstance(outputs, tuple):
                xent_loss = DeepSupervision(criterion_xent, outputs, pids)
            else:
                xent_loss = criterion_xent(outputs, pids)
        if args.loss == 'xent,htri':
            loss = xent_loss + htri_loss
        elif args.loss == 'xent':
            loss = xent_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if args.loss == 'xent':
            xentlosses.update(xent_loss.item(),pids.size(0))
        elif args.loss == 'xent,htri':
            xentlosses.update(xent_loss.item(),pids.size(0))
            htrilosses.update(htri_loss.item(),pids.size(0))
        losses.update(loss.item(), pids.size(0))

        # add the losses to summary writer
        if summary is not None:
            summary.add_scalars('loss', {'Total loss': loss.item(), 'xentloss': xent_loss.item(), 'htriloss': htri_loss.item()}, length * epoch + batch_idx)
            summary.add_scalar('lr', optimizer.param_groups[0]['lr'], length * epoch + batch_idx)

        if (batch_idx+1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Lr: {3:.2e}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch+1, batch_idx+1, len(trainloader), optimizer.param_groups[0]['lr'], batch_time=batch_time,data_time=data_time, loss=losses))
            if args.loss == 'xent,htri':
                print('XentLoss: {xentlosses.val:.4f} ({xentlosses.avg:.4f})\t'
                      'HtriLoss: {htrilosses.val:.4f} ({htrilosses.avg:.4f})\t'.format(xentlosses=xentlosses, htrilosses=htrilosses))


def test(epoch, model, queryloader, galleryloader, use_gpu=True, ranks=[1, 5, 10, 20], summary=None):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = extract_feature(imgs,model)
            # features = features/torch.norm(features,p=2,dim=1,keepdim=True)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
            if use_gpu:
                imgs = imgs.cuda()
            
            end = time.time()
            features = extract_feature(imgs,model)
            # features = features/torch.norm(features,p=2,dim=1,keepdim=True)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)
        if args.save_fea:
            print("Saving result.mat to {}".format(args.save_dir))
            result = {'gallery_f':gf.numpy(),'gallery_label':g_pids,'gallery_cam':g_camids,'query_f':qf.numpy(),'query_label':q_pids,'query_cam':q_camids}
            sio.savemat(os.path.join(args.save_dir, 'dp_result.mat'),result)
            return

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
    
    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, args.test_batch))

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=args.use_metric_cuhk03)

    print("----------Results-----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
    #print("---- Start Reranking... ----")
    #rerank_cmc, rerank_mAP = rerank_main(qf,q_camids,q_pids,gf,g_camids,g_pids)
    #print("Rerank_mAP: {:.1%}".format(rerank_mAP))
    #print("Rerank_CMC curve")
    #for r in ranks:
    #    print("Rank-{:<3}: {:.1%}".format(r, rerank_cmc[r-1]))
    print("----------------------------")
    if summary is not None:
        summary.add_scalars('rank result', {'rank1': round(cmc[0],3) * 100, 'rank5': round(cmc[1],3) * 100, 'mAP': round(mAP,3) * 100}, epoch)
    return cmc[0] #rerank_cmc[0]


#########################3
def fliphor(inputs):
    inv_idx = torch.arange(inputs.size(3)-1,-1,-1).long()  # N x C x H x W
    return inputs.index_select(3,inv_idx.cuda())

def extract_feature(inputs,model):
    #features = torch.FloatTensor()
    ff = torch.FloatTensor(inputs.size(0), 2048).zero_()
    ff = ff.cuda()
    for i in range(2):
        if i==1:
            inputs = fliphor(inputs)
        outputs = model(inputs)
        ff = ff + outputs
    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
    features = ff.div(fnorm.expand_as(ff))
    return features

if __name__ == '__main__':
    main()
