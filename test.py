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
from tensorboardX import SummaryWriter

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
parser.add_argument('--resume-all', type=str, default='', metavar='PATH')
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

    sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))

    # tensorboardX
    writer = SummaryWriter(log_dir=osp.join(args.save_dir,'summary'))

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

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pin_memory = True if use_gpu else False

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

    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

        if use_gpu:
            model = nn.DataParallel(model).cuda()

    start_time = time.time()
    if args.resume_all:
        print("Loading all checkpoints from '{}'".format(args.resume_all))
        pths = glob.glob(osp.join(args.resume_all,'checkpoint_ep*.tar'))
        best_epoch = 0
        best_rank1 = -np.inf
        # best_rerankepoch = 0
        # best_rerank1 = -np.inf
        for pth in pths:
            epoch = list(map(int, re.findall(pattern=r'ep(\d+)\.pth',string=pth)))
            print("Test epoch {}".format(epoch[0]))
            checkpoint = torch.load(pth)
            model.load_state_dict(checkpoint['state_dict'])
            if use_gpu:
                model = nn.DataParallel(model).cuda()
            rank1 = test(epoch, model, queryloader, galleryloader, use_gpu=True, summary=writer)
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch[0]
            model = model.module
        shutil.copyfile(args.resume_all + 'checkpoint_ep' + str(best_epoch) + '.pth.tar', args.resume_all + 'best_checkpoint.pth.tar')
        print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))


    test_time = round(time.time() - start_time)
    test_time = str(datetime.timedelta(seconds=test_time))  
    print("Finished. Testtime (h:m:s): {}.".format(test_time))

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
