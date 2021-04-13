# coding=utf-8
import os
import datetime
import argparse
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.utils.data as torchdata
from torchvision import datasets, models
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

import transforms
from eval_model import *
from models.LoadModel import MainModel
from dataset_DCL import collate_fn4train, collate_fn4val, collate_fn4test, collate_fn4backbone, dataset
from utils import *
import pdb

os.environ['CUDA_DEVICE_ORDRE'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


# parameters setting
def parse_args():
    parser = argparse.ArgumentParser(description='dcl parameters')
    parser.add_argument('--data', dest='dataset',
                        default='CUB', type=str)
    parser.add_argument('--save', dest='resume',
                        default=None,
                        type=str)
    parser.add_argument('--backbone', dest='backbone',
                        default='resnet50', type=str)
    parser.add_argument('--auto_resume', dest='auto_resume',
                        action='store_true')
    parser.add_argument('--epoch', dest='epoch',
                        default=360, type=int)
    parser.add_argument('--tb', dest='train_batch',
                        default=16, type=int)
    parser.add_argument('--vb', dest='val_batch',
                        default=512, type=int)
    parser.add_argument('--sp', dest='save_point',
                        default=5000, type=int)
    parser.add_argument('--cp', dest='check_point',
                        default=5000, type=int)
    parser.add_argument('--lr', dest='base_lr',
                        default=0.0008, type=float)
    parser.add_argument('--lr_step', dest='decay_step',
                        default=60, type=int)
    parser.add_argument('--cls_lr_ratio', dest='cls_lr_ratio',
                        default=10.0, type=float)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        default=0, type=int)
    parser.add_argument('--tnw', dest='train_num_workers',
                        default=16, type=int)
    parser.add_argument('--vnw', dest='val_num_workers',
                        default=32, type=int)
    parser.add_argument('--detail', dest='discribe',
                        default='', type=str)
    parser.add_argument('--size', dest='resize_resolution',
                        default=512, type=int)
    parser.add_argument('--crop', dest='crop_resolution',
                        default=448, type=int)
    parser.add_argument('--cls_2', dest='cls_2',
                        action='store_true')
    parser.add_argument('--cls_mul', dest='cls_mul',
                        action='store_true')
    parser.add_argument('--swap_num', default=[7, 7],
                        nargs=2, metavar=('swap1', 'swap2'),
                        type=int, help='specify a range')
    args = parser.parse_args()
    return args


def auto_load_resume(load_dir):
    folders = os.listdir(load_dir)
    date_list = [int(x.split('_')[1].replace(' ', 0)) for x in folders]
    choosed = folders[date_list.index(max(date_list))]
    weight_list = os.listdir(os.path.join(load_dir, choosed))
    acc_list = [x[:-4].split('_')[-1] if x[:7] == 'weights' else 0 for x in weight_list]
    acc_list = [float(x) for x in acc_list]
    choosed_w = weight_list[acc_list.index(max(acc_list))]
    return os.path.join(load_dir, choosed, choosed_w)


def dt():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")


def load_data_transformers(resize_reso=512, crop_reso=448, swap_num=[7, 7]):
    center_resize = 600
    Normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    data_transforms = {
        'swap': transforms.Compose([
            transforms.Randomswap((swap_num[0], swap_num[1])),
        ]),
        'common_aug': transforms.Compose([
            transforms.Resize((resize_reso, resize_reso)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomCrop((crop_reso, crop_reso)),
            transforms.RandomHorizontalFlip(),
        ]),
        'train_totensor': transforms.Compose([
            transforms.Resize((crop_reso, crop_reso)),
            # ImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val_totensor': transforms.Compose([
            transforms.Resize((crop_reso, crop_reso)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'test_totensor': transforms.Compose([
            transforms.Resize((resize_reso, resize_reso)),
            transforms.CenterCrop((crop_reso, crop_reso)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'None': None,
    }
    return data_transforms


class LoadConfig(object):
    def __init__(self, args, version):
        if version == 'train':
            get_list = ['train', 'val']
        elif version == 'val':
            get_list = ['val']
        elif version == 'test':
            get_list = ['test']
        else:
            raise Exception("train/val/test ???\n")

        ###############################
        #### add dataset info here ####
        ###############################

        # put image data in $PATH/data
        # put annotation txt file in $PATH/anno

        if args.dataset == 'product':
            self.dataset = args.dataset
            self.rawdata_root = './../FGVC_product/data'
            self.anno_root = './../FGVC_product/anno'
            self.numcls = 2019
        elif args.dataset == 'CUB':
            self.dataset = args.dataset
            self.rawdata_root = './dataset/CUB_200_2011/data'
            self.anno_root = './dataset/CUB_200_2011/anno'
            self.numcls = 200
        elif args.dataset == 'STCAR':
            self.dataset = args.dataset
            self.rawdata_root = './dataset/st_car/data'
            self.anno_root = './dataset/st_car/anno'
            self.numcls = 196
        elif args.dataset == 'AIR':
            self.dataset = args.dataset
            self.rawdata_root = './dataset/aircraft/data'
            self.anno_root = './dataset/aircraft/anno'
            self.numcls = 100
        else:
            raise Exception('dataset not defined ???')

        # annotation file organized as :
        # path/image_name cls_num\n

        if 'train' in get_list:
            self.train_anno = pd.read_csv(os.path.join(self.anno_root, 'ct_train.txt'), \
                                          sep=" ", \
                                          header=None, \
                                          names=['ImageName', 'label'])

        if 'val' in get_list:
            self.val_anno = pd.read_csv(os.path.join(self.anno_root, 'ct_val.txt'), \
                                        sep=" ", \
                                        header=None, \
                                        names=['ImageName', 'label'])

        if 'test' in get_list:
            self.test_anno = pd.read_csv(os.path.join(self.anno_root, 'ct_test.txt'), \
                                         sep=" ", \
                                         header=None, \
                                         names=['ImageName', 'label'])

        self.swap_num = args.swap_num

        self.save_dir = './net_model'
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.backbone = args.backbone

        self.use_dcl = True
        self.use_backbone = False if self.use_dcl else True
        self.use_Asoftmax = False
        self.use_focal_loss = False
        self.use_fpn = False
        self.use_hier = False

        self.weighted_sample = False
        self.cls_2 = True
        self.cls_2xmul = False

        self.log_folder = './logs'
        if not os.path.exists(self.log_folder):
            os.mkdir(self.log_folder)


if __name__ == '__main__':
    args = parse_args()
    print(args)
    Config = LoadConfig(args, 'train')
    Config.cls_2 = args.cls_2
    Config.cls_2xmul = args.cls_mul
    assert Config.cls_2 ^ Config.cls_2xmul

    transformers = load_data_transformers(args.resize_resolution, args.crop_resolution, args.swap_num)

    # inital dataloader
    train_set = dataset(Config=Config, \
                        anno=Config.train_anno, \
                        common_aug=transformers["common_aug"], \
                        swap=transformers["swap"], \
                        totensor=transformers["train_totensor"], \
                        train=True)

    trainval_set = dataset(Config=Config, \
                           anno=Config.train_anno, \
                           common_aug=transformers["None"], \
                           swap=transformers["None"], \
                           totensor=transformers["val_totensor"], \
                           train=False,
                           train_val=True)

    val_set = dataset(Config=Config, \
                      anno=Config.val_anno, \
                      common_aug=transformers["None"], \
                      swap=transformers["None"], \
                      totensor=transformers["test_totensor"], \
                      test=True)

    dataloader = {}
    dataloader['train'] = torch.utils.data.DataLoader(train_set, \
                                                      batch_size=args.train_batch, \
                                                      shuffle=True, \
                                                      num_workers=args.train_num_workers, \
                                                      collate_fn=collate_fn4train if not Config.use_backbone else collate_fn4backbone,
                                                      drop_last=True if Config.use_backbone else False,
                                                      pin_memory=True)

    setattr(dataloader['train'], 'total_item_len', len(train_set))

    dataloader['trainval'] = torch.utils.data.DataLoader(trainval_set, \
                                                         batch_size=args.val_batch, \
                                                         shuffle=False, \
                                                         num_workers=args.val_num_workers, \
                                                         collate_fn=collate_fn4val if not Config.use_backbone else collate_fn4backbone,
                                                         drop_last=True if Config.use_backbone else False,
                                                         pin_memory=True)

    setattr(dataloader['trainval'], 'total_item_len', len(trainval_set))
    setattr(dataloader['trainval'], 'num_cls', Config.numcls)

    dataloader['val'] = torch.utils.data.DataLoader(val_set, \
                                                    batch_size=args.val_batch, \
                                                    shuffle=False, \
                                                    num_workers=args.val_num_workers, \
                                                    collate_fn=collate_fn4test if not Config.use_backbone else collate_fn4backbone,
                                                    drop_last=True if Config.use_backbone else False,
                                                    pin_memory=True)

    setattr(dataloader['val'], 'total_item_len', len(val_set))
    setattr(dataloader['val'], 'num_cls', Config.numcls)

    cudnn.benchmark = True

    print('Choose model and train set', flush=True)
    model = MainModel(Config)

    # load model
    if (args.resume is None) and (not args.auto_resume):
        print('train from imagenet pretrained models ...', flush=True)
    else:
        if not args.resume is None:
            resume = args.resume
            print('load from pretrained checkpoint %s ...' % resume, flush=True)
        elif args.auto_resume:
            resume = auto_load_resume(Config.save_dir)
            print('load from %s ...' % resume)
        else:
            raise Exception("no checkpoints to load")

        model_dict = model.state_dict()
        pretrained_dict = torch.load(resume)
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    print('Set cache dir')
    time = datetime.datetime.now()
    filename = '%s_%d%d%d_%s' % (args.discribe, time.month, time.day, time.hour, Config.dataset)
    save_dir = os.path.join(Config.save_dir, filename)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.cuda()
    model = nn.DataParallel(model)

    # optimizer prepare
    if Config.use_backbone:
        ignored_params = list(map(id, model.module.classifier.parameters()))
    else:
        ignored_params1 = list(map(id, model.module.classifier.parameters()))
        ignored_params2 = list(map(id, model.module.classifier_swap.parameters()))
        ignored_params3 = list(map(id, model.module.Convmask.parameters()))

        ignored_params = ignored_params1 + ignored_params2 + ignored_params3
    print('the num of new layers:', len(ignored_params))
    base_params = filter(lambda p: id(p) not in ignored_params, model.module.parameters())

    lr_ratio = args.cls_lr_ratio
    base_lr = args.base_lr
    if Config.use_backbone:
        optimizer = optim.SGD([{'params': base_params},
                               {'params': model.module.classifier.parameters(), 'lr': base_lr}], lr=base_lr,
                              momentum=0.9)
    else:
        optimizer = optim.SGD([{'params': base_params},
                               {'params': model.module.classifier.parameters(), 'lr': lr_ratio * base_lr},
                               {'params': model.module.classifier_swap.parameters(), 'lr': lr_ratio * base_lr},
                               {'params': model.module.Convmask.parameters(), 'lr': lr_ratio * base_lr},
                               ], lr=base_lr, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=0.1)

    # train entry

    epoch_num = args.epoch
    start_epoch = args.start_epoch
    optimizer = optimizer
    exp_lr_scheduler = exp_lr_scheduler
    data_loader = dataloader
    save_dir = save_dir
    data_size = args.crop_resolution
    savepoint = args.save_point
    checkpoint = args.check_point

    step = 0
    eval_train_flag = False
    rec_loss = []
    checkpoint_list = []

    train_batch_size = data_loader['train'].batch_size
    train_epoch_step = data_loader['train'].__len__()
    train_loss_recorder = LossRecord(train_batch_size)

    if savepoint > train_epoch_step:
        savepoint = 1 * train_epoch_step
        checkpoint = savepoint

    date_suffix = dt()
    log_file = open(os.path.join(Config.log_folder, 'formal_log_r50_dcl_%s_%s.log' % (str(data_size), date_suffix)),
                    'a')

    add_loss = nn.L1Loss()
    get_ce_loss = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, epoch_num - 1):
        exp_lr_scheduler.step(epoch)
        model.train(True)

        save_grad = []
        for batch_cnt, data in enumerate(data_loader['train']):
            step += 1
            loss = 0
            model.train(True)

            if Config.use_dcl:
                inputs, labels, labels_swap, swap_law, img_names = data
                print(data)
                inputs = inputs.cuda()
                labels = torch.from_numpy(np.array(labels)).cuda()
                labels_swap = torch.from_numpy(np.array(labels_swap)).cuda()
                swap_law = torch.from_numpy(np.array(swap_law)).float().cuda()

            optimizer.zero_grad()

            if inputs.size(0) < 2 * train_batch_size:
                outputs = model(inputs, inputs[0:-1:2])

            else:
                outputs = model(inputs, None)
                # print(outputs.shape)

            ce_loss = get_ce_loss(outputs[0], labels)

            loss += ce_loss

            alpha_ = 1
            beta_ = 1
            gamma_ = 0.01
            if Config.use_dcl:
                swap_loss = get_ce_loss(outputs[1], labels_swap) * beta_
                loss += swap_loss
                law_loss = add_loss(outputs[2], swap_law) * gamma_
                loss += law_loss

            loss.backward()
            torch.cuda.synchronize()

            optimizer.step()
            torch.cuda.synchronize()

            if Config.use_dcl:
                print(
                    'step: {:-8d} / {:d} loss=ce_loss+swap_loss+law_loss: {:6.4f} = {:6.4f} + {:6.4f} + {:6.4f} '.format(
                        step, train_epoch_step, loss.detach().item(), ce_loss.detach().item(),
                        swap_loss.detach().item(), law_loss.detach().item()), flush=True)
            if Config.use_backbone:
                print('step: {:-8d} / {:d} loss=ce_loss+swap_loss+law_loss: {:6.4f} = {:6.4f} '.format(step,
                                                                                                       train_epoch_step,
                                                                                                       loss.detach().item(),
                                                                                                       ce_loss.detach().item()),
                      flush=True)
            rec_loss.append(loss.detach().item())

            train_loss_recorder.update(loss.detach().item())

            # evaluation & save
            if step % checkpoint == 0:
                rec_loss = []
                print(32 * '-', flush=True)
                print('step: {:d} / {:d} global_step: {:8.2f} train_epoch: {:04d} rec_train_loss: {:6.4f}'.format(step,
                                                                                                                  train_epoch_step,
                                                                                                                  1.0 * step / train_epoch_step,
                                                                                                                  epoch,
                                                                                                                  train_loss_recorder.get_val()),
                      flush=True)
                print('current lr:%s' % exp_lr_scheduler.get_lr(), flush=True)
                if eval_train_flag:
                    trainval_acc1, trainval_acc2, trainval_acc3 = eval_turn(Config, model, data_loader['trainval'],
                                                                            'trainval', epoch, log_file)
                    if abs(trainval_acc1 - trainval_acc3) < 0.01:
                        eval_train_flag = False

                val_acc1, val_acc2, val_acc3 = eval_turn(Config, model, data_loader['val'], 'val', epoch, log_file)

                save_path = os.path.join(save_dir,
                                         'weights_%d_%d_%.4f_%.4f.pth' % (epoch, batch_cnt, val_acc1, val_acc3))
                torch.cuda.synchronize()
                # torch.save(model.state_dict(), save_path)
                print('saved model to %s' % (save_path), flush=True)
                torch.cuda.empty_cache()

            # save only
            elif step % savepoint == 0:
                train_loss_recorder.update(rec_loss)
                rec_loss = []
                save_path = os.path.join(save_dir, 'savepoint_weights-%d-%s.pth' % (step, dt()))

                checkpoint_list.append(save_path)
                if len(checkpoint_list) == 6:
                    os.remove(checkpoint_list[0])
                    del checkpoint_list[0]
                # torch.save(model.state_dict(), save_path)
                torch.cuda.empty_cache()

    log_file.close()
