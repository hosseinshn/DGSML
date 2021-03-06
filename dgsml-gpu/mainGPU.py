import argparse
import os
import sys
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import torchvision.models as models
from collections import OrderedDict
import math
from scipy.stats import entropy
from itertools import combinations
import torch.optim as optim
import itertools
from collections import OrderedDict
sys.setrecursionlimit(1000000)
import warnings
from pycocotools.coco import COCO
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR

from utils import *
from DataloaderGPU import *
from DGMSL_fncGPU import *
from Alexnetv2 import alexnet, Classifier_homo
from Alexnetv2 import zero_grad, update, cloned_state_dict

warnings.filterwarnings("ignore")

def main():
    train_arg_parser = argparse.ArgumentParser()
    train_arg_parser.add_argument("--dataset", type=str, default='PACS', help='input dataset to train the model') 
    train_arg_parser.add_argument("--target_domain", type=int, default=0, help="index of the target domain")
    train_arg_parser.add_argument("--num_train_domain", type=int, default=3, help="number of source domains")
    train_arg_parser.add_argument("--data_root", type=str, default='', help="path to images")        
    train_arg_parser.add_argument("--filelist", type=str, default='', help="path to the dataset filelist")
    train_arg_parser.add_argument("--batch_size", type=int, default=32, help="batch size for training, default is 128")
    train_arg_parser.add_argument("--num_classes", type=int, default=7, help="number of classes")
    train_arg_parser.add_argument("--iteration", type=int, default=30, help="iteration of training domains")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3, help='learning rate of the model')
    train_arg_parser.add_argument("--meta_lr", type=float, default=1e-4, help='meta learning rate of the model')
    train_arg_parser.add_argument("--unlabeled_rate", type=float, default=0.5, help='the rate of unlabeled samples in each domain')
    train_arg_parser.add_argument("--save_logs", type=str, default='./DGMSL/logs/PACS/', help='path of folder to write log')
    train_arg_parser.add_argument("--save_models", type=str, default='./DGMSL/models/PACS/', help='folder for saving model')
    train_arg_parser.add_argument("--save_results", type=str, default='./DGMSL/results/PACS/', help='folder for saving model')
    train_arg_parser.add_argument("--SSL_coef", type=float, default=1e-4, help='Semi-superivsed regularization coefficient')
    train_arg_parser.add_argument("--gloabl_coef", type=float, default=1e-2, help='global alignment regularization coefficient')        
    train_arg_parser.add_argument("--class_wise_coef", type=float, default=1e-2, help='class-wise alignment regularization coefficient')
    train_arg_parser.add_argument("--momentum", type=float, default=0.9, help='momentum for the optimizer')
    train_arg_parser.add_argument("--weight_decay", type=float, default=0.0, help='weight decay regularization coefficient')        
    train_arg_parser.add_argument("--seed", type=int, default=42, help='set the random seed')
    train_arg_parser.add_argument('--save_step', type=int, default=10, help='when to do the validation step and save the model') 
    train_arg_parser.add_argument('--device_num', type=int, default=0, help='the GPU number for multiple GPUs')
        
        
    args = train_arg_parser.parse_args()
   
    paths = [args.save_logs, args.save_models, args.save_results]
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)
        with open(path+'/args.txt', 'w') as f:
            f.write('{}\n'.format(str(args)))    
    
    np.random.seed(args.seed)
    print(torch.cuda.is_available())

    torch.manual_seed(args.seed)
    
    domains_name = get_domain_name(args)
    data_folder, train_data, val_data, test_data = get_data_folder(args)
    unseen_index = args.target_domain
    
    train_paths, val_paths, unseen_data_path = get_path(args, data_folder, train_data, val_data, test_data, unseen_index)

    batImageGenTrains, batImageGenTrains_metatest, batImageGenVals, batImageGenTests = get_dataloader(
        args, train_paths, val_paths, unseen_data_path)

    FT = alexnet(pretrained=True)    

    CLS = Classifier_homo(args.num_classes)

    FT.cuda()
    CLS.cuda() 
    optimizer1 = optim.SGD(FT.parameters(), lr=args.lr, momentum=args.momentum, weight_decay = args.weight_decay)
    optimizer2 = optim.SGD(CLS.parameters(), lr=args.lr, momentum=args.momentum, weight_decay = args.weight_decay)
    

    ce_loss = torch.nn.CrossEntropyLoss()
    
    best_acc = 0.0
    
    for ite in range(args.iteration):
        index = np.random.permutation(args.num_train_domain)
        meta_train_idx = index[0:2]
        meta_test_idx = index[2:]
        
        iter_loss = train_MAML(args, FT, CLS, meta_train_idx, meta_test_idx, ce_loss, optimizer1, optimizer2, batImageGenTrains, batImageGenTrains_metatest)
        
        f = open(os.path.join(args.save_logs, 'args.txt'), mode='a')
        f.write('iteration:{}, train total loss:{}\n'.format(ite, iter_loss))
        f.close()        
        
        if ite % args.save_step == 0:
            val_accuracy = validate_workflow(args, FT, CLS, batImageGenVals)
            if val_accuracy > best_acc: 
                best_acc = val_accuracy
                f = open(os.path.join(args.save_results, 'Best_val.txt'), mode='a')
                f.write('iteration:{}, best validation accuracy:{}\n'.format(ite, best_acc))
                f.close()
                torch.save(FT, os.path.join(args.save_models, 'Best_FT.pt'))
                torch.save(CLS, os.path.join(args.save_models, 'Best_CLS.pt'))
            

    Best_FT = torch.load(os.path.join(args.save_models, 'Best_FT.pt'))
    Best_CLS = torch.load(os.path.join(args.save_models, 'Best_CLS.pt'))
    Best_FT.eval()
    Best_CLS.eval()
    
    test_acc = heldout_test(args, Best_FT, Best_CLS, batImageGenTests)   
    
    f = open(os.path.join(args.save_results, 'Target.txt'), mode='a')
    f.write('Target accuracy:{}\n'.format(test_acc))
    f.close()

        
if __name__ == "__main__":
    main()
