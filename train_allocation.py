import config
import framework
import argparse
import model
import os
import torch
import numpy as np
import random

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='MaterialSet_allocation', help='name of the model')
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--multi_gpu', type=bool, default=False)
parser.add_argument('--dataset', type=str, default='MaterialSet_allocation')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--max_epoch', type=int, default=50)
parser.add_argument('--test_epoch', type=int, default=1)
parser.add_argument('--train_prefix', type=str, default='train_4') # 要改
parser.add_argument('--dev_prefix', type=str, default='dev_4') # 要改
parser.add_argument('--test_prefix', type=str, default='test_sets')
parser.add_argument('--max_len', type=int, default=150)
parser.add_argument('--rel_num', type=int, default=44)
parser.add_argument('--period', type=int, default=50)
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--datatype', type=int, default=4) # 要改
parser.add_argument('--traintype', type=str, default="wo_att", help="ablation test") # 要改
parser.add_argument('--gpu', type=str, default="0")
# 在命令行里面只需要改 --traintype 和 --gpu
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

con = config.Config(args)

fw = framework.Framework(con)

model_used = model.MaterialSet_allocation

fw.train_allocation(model_used)
