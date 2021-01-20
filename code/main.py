import os

import torch

import data
import model
import loss
import option
from trainer.trainer_dbvsr import TRAINER_DBVSR
from trainer.trainer_baseline_lr import TRAINER_BASELINE_LR
from trainer.trainer_baseline_hr import TRAINER_BASELINE_HR
from logger import logger

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

args = option.args # 载入初始化参数
torch.manual_seed(args.seed) # 为GPU设置随机数种子
chkp = logger.Logger(args) # 日志
# CZY 加入后方便单步调试，相当于直接指定参数
args.test_only = False
args.resume = True # 断点重续选True
########################################

if args.model == 'DBVSR':
    print("Selected model: {}".format(args.model))
    model = model.Model(args, chkp) # 初始化模型
    loss = loss.Loss(args, chkp) if not args.test_only else None # 初始化损失
    loader = data.Data(args) # 初始化参数加载
    t = TRAINER_DBVSR(args, loader, model, loss, chkp) # 初始化训练器
    while not t.terminate(): # 还没到最后epoch的时候执行下列语句
        t.train()
        t.test()

elif args.model == 'baseline_lr':
    print("Selected model: {}".format(args.model))
    model = model.Model(args, chkp)
    loss = loss.Loss(args, chkp) if not args.test_only else None
    loader = data.Data(args)
    t = TRAINER_BASELINE_LR(args, loader, model, loss, chkp)
    while not t.terminate():
        t.train()
        t.test()

elif args.model == 'baseline_hr':
    print("Selected model: {}".format(args.model))
    model = model.Model(args, chkp)
    loss = loss.Loss(args, chkp) if not args.test_only else None
    loader = data.Data(args)
    t = TRAINER_BASELINE_HR(args, loader, model, loss, chkp)
    while not t.terminate():
        t.train()
        t.test()

else:
    raise NotImplementedError('Model [{:s}] is not found'.format(args.model))

chkp.done()
