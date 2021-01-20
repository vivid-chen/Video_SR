import decimal
import torch
from tqdm import tqdm
from utils import utils
from trainer.trainer import Trainer
import torch.optim as optim


class TRAINER_DBVSR(Trainer):
    def __init__(self, args, loader, my_model, my_loss, ckp):
        super(TRAINER_DBVSR, self).__init__(args, loader, my_model, my_loss, ckp)
        print("Using TRAINER_DBVSR")
        self.optimizer = self.make_optimizer()
    
    # 优化器
    def make_optimizer(self):
        kwargs = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
        a = optim.Adam([{"params": self.model.model.module.pwcnet.parameters(), "lr": self.args.pwc_lr},
                        {"params": self.model.model.module.head.parameters(), "lr": self.args.lr},
                        {"params": self.model.model.module.body.parameters(), "lr": self.args.lr},
                        {"params": self.model.model.module.tail.parameters(), "lr": self.args.lr}], **kwargs)

        return a

    # clip gradient可以有效控制权重在一定范围内，防止梯度爆炸
    def clip_gradient(self, optimizer, grad_clip):
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.clamp_(-grad_clip, grad_clip)

    def train(self):
        print("Now training")

        
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_last_lr() # 这个lr指的是learning rate
        self.ckp.write_log(
            'Epoch {:3d} with \tpwcLr {:.2e}\trcanLr {:.2e}\t'.format(epoch, decimal.Decimal(lr[0]),
                                                                      decimal.Decimal(lr[1])))
        self.loss.start_log()
        self.model.train()
        self.ckp.start_log()

        for batch, (input, gt, _, kernel, input__bic) in enumerate(self.loader_train):
            # 检查通道
            if self.args.n_colors == 1 and input.size()[-3] == 3:
                raise Exception("Now just Support RGB mode, not Support Ycbcr mode! "
                                "See args.n_colors={} and input_channel={}"
                                .format(self.args.n_colors, input.size()[-3]))

            # 指定变量存储设备
            input = input.to(self.device)
            input__bic = input__bic.to(self.device)
            kernel = kernel.to(self.device)
            gt = gt[:, self.args.n_sequences // 2, :, :, :].to(self.device)
            sr_output = self.model(input, input__bic, kernel) # 模糊核之后的中间SR输出结果
            self.optimizer.zero_grad() # 反向传播前手动将梯度置零
            loss = self.loss(sr_output, gt) # 得到中间输出结果和GT的loss
            loss.backward() # 反向传播

            # 梯度截断，将梯度约束在某一个区间内，在优化器更新之前进行梯度截断操作
            self.clip_gradient(self.optimizer, self.args.grad_clip) 
            self.optimizer.step()
            self.ckp.report_log(loss.item())

            # 每隔print_every次batch记录一次日志
            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\tLoss : [loss: {:.6f}]'.format(
                    (batch + 1) * self.args.train_batch_size,
                    len(self.loader_train.dataset),
                    self.ckp.loss_log[-1] / (batch + 1)
                ))

        self.scheduler.step() # 更新学习率，按照epoch为单位进行更换
        self.loss.step() # 

        self.loss.end_log(len(self.loader_train))

    def test(self):
        # epoch = self.scheduler.last_epoch + 1 # 感觉这里不用+1
        epoch = self.scheduler.last_epoch
        self.ckp.write_log('\nEvaluation:')
        self.model.eval()
        self.ckp.start_log(train=False)
        with torch.no_grad(): # 上下文管理器，被该语句包围起来的部分将不会track 梯度
            tqdm_test = tqdm(self.loader_test, ncols=80)
            for idx_img, (input, gt, filename, kernel, input_bic) in enumerate(tqdm_test):

                filename = filename[self.args.n_sequences // 2][0]

                if self.args.n_colors == 1 and input.size()[-3] == 3:
                    raise Exception("Now just Support RGB mode, not Support Ycbcr mode! "
                                    "See args.n_colors={} and input_channel={}"
                                    .format(self.args.n_colors, input.size()[-3]))

                input = input.to(self.device)
                kernel = kernel.to(self.device)
                gt = gt[:, self.args.n_sequences // 2, :, :, :].to(self.device)
                input_bic = input_bic.to(self.device)

                sr_output = self.model(input, input_bic, kernel) # 通过模糊核生成中间SR结果
                PSNR = utils.calc_psnr(gt, sr_output, rgb_range=self.args.rgb_range, is_rgb=True)

                self.ckp.report_log(PSNR, train=False)

                if self.args.save_images:
                    sr_output = utils.postprocess(sr_output, rgb_range=self.args.rgb_range, ycbcr_flag=False,
                                                  device=self.device)[0]

                    save_list = [sr_output]

                    self.ckp.save_images(filename, save_list, self.args.testset)

            self.ckp.end_log(len(self.loader_test), train=False)
            best = self.ckp.psnr_log.max(0)
            self.ckp.write_log(
                '[{}]\taverage stage_PSNR: {:.3f}(Best: {:.3f} @epoch {})'.format(
                    self.args.data_test,
                    self.ckp.psnr_log[-1],
                    best[0], best[1] + 1))

            if not self.args.test_only:
                self.ckp.save(self, epoch, is_best=(best[1] + 1 == epoch))
