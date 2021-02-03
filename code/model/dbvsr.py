from model.flow_pwc import Flow_PWC
from model.fcnnet import FCNNET
import torch
from model.deconv_layer import Deconv_Layer
from tool.kernel_shift import kernel_shift
from model.modules import *


def make_model(args, parent=False):
    return DBVSR(args)


class DBVSR(nn.Module):
    def __init__(self, args):
        super(DBVSR, self).__init__()

        n_colors = args.n_colors
        n_sequences = args.n_sequences # 使用帧数量
        n_resgroups = args.n_resgroups # 残差group数量
        n_resblocks = args.n_resblocks # 残差块数量
        n_feats = args.n_feats
        reduction = args.reduction
        res_scale = args.res_scale

        self.args = args
        self.scale = args.scale

        self.pwcnet = Flow_PWC(load_pretrain=True, pretrain_fn=args.pwc_pretrain, device='cuda') # 载入预训练模型
        # self.pwcnet = Flow_PWC(load_pretrain=False, pretrain_fn=args.pwc_pretrain, device='cuda')

        self.deconv_layer = Deconv_Layer()
        # self.deconv_layer = nn.DataParallel(self.deconv_layer) # 多GPU

        # define head module 定义头
        modules_head = [
            nn.Conv2d(n_sequences * (n_colors * (self.scale ** 2 + 1)), n_feats, kernel_size=3, stride=1, padding=1)]

        # define body module
        modules_body = [
            ResidualGroup(
                n_feats, kernel_size=3, reduction=reduction, act=nn.ReLU(False), res_scale=res_scale,
                n_resblocks=n_resblocks)
            for _ in range(n_resgroups)]
        modules_body.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1))

        # define tail module
        modules_tail = [Upsampler(self.scale, n_feats),
                        nn.Conv2d(n_feats, n_colors, kernel_size=3, stride=1, padding=1)]

        self.head = nn.Sequential(*modules_head)
        self.head = nn.DataParallel(self.head) # mul_GPU

        self.body = nn.Sequential(*modules_body)
        self.body = nn.DataParallel(self.body) # mul_GPU

        self.tail = nn.Sequential(*modules_tail)
        self.tail = nn.DataParallel(self.tail) # mul_GPU

        self.fcnnet = FCNNET(args)
        self.fcnnet.load_state_dict(torch.load(args.fc_pretrain), strict=False)

    def spatial2depth(self, spatial, scale):
        depth_list = []
        for i in range(scale):
            for j in range(scale):
                depth_list.append(spatial[:, :, i::scale, j::scale])
        depth = torch.cat(depth_list, dim=1)
        return depth

    def forward(self, x, x_bicubic, kernel):
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache() # 释放无关内存
            
        kernel = kernel[0]
        if not x.ndimension() == 5:
            raise Exception("x.ndimension must equal 5: see x.ndimension={}".format(x.ndimension()))

        b, n, c, h, w = x.size()
        kernel_size, _ = kernel.size()
        # 下面两句太浪费显存了
        # frame_list = [x[:, i, :, :, :] for i in range(n)]
        # bicubic_list = [x_bicubic[:, i, :, :, :] for i in range(n)]
        # 改成了这样
        frame_list = []
        bicubic_list = []
        for i in range(n):
            frame_list.append(x[:, 0, :, :, :])
            x = x[:, 1:, :, :, :]
            bicubic_list.append(x_bicubic[:, 0, :, :, :])
            x_bicubic = x_bicubic[:, 1:, :, :, :]

        # x_mid_bicubic = bicubic_list[n // 2]
        del x # 节省显存
        del x_bicubic # 节省显存

        # fcnnet kernel prediction net
        kernel_input = kernel.reshape(1, kernel_size * kernel_size)
        del kernel
        kernel_output = self.fcnnet(kernel_input)
        del kernel_input
        kernel_output_shift = kernel_output.view(kernel_size, kernel_size)
        del kernel_output
        kernel_output_shift = kernel_shift(kernel_output_shift.detach().cpu().numpy()) # 将张量图中的变量弄到numpy，然后去转换函数转换
        kernel_output_shift = torch.FloatTensor(kernel_output_shift).cuda() # 将numpy重新变成张量
        kernel_output_shift = kernel_output_shift.view(1, 1, kernel_size, kernel_size)
        del kernel_size

        # CZY######################################################################
        
        # mid_deconv = self.deconv_layer(bicubic_list[n // 2], kernel_output_shift.detach())
        # del kernel_output_shift

        # pwc for flow and warp
        # CZY######################################################################
        # warp_list = [] # 这里为了节省显存，继续用frame_list，存储数据
        # flow_list = [] # 感觉不需要
        # bic_warp_list = [] # 这里为了节省显存，继续用bicubic_list，存储数据
        # bic_flow_list = [] # 感觉不需要
        for i in range(n):
            if i == n//2:
                continue
            frame_list[i], _ = self.pwcnet(frame_list[n//2], frame_list[i])
            # flow_list.append(flow_tmp)

            bicubic_list[i], _ = self.pwcnet(bicubic_list[n//2], bicubic_list[i])
            # bic_flow_list.append(bic_flow_tmp)
        # CZY######################################################################

        # warp0_1, flow0_1 = self.pwcnet(frame_list[1], frame_list[0])
        # warp2_1, flow2_1 = self.pwcnet(frame_list[1], frame_list[2])

        # bic_warp0_1, bic_flow0_1 = self.pwcnet(bicubic_list[1], bicubic_list[0])
        # bic_warp2_1, bic_flow2_1 = self.pwcnet(bicubic_list[1], bicubic_list[2])

        # deconv
        # CZY######################################################################
        # 将全部bic都用kernel处理 在pwc后
        for i in range(n):
            bicubic_list[i] = self.deconv_layer(bicubic_list[i], kernel_output_shift.detach())
            bicubic_list[i] = self.spatial2depth(bicubic_list[i], scale=self.scale)

        # mid_deconv = self.deconv_layer(bicubic_list[n // 2], kernel_output_shift.detach())
        # del kernel_output_shift

        # CZY######################################################################
        # mid_frame = frame_list[n//2]
        # mid_deconv = bicubic_list[n//2]
        # del frame_list[n//2] # 删除节省显存，并将序号和flow_list对齐
        
        # CZY######################################################################
        # 和上面的循环合并
        # for i in range(n):
        #     bicubic_list[i] = self.spatial2depth(bicubic_list[i], scale=self.scale)

        # bic_warp0_1_depth = self.spatial2depth(bic_warp0_1, scale=self.scale)
        # bic_warp2_1_depth = self.spatial2depth(bic_warp2_1, scale=self.scale)
        # mid_deconv_depth = self.spatial2depth(bicubic_list[n//2], scale=self.scale)
        # del bicubic_list[n//2] # 删除节省显存，并将序号和flow_list对齐

        # sr_input = torch.cat((bic_warp0_1_depth, warp0_1, mid_deconv_depth, frame_list[1], bic_warp2_1_depth, warp2_1),
        #                      dim=1)
        if n == 3:
            sr_input = torch.cat((  bicubic_list[0], frame_list[0], 
                                    bicubic_list[1], frame_list[1], 
                                    bicubic_list[2], frame_list[2]),
                                    dim=1)
        elif n == 5:
            sr_input = torch.cat((  bicubic_list[0], frame_list[0], 
                                    bicubic_list[1], frame_list[1], 
                                    bicubic_list[2], frame_list[2], 
                                    bicubic_list[3], frame_list[3], 
                                    bicubic_list[4], frame_list[4]),
                                    dim=1)
        elif n == 7:
            sr_input = torch.cat((  bicubic_list[0], frame_list[0], 
                                    bicubic_list[1], frame_list[1], 
                                    bicubic_list[2], frame_list[2],
                                    bicubic_list[3], frame_list[3], 
                                    bicubic_list[4], frame_list[4], 
                                    bicubic_list[5], frame_list[5], 
                                    bicubic_list[6], frame_list[6]),
                                    dim=1)

        # 节省显存
        # del mid_deconv_depth
        # del mid_frame
        del bicubic_list, frame_list


        # RCAN super-resolution with upsample
        head_out = self.head(sr_input)
        del sr_input # 节省显存
        body_out = self.body(head_out)
        sr_output = self.tail(body_out + head_out)
        del head_out
        del body_out

        return sr_output

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
