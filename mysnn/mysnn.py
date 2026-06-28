import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
from attention import SE, TA, SpatialAttention
AMP=0.3

class PseudoSpikeRect(torch.autograd.Function):
    """Define custom autograd function for Spike Function """

    @staticmethod
    def forward(ctx, input, vth, grad_win):
        ctx.save_for_backward(input)
        ctx.vth = vth
        ctx.grad_win = grad_win
        return input.gt(vth).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        vth = ctx.vth
        grad_win = ctx.grad_win
        grad_input = grad_output.clone()
        spike_pseudo_grad = abs(input - vth) < grad_win
        return AMP * grad_input * spike_pseudo_grad.float(), None, None


class FeedForwardCUBALIFCell(nn.Module):
    def __init__(self, psp_func, pseudo_grad_ops, param):
        """
        :param psp_func: pre-synaptic function 卷积
        :param pseudo_grad_ops: pseudo gradient operation 神经元
        :param param: (current decay, voltage decay, voltage threshold, gradient window parameter) 神经元初始化参数
        """
        # LIF神经元模型
        super(FeedForwardCUBALIFCell, self).__init__()
        self.psp_func = psp_func
        self.pseudo_grad_ops = pseudo_grad_ops
        self.cdecay, self.vdecay, self.vth, self.grad_win = param
        self.reactivation = True
        self.conv1 = nn.Conv1d(512, 512, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, input_data, state):
        """
        :param input_data: input spike from pre-synaptic neurons
        :param state: (output spike of last timestep, current of last timestep, voltage of last timestep)
        :return: output spike, (output spike, current, voltage)
        """

        pre_spike, pre_current, pre_volt = state

        current = self.cdecay * pre_current + self.psp_func(input_data)

        volt = self.vdecay * pre_volt * (1. - pre_spike) + current

        output = self.pseudo_grad_ops(volt, self.vth, self.grad_win)
        #协同学习
        #

        return output, (output, current, volt)


VDECAY = 0.1
CDECAY = 0.1
VTH = 0.1
GRAD_WIN = 0.3
TH_AMP = 0.01
TH_DECAY = 0.1
BASE_TH = 0.01
PARAM_LIST = [CDECAY, VDECAY, VTH, GRAD_WIN, TH_AMP, TH_DECAY, BASE_TH]


class snn(nn.Module):
    def __init__(self, device, params = PARAM_LIST, USE_SE=False, USE_SA=False, USE_TA=False):
        """
        :param device : 运行设备 gpu or cpu
        :param params :[cdecay,vdecay,vth,grad_win,th_amp,th_decay,base_th]
        残差网络模版： resnet_block_1d1(256, 512, 1,param=[2,1,0])
        """
        super(snn, self).__init__()
        self.USE_SE = USE_SE
        self.USE_SA = USE_SA
        self.USE_TA = USE_TA
        self.device = device
        pseudo_grad_ops = PseudoSpikeRect.apply
        if self.USE_SE:
            self.SE = SE(512)
        if self.USE_TA:
            self.TA = TA(28)
        if self.USE_SA:
            self.SA = SpatialAttention()
        self.cdecay, self.vdecay, self.vth, self.grad_win, self.th_amp, self.th_decay, self.base_th = params

        self.c1_vdecay = nn.Parameter(torch.zeros(1, 512, device=self.device) * self.vdecay)
        self.c2_vdecay = nn.Parameter(torch.zeros(1, 512, device=self.device) * self.vdecay)
        self.c3_vdecay = nn.Parameter(torch.zeros(1, 512, device=self.device) * self.vdecay)

        self.c1_cdecay = nn.Parameter(torch.ones(1, 512, device=self.device) * self.cdecay)
        self.c2_cdecay = nn.Parameter(torch.ones(1, 512, device=self.device) * self.cdecay)
        self.c3_cdecay = nn.Parameter(torch.ones(1, 512, device=self.device) * self.cdecay)

        self.c1_kdecay = nn.Parameter(torch.ones(1, 512, device=self.device) * self.cdecay)
        self.c2_kdecay = nn.Parameter(torch.ones(1, 512, device=self.device) * self.cdecay)
        self.c3_kdecay = nn.Parameter(torch.ones(1, 512, device=self.device) * self.cdecay)

        self.conv1 = FeedForwardCUBALIFCell(nn.Linear(512, 512),
                                            pseudo_grad_ops,
                                            [self.c1_cdecay, self.c1_vdecay, self.vth, self.grad_win], )

        # self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv2 = FeedForwardCUBALIFCell(nn.Linear(512, 512),
                                            pseudo_grad_ops, [self.c2_cdecay, self.c2_vdecay, self.vth, self.grad_win])

        self.conv3 = FeedForwardCUBALIFCell(nn.Linear(512, 512),
                                            pseudo_grad_ops, [self.c3_cdecay, self.c3_vdecay, self.vth, self.grad_win])
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_spike, timewindow=28):
        device = self.device
        batch_size = input_spike.shape[0]
        output = torch.zeros(batch_size, 512, device=device)
        c1_current = torch.zeros(batch_size, 512, device=device)
        c1_volt = torch.zeros(batch_size, 512, device=device)
        c1_spike = torch.zeros(batch_size, 512, device=device)
        c1_state = (c1_spike, c1_current, c1_volt)

        c2_current = torch.zeros(batch_size, 512, device=device)
        c2_volt = torch.zeros(batch_size, 512, device=device)
        c2_spike = torch.zeros(batch_size, 512, device=device)
        c2_state = (c2_spike, c2_current, c2_volt)

        c3_current = torch.zeros(batch_size, 512, device=device)
        c3_volt = torch.zeros(batch_size, 512, device=device)
        c3_spike = torch.zeros(batch_size, 512, device=device)
        c3_state = (c3_spike, c3_current, c3_volt)
        # (batch_size,512,28)
        if self.USE_TA:
            input_spike = self.TA(input_spike.view(batch_size, 512, timewindow))
        input_spike = input_spike.view(batch_size, 512, timewindow)
        for step in range(timewindow):
            input_data = input_spike[:, :, step]
            input_data.view(batch_size, 512)
            if self.USE_SE:
                input_data = self.SE(input_data)
            output_spike, c1_state = self.conv1(input_data, c1_state)
            output_spike, c2_state = self.conv2(output_spike, c2_state)
            output_spike, c3_state = self.conv3(output_spike, c3_state)
            output_spike = output_spike.view(output_spike.shape[0], 512)

            output += output_spike
        #output = self.dropout(output)
        return output
