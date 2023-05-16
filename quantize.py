import torch
import torch.nn as nn
import torch.nn.functional as F
from decimal import *
import numpy as np
import cupy as cp
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
def uniform_quantize(k):
    class u_quantize(torch.autograd.Function): # torch.autograd.Functionの継承 torch.autogradは自動微分関数

        @staticmethod #インスタンスを作らなくても関数を使える(引数にselfいらない)
        def forward(ctx, input): #inputはuniform_qやuniform_qeの引数
            if k == 32:
                out = input
            elif k == 1:
                out = torch.sign(input)
            else:
                #---2^k-1か2^k---
                # n = float(2 ** k - 1)
                n = float(2 ** k)
                print(input)
                #---3パターンのoutのうちどれか1つ
                # out = torch.round(input * n) / n
                # out = torch.floor(input * n) / n
                out1 = torch.round(input * n) / n
                out2 = torch.floor(input * n) / n
                out = torch.where(out1<=0.875,out1,out2) 
                # print(out)
            return out

        @staticmethod
        def backward(ctx, grad_output): # 引数はctxとforward関数の出力の勾配
            grad_input = grad_output.clone() 
            return grad_input # 返り値はforwad関数の引数の勾配
    return u_quantize().apply

#重みの量子化
class weight_quantize_fn(nn.Module): #weight_quantize_fnclassがnn.Moduleというclassを継承している
    def __init__(self, w_bit):
        super(weight_quantize_fn, self).__init__()
        self.w_bit = w_bit
        self.uniform_q = uniform_quantize(k = w_bit)
        self.uniform_qe = uniform_quantize(k = 1)

    def forward(self, x):
        if self.w_bit == 32:
            weight_q = x
        elif self.w_bit == 1:
            E = torch.mean(torch.abs(x)).detach() #入力xの絶対値(tensor)の平均を出す .detach() は同一デバイス上に新しいテンソルを作成する。計算グラフからは切り離され、requires_grad=Falseになる
            E = self.uniform_qe(E)
            weight_q = self.uniform_q(x / E) * E
        else:
            weight = torch.tanh(x)
            max_w = torch.max(torch.abs(weight)).detach()
            weight = weight / 2 / max_w + 0.5
            weight_q = max_w * (2 * self.uniform_q(weight) - 1) #max_wいる？？
            # weight_q = 2 * self.uniform_q(weight) - 1 #max_wいる？？
        return weight_q

#出力の量子化
class activation_quantize_fn(nn.Module): #activation_quantize_fnclassがnn.Moduleというclassを継承している
    def __init__(self, a_bit):
        super(activation_quantize_fn, self).__init__()
        self.a_bit = a_bit
        self.uniform_q = uniform_quantize(k = a_bit)

    def forward(self, x):
        if self.a_bit == 32:
            activation_q = x
        else:
            activation_q = self.uniform_q(x)
        return activation_q

#量子化畳み込み
def conv2d_Q_fn(w_bit): 
    class Conv2d_Q(nn.Conv2d): #Conv2d_Qclassがnn.Conv2dというclassを継承している
        def __init__(self, in_channels, out_channels, kernel_size, stride = 1, #convolution(畳み込み)の定義で,第1引数はその入力のチャネル数,第2引数は畳み込み後のチャネル数,第3引数は畳み込みをするための正方形フィルタ(カーネル)の1辺のサイズ,第4引数はカーネルをずらす間隔
                 padding = 0, dilation = 1, groups = 1, bias = False):
            super(Conv2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias)
            self.w_bit = w_bit
            self.quantize_fn = weight_quantize_fn(w_bit=w_bit) #重みの量子化

        def forward(self, input, order=None):
            weight_q = self.quantize_fn(self.weight)
            return F.conv2d(input, weight_q, self.bias, self.stride,
                      self.padding, self.dilation, self.groups)
    return Conv2d_Q

#量子化全結合層
def linear_Q_fn(w_bit): 
    class Linear_Q(nn.Linear):
        def __init__(self, in_features, out_features, bias = False):
            super(Linear_Q, self).__init__(in_features, out_features, bias)
            self.w_bit = w_bit
            self.quantize_fn = weight_quantize_fn(w_bit = w_bit) #重みの量子化

        def forward(self, input):
            weight_q = self.quantize_fn(self.weight)
            return F.linear(input, weight_q, self.bias)
    return Linear_Q