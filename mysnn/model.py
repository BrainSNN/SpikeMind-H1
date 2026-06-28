import torch.nn as nn
import torch
from attention import SE, TA, SpatialAttention
from mysnn import PseudoSpikeRect
from shared_parameters import *
from torch.autograd import Function
from torch.utils.data import Dataset
import pickle
import os
import glob
import scipy.io as sio
import numpy as np

class MODMAset(Dataset):
    def __init__(self, root,files):
        self.root = root
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        try:
            sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        except:
            sample = pickle.load(open(os.path.join(self.root, self.files[index-1]), "rb"))
        X = sample["X"]
        Y = sample["y"]
        data = torch.FloatTensor(X / 10)
        return data,Y
    

class SEEDEEGDataset(Dataset):
    """
    读取 SEED 数据集 Preprocessed_EEG 下所有 .mat 文件中的所有实验数据
    在 __init__ 中一次性加载全部 EEG 到内存，并只保留前 2000 个时刻

    返回:
        eeg: torch.FloatTensor
        label: torch.LongTensor
    """

    def __init__(self, root_dir="/home/data/data/dataset/seed/SEED_EEG/Preprocessed_EEG", transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # 15次实验固定标签
        self.labels = [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0]

        # 找到所有mat文件
        self.mat_files = sorted(glob.glob(os.path.join(self.root_dir, "*.mat")))
        if len(self.mat_files) == 0:
            raise FileNotFoundError(f"在路径 {self.root_dir} 下没有找到 .mat 文件")

        # 存储所有样本
        self.data = []
        self.data_labels = []

        for mat_file in self.mat_files[:10]:
            mat_data = sio.loadmat(mat_file)

            for i in range(15):
                key = self._find_experiment_key(mat_data, i + 1)
                if key is None:
                    print(f"Warning: 文件 {os.path.basename(mat_file)} 中未找到第{i+1}次实验对应的key")
                    continue

                eeg = mat_data[key]  # numpy array
                eeg = np.array(eeg, dtype=np.float32)

                # 只取前2000个时刻数据
                # 默认数据形状一般为 (channels, time)
                eeg = eeg[:, :2000]

                self.data.append(eeg)
                self.data_labels.append(self.labels[i])

        if len(self.data) == 0:
            raise RuntimeError("没有成功加载任何样本，请检查.mat文件内容")

    def _find_experiment_key(self, mat_data, exp_idx):
        """
        查找第 exp_idx 次实验的key
        例如 exp_idx=1 时，匹配 *_eeg1
        """
        target_suffix = f"_eeg{exp_idx}"
        for key in mat_data.keys():
            if key.startswith("__"):
                continue
            if key.endswith(target_suffix):
                return key
        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        eeg = self.data[idx]
        label = self.data_labels[idx]

        if self.transform is not None:
            eeg = self.transform(eeg)
        else:
            eeg = torch.from_numpy(eeg)

        label = torch.tensor(label, dtype=torch.long)

        return eeg, label


class EEGNet(nn.Module):
    def __init__(self, nb_classes, Channels=128, Samples=512, dropout_rate=0.5, kernelLength=64, F1=8, D=2, F2=16):
        super(EEGNet, self).__init__()

        # 1st convolution block
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, kernelLength), padding='same')
        self.batch_norm1 = nn.BatchNorm2d(F1)
        self.depthwise_conv = nn.Conv2d(in_channels=F1, out_channels=F1 * D, kernel_size=(Channels, 1), groups=F1,
                                        padding='same')
        self.batch_norm2 = nn.BatchNorm2d(F1 * D)

        # Activation and pooling
        self.activation = nn.ELU()
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)

        # 2nd convolution block (separable convolution)
        self.separable_conv = nn.Conv2d(in_channels=F1 * D, out_channels=F2, kernel_size=(1, 16), padding='same')
        self.batch_norm3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)

        # Fully connected layers
        self.flatten = nn.Flatten()
        # Calculate the correct size after the convolution and pooling layers
        self.fc = nn.Linear(Channels * F2 * (Samples // 32), nb_classes)  # Size of output after pooling
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Block 1
        x = x.unsqueeze(1)
        x = self.conv1(x)  # Conv1
        x = self.batch_norm1(x)  # BatchNorm1
        x = self.depthwise_conv(x)  # DepthwiseConv2D
        x = self.batch_norm2(x)  # BatchNorm2
        x = self.activation(x)  # ELU Activation
        x = self.pool1(x)  # AveragePooling2D
        x = self.dropout1(x)  # Dropout1

        # Block 2
        x = self.separable_conv(x)  # SeparableConv2D
        x = self.batch_norm3(x)  # BatchNorm3
        x = self.activation(x)  # ELU Activation
        x = self.pool2(x)  # AveragePooling2D
        map = x
        x = self.dropout2(x)  # Dropout2

        # Fully Connected Layer
        x = self.flatten(x)  # Flatten
        x = self.fc(x)  # Dense layer
        x = self.softmax(x)  # Softmax activation

        return x,map


class conv1net(nn.Module):
    def __init__(self, F1=8, kernLength=64, D=2, Chans=22, dropout=0.25):
        super(conv1net, self).__init__()

        F2 = F1 * D
        # reshape
        self.conv0 = nn.Conv1d(in_channels=Chans, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(size=512, mode='linear', align_corners=False)

        # block1
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)
        self.depthwiseConv2D = nn.Conv2d(in_channels=F1, out_channels=F1 * D, kernel_size=1, groups=F1, padding='same')
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.elu = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)

        # block2
        self.separableConv2D = nn.Conv2d(F1 * D, F2, (1, 16), padding='same', bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.avgpool2 = nn.AvgPool2d((1, 4))
        self.dropout2 = nn.Dropout(dropout)
        self.convtrans = nn.ConvTranspose2d(16, 1, (1, 16), stride=(1, 16))

    def forward(self, x):
        if x.shape[2] != 128 or x.shape[3] != 512:
            x = x[:, 0, :, :]
            x = self.conv0(x)
            x = self.upsample(x)
            x = x.unsqueeze(1)
        x = self.conv1(x)
        map = x
        x = self.batchnorm1(x)
        x = self.depthwiseConv2D(x)
        x = self.batchnorm2(x)
        x = self.elu(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        x = self.separableConv2D(x)
        x = self.batchnorm3(x)
        x = self.elu(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        x = self.convtrans(x)
        x = x[:, 0, :, :]
        x=x.unsqueeze(1)
        return x,map
import os






class HPCell(nn.Module):
    def __init__(self, psp_func, NumNodes=512, Pse=PseudoSpikeRect, prams=PARAM_LIST):
        super(HPCell, self).__init__()


        self.psp_func = psp_func

        # self.hebb = torch.zeros(NumNodes, NumNodes).to(self.device)
        self.register_buffer('hebb', torch.zeros(NumNodes, NumNodes))

        self.c_decay = nn.Parameter(torch.ones(1, NumNodes) * prams[0])

        self.v_decay = nn.Parameter(torch.ones(1, NumNodes) * prams[1])

        self.alpha = nn.Parameter((1e-2 * torch.rand(1)), requires_grad=True)

        self.beta = nn.Parameter((1e-2 * torch.rand(1, NumNodes)), requires_grad=True)

        self.eta = nn.Parameter((1e-2 * torch.rand(1, NumNodes)), requires_grad=True)

        self.pseudo_grad_ops = Pse.apply

        self.current = torch.zeros(batch_size, NumNodes)
        # self.current = nn.Parameter(torch.zeros(batch_size, NumNodes), requires_grad=False).to(self.device)

        self.volt = torch.zeros(batch_size, NumNodes)

        self.spike = torch.zeros(batch_size, NumNodes)

        self.state = (self.spike, self.current, self.volt)

        self.numNodes = NumNodes

        self.spikes = []

    def forward(self, input_data):
        pre_spike, pre_current, pre_volt = self.state
        current = self.c_decay * pre_current + self.psp_func(input_data) + self.alpha * input_data.mm(self.hebb)
        volt = self.v_decay * pre_volt * (1. - pre_spike) + current
        output = self.pseudo_grad_ops(volt, VTH, GRAD_WIN)
        h = w_decay * self.hebb + torch.bmm((input_data * self.beta).unsqueeze(2),
                                            ((volt / thresh) - self.eta).tanh().unsqueeze(1)).mean(dim=0).squeeze()
        self.state = (output, current, volt)
        self.hebb = h.data
        self.hebb = self.hebb.clamp(min=-4, max=4)
        self.spikes.append(output.data)
        return output, self.state, self.hebb

    def reset(self,b=batch_size):
        device = self.hebb.device
        self.current = torch.zeros(b, self.numNodes).to(device)
        self.volt = torch.zeros(b, self.numNodes).to(device)
        self.spike = torch.zeros(b, self.numNodes).to(device)
        self.state = (self.spike, self.current, self.volt)
        self.spikes = []


class snn(nn.Module):
    def __init__(self, numnodes=512, params=PARAM_LIST, USE_SE=False, USE_SA=False,
                 USE_TA=False,time_window=512,device='cuda'):
        """
        :param device : 运行设备 gpu or cpu
        :param params :[cdecay,vdecay,vth,grad_win,th_amp,th_decay,base_th]
        残差网络模版： resnet_block_1d1(256, 512, 1,param=[2,1,0])
        """
        super(snn, self).__init__()
        self.USE_SE = USE_SE
        self.USE_SA = USE_SA
        self.USE_TA = USE_TA
        self.time_window = time_window
        self.device = device
        pseudo_grad_ops = PseudoSpikeRect.apply
        self.numnodes = numnodes
        if self.USE_SE:
            self.SE = SE(numnodes)
        if self.USE_TA:
            self.TA = TA(self.time_window)
        if self.USE_SA:
            self.SA = SpatialAttention()
        self.cdecay, self.vdecay, self.vth, self.grad_win, self.th_amp, self.th_decay, self.base_th = params

        self.fc1 = nn.Linear(numnodes, numnodes)
        self.fc2 = nn.Linear(numnodes, numnodes)
        self.fc3 = nn.Linear(numnodes, numnodes)

        self.pseudo_grad_ops = PseudoSpikeRect
        self.conv1 = HPCell(self.fc1, NumNodes=numnodes, Pse=self.pseudo_grad_ops,
                            prams=[self.cdecay, self.vdecay, self.vth, self.grad_win])
        self.conv2 = HPCell(self.fc2, NumNodes=numnodes, Pse=self.pseudo_grad_ops,
                            prams=[self.cdecay, self.vdecay, self.vth, self.grad_win])
        self.conv3 = HPCell(self.fc3, NumNodes=numnodes, Pse=self.pseudo_grad_ops,
                            prams=[self.cdecay, self.vdecay, self.vth, self.grad_win])

        self.dropout = nn.Dropout(0)

    def forward(self, input_spike, time_window=512):
        b = input_spike.shape[0]
        self.conv1.reset(b=b)
        self.conv2.reset(b=b)
        self.conv3.reset(b=b)

        output = []

        # (batch_size,512,28)
        if self.USE_TA:
            input_spike = self.TA(input_spike.view(b, self.numnodes, self.time_window))
        input_spike = input_spike.view(b, self.numnodes, self.time_window)
        input_spike = self.dropout(input_spike)
        for step in range(time_window):
            input_data = input_spike[:, :, step]
            input_data.view(b, self.numnodes)
            if self.USE_SE:
                input_data = self.SE(input_data)
            output_spike, c1_state, hebb1 = self.conv1(input_data)
            output_spike = self.dropout(output_spike)
            output_spike, c2_state, hebb2 = self.conv2(output_spike)
            output_spike = self.dropout(output_spike)
            output_spike, c3_state, hebb3 = self.conv3(output_spike)
            output_spike = self.dropout(output_spike)
            output_spike = output_spike.view(b, self.numnodes)

            output.append(output_spike)
        output = torch.stack(output, dim=2)
        return output

# class EEG_SNN(nn.Module):
#     def __init__(self,n_channels,time_window):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv1d(n_channels,128,kernel_size=5,stride=1,padding=2),
#             nn.Sigmoid()
#         )

#         self.snn = snn(numnodes=128,time_window=time_window)
#         self.classifier = nn.Linear(128,2)
#     def forward(self,x):
#         x = self.conv(x)
#         x = self.snn(x)
#         return self.classifier(x.mean(-1))

class classfier(nn.Module):
    def __init__(self, num_nodes=512, output_feature=4):
        super(classfier, self).__init__()
        self.node = num_nodes
        self.snn = snn(numnodes=self.node, USE_SE=0, USE_TA=0,time_window=512)
        # self.lstm = nn.LSTM(512, 1, 2, batch_first=True)
        self.fc = nn.Linear(self.node, output_feature)

    def forward(self, x):
        batch_size=x.shape[0]
        x = x.reshape(batch_size, self.node, 512)
        x = self.snn(x, 32)  #output shape: (batch_size,128,512)
        # x = self.lstm(x)
        # x=x[0].sum(dim=2)
        # x=x[0][:,:,0]
        x=x[:,:,0]
        x = self.fc(x)
        return x

class EEG_SNN(nn.Module):
    def __init__(self,chans, num_nodes=512, output_feature=2):
        super().__init__()
        self.node = num_nodes
        self.reshape = conv1net(F1=8, kernLength=125, D=2, Chans=chans, dropout=0.1)
        self.cls = classfier(num_nodes, output_feature)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x,alpha=0.1):
        x,map = self.reshape(x.unsqueeze(1))
        y = self.cls(x)
        # domain = self.softmax(domain)
        # cls = self.softmax(cls)
        return y,map


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

if __name__ == '__main__':
    net = EEG_SNN(n_channels=19,time_window=2000)
    net.state_dict()
    x = torch.randn(32,19,2000)
    y = net(x)
    print(y.shape)