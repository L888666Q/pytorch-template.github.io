#coding=utf-8
import os
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
#import util
#from models import nin
from model.model import ASPNET
import numpy as np

parser = argparse.ArgumentParser()
# 数据集路径
parser.add_argument('--data', action='store', default='../data',
                    help='dataset path')
# 使用cpu设置为true
parser.add_argument('--cpu', action='store_true',
                    help='disables CUDA training')
# percent(剪枝率)
parser.add_argument('--percent', type=float, default=0.5,
                    help='nin:0.5')
# 正常|规整剪枝标志
parser.add_argument('--normal_regular', type=int, default=1,
                    help='--normal_regular_flag (default: normal)')
# model层数
parser.add_argument('--layers', type=int, default=6,
                    help='layers (default: 9)')
# 稀疏训练后的model
parser.add_argument('--model', default='STANDALONE/model_STANDALONE_2022_02_17_08_52/ckpt_112.pth', type=str, metavar='PATH',
                    help='path to raw trained model (default: none)')
# 剪枝后保存的model
parser.add_argument('--save', default='STANDALONE/model_STANDALONE_2022_02_17_08_52/nin_prune.pth', type=str, metavar='PATH',
                    help='path to save prune model (default: none)')
args = parser.parse_args()
base_number = args.normal_regular
layers = args.layers
print(args)

if base_number <= 0:
    print('\r\n!base_number is error!\r\n')
    base_number = 1

#model = ASP0809_tpc()
model = ASPNET()
if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        model.load_state_dict(torch.load(args.model)['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
print('旧模型: ', model)
total = 0
i = 0
for m in model.modules():
        # 如果是BN层统计一下通道
        print(m)
        if isinstance(m, nn.BatchNorm2d):
            if i < layers - 1:
                i += 1
                print(i)
                total += m.weight.data.shape[0]
print(total)
#print(gggggg)
# 确定剪枝的全局阈值
bn = torch.zeros(total)
index = 0
i = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        if i < layers - 1:
            i += 1
            size = m.weight.data.shape[0]
            bn[index:(index+size)] = m.weight.data.abs().clone()
            index += size
# 按照权值大小排序
y, j = torch.sort(bn)
print(y)
print(j)
thre_index = int(total * args.percent)
if thre_index == total:
    thre_index = total - 1
# 确定要剪枝的阈值
thre_0 = y[thre_index]

#********************************预剪枝*********************************
pruned = 0
cfg_0 = []
cfg = []
cfg_mask = []
i = 0
for k, m in enumerate(model.modules()):
    if isinstance(m, nn.BatchNorm2d):
        if i < layers - 1:
            i += 1

            weight_copy = m.weight.data.clone()
            # 要保留的通道
            mask = weight_copy.abs().gt(thre_0).float()
            remain_channels = torch.sum(mask)
            # 如果全部剪掉的话就提示应该调小剪枝程度了
            if remain_channels == 0:
                print('\r\n!please turn down the prune_ratio!\r\n')
                remain_channels = 1
                mask[int(torch.argmax(weight_copy))]=1

            # ******************规整剪枝******************
            v = 0
            n = 1
            if remain_channels % base_number != 0:
                if remain_channels > base_number:
                    while v < remain_channels:
                        n += 1
                        v = base_number * n
                    if remain_channels - (v - base_number) < v - remain_channels:
                        remain_channels = v - base_number
                    else:
                        remain_channels = v
                    if remain_channels > m.weight.data.size()[0]:
                        remain_channels = m.weight.data.size()[0]
                    remain_channels = torch.tensor(remain_channels)
                        
                    y, j = torch.sort(weight_copy.abs())
                    thre_1 = y[-remain_channels]
                    mask = weight_copy.abs().ge(thre_1).float()
            # 剪枝掉的通道数个数
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            cfg_0.append(mask.shape[0])
            cfg.append(int(remain_channels))
            if  i < layers - 1:
                cfg.append('M')
            cfg_mask.append(mask.clone())
            print('layer_index: {:d} \t total_channel: {:d} \t remaining_channel: {:d} \t pruned_ratio: {:f}'.
                format(k, mask.shape[0], int(torch.sum(mask)), (mask.shape[0] - torch.sum(mask)) / mask.shape[0]))
pruned_ratio = float(pruned/total)
#cfg.append(4)
print(total)
print(pruned)
print('\r\n!预剪枝完成!')
print('total_pruned_ratio: ', pruned_ratio)
#********************************预剪枝后model测试*********************************
def test():
    # 加载测试数据
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root = args.data, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
        batch_size = 64, shuffle=False, num_workers=1)
    model.eval()
    correct = 0

    for data, target in test_loader:
        if not args.cpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    acc = 100. * float(correct) / len(test_loader.dataset)
    print('Accuracy: {:.2f}%\n'.format(acc))
    return
print('************预剪枝模型测试************')
print(cfg)
#if not args.cpu:
#    model.cuda()
#test()

#********************************剪枝*********************************
#newmodel = nin.Net(cfg)
#newmodel = ASP0809_tpc(cfg=cfg)
newmodel = ASPNET(net_name=cfg) # 剪枝后的模型
if not args.cpu:
    newmodel.cuda()
layer_id_in_cfg = 0
start_mask = torch.ones(1)
end_mask = cfg_mask[layer_id_in_cfg]
i = 0
print(newmodel)
for [m0, m1] in zip(model.modules(), newmodel.modules()):
    if isinstance(m0, nn.BatchNorm2d):
        if i < layers - 1:
            i += 1
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            m1.weight.data = m0.weight.data[idx1].clone()
            m1.bias.data = m0.bias.data[idx1].clone()
            m1.running_mean = m0.running_mean[idx1].clone()
            m1.running_var = m0.running_var[idx1].clone()
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):  
                end_mask = cfg_mask[layer_id_in_cfg]
        else:
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()
    elif isinstance(m0, nn.Conv2d):
        if i < layers - 1:
            print(i)
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            print(idx0)
            print(idx1)
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w = m0.weight.data[:, idx0, :, :].clone()
            m1.weight.data = w[idx1, :, :, :].clone()
            m1.bias.data = m0.bias.data[idx1].clone()
        else:
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            m1.weight.data = m0.weight.data[:, idx0, :, :].clone()
            m1.bias.data = m0.bias.data.clone()
    elif isinstance(m0, nn.Linear):
            print("linear")
            #idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            #if idx0.size == 1:
                #idx0 = np.resize(idx0, (1,))
            #m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.weight.data = m0.weight.data.clone()
#******************************剪枝后model测试*********************************
print('新模型: ', newmodel)
print('**********剪枝后新模型测试*********')
model = newmodel
#test()
#******************************剪枝后model保存*********************************
print('**********剪枝后新模型保存*********')
torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, args.save)
print('**********保存成功*********\r\n')

#*****************************剪枝前后model对比********************************
print('************旧模型结构************')
print(cfg_0)
print('************新模型结构************')
print(cfg, '\r\n')