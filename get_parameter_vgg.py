import torch
import numpy as np
import torchvision
from torchvision import transforms as transforms
from torch.autograd import Variable

from model.model import ASPNET


def GetConvParam(index, net):
    header = ""
    conv_i = 1
    for i in index: #0,4,8,12,16
        array = net.features[i].weight
        shape = array.size()
        param = net.features[i].weight.view(-1,9).detach().numpy()
        np.savetxt("param.txt",param, fmt = '%f', delimiter=',', newline = ',\n')
        txt_p = open("param.txt", "r")
        paramtostr = txt_p.read()
        header_i = "static float hw_conv{0}_weights[{1[1]}*{1[0]}*9] = {{\n{2}}};".format(conv_i,shape, paramtostr)
        txt_p.close()
        header += header_i + "\n"
        
        array = net.features[i].bias
        shape = array.size()
        param = net.features[i].bias.view(1,shape[0]).detach().numpy()
        np.savetxt("param.txt",param, fmt = '%f', delimiter=',', newline = ',\n')
        txt_p = open("param.txt", "r")
        paramtostr = txt_p.read()
        header_i = "static float hw_conv{0}_bias[{1[0]}] = {{\n{2}}};".format(conv_i,shape, paramtostr)
        txt_p.close()
        header += header_i + "\n"
        
        conv_i = conv_i + 1
        
    return header
def GetFCParam(index, net):
    header = ""
    fc_i = 1

    array = net.classifier.weight
    shape = array.size()
    param = net.classifier.weight.view(-1,4).detach().numpy()
    np.savetxt("param.txt",param, fmt = '%f', delimiter=',', newline = ',\n')
    txt_p = open("param.txt", "r")
    paramtostr = txt_p.read()
    header_i = "static float hw_fc_weights[{0[1]}*{0[0]}] = {{\n{1}}};".format(shape, paramtostr, fc_i)
    txt_p.close()
    header += header_i + "\n"
    
    array = net.classifier.bias
    shape = array.size()
    param = net.classifier.bias.view(1,shape[0]).detach().numpy()
    np.savetxt("param.txt",param, fmt = '%f', delimiter=',', newline = ',\n')
    txt_p = open("param.txt", "r")
    paramtostr = txt_p.read()
    header_i = "static float hw_fc_bias[{0[0]}] = {{\n{1}}};".format(shape, paramtostr, fc_i)
    txt_p.close()
    header += header_i + "\n"
            
    fc_i = fc_i + 1
        
    return header
def GetNormParam(index, net):
    header = ""
    norm_i = 1
    for i in index: #1,5,9,13,17
        paramtostr = ""
        charactor = [net.features[i].bias, net.features[i].weight, net.features[i].running_mean,net.features[i].running_var]
        for array in charactor:
            shape = array.size()
            param = array.view(1,shape[0]).detach().numpy()
            np.savetxt("param.txt",param, fmt = '%f', delimiter=',', newline = ',\n')
            txt_p = open("param.txt", "r")
            paramtostr += txt_p.read()
            txt_p.close()
        header_i = "static float hw_norm{0}[4*{1[0]}] = {{\n{2}}};".format(norm_i,shape, paramtostr)  
        header += header_i + "\n"
        norm_i = norm_i + 1
    return header
    

device = torch.device('cpu')
checkpoint=torch.load('./saved/models/SPOOF-NET/0507_090801/checkpoint-epoch120.pth')

net = ASPNET().to(device)
net.load_state_dict(checkpoint['state_dict'])

param = ""
conv_idxs = [0,4,8,12,16]
norm_idxs = [1,5,9,13,17]
fc_idxs = [0,1]
param = GetNormParam(norm_idxs, net) + GetConvParam(conv_idxs, net) + GetFCParam(fc_idxs, net)

file_name = "6159_spoof_param.txt"
txt_p = open(file_name, "w")
txt_p.write(param)
txt_p.close()
print("get param succ: " + file_name)

'''
#test
batch_size = 256;
test_dir = "D://test"
print(test_dir)
test_transform = transforms.Compose([transforms.Resize(config.input_size),transforms.Grayscale(),transforms.ToTensor()])
test_set = torchvision.datasets.ImageFolder(test_dir, test_transform)

print(test_set.imgs)

val_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0) #  pin_memory=True

net.eval()
with torch.no_grad():
            for batch_num, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)
                print(data[0] * 255)
                output = net(data)

                print (output)

'''