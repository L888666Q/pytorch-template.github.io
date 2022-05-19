from torchvision import datasets, transforms
from base import BaseDataLoader
import torch
from torch.utils.data import DataLoader
import torchvision


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, data_split_dir, batch_size, shuffle=True, is_split=False, num_workers=1, training=True):

        if is_split == True:
            data_set_split(data_dir, data_split_dir)
            data_split_dir = data_dir + "/../" + data_split_dir
        else:
            data_split_dir = data_split_dir

        trsfm = transforms.Compose([
            transforms.RandomRotation([-10,10]),#随机旋转 旋角度太大 窄边 补0太多效果没那么好 
            #transforms.Resize([188,188]), #缩放
            #transforms.Resize(config.input_size),
            transforms.Grayscale(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=(0.9,1.1),contrast=(0.8,1.2)),
            #transforms.CenterCrop([132,132]),
            #transforms.RandomCrop(config.input_size),
            transforms.ToTensor(),
            #transforms.Normalize((0.1307,), (0.3081,))
        ])
        val_trsfm = transforms.Compose([
            #transforms.RandomRotation([-10,10]),#随机旋转 旋角度太大 窄边 补0太多效果没那么好 
            #transforms.Resize([188,188]), #缩放
            #transforms.Resize(config.input_size),
            transforms.Grayscale(),
            #transforms.RandomHorizontalFlip(p=0.5),
            #transforms.RandomVerticalFlip(p=0.5),
            #transforms.ColorJitter(brightness=(0.9,1.1),contrast=(0.8,1.2)),
            #transforms.CenterCrop([132,132]),
            #transforms.RandomCrop(config.input_size),
            transforms.ToTensor(),
            #transforms.Normalize((0.1307,), (0.3081,))
        ])
        #data_split_dir = data_dir + "/../" + data_split_dir
        self.data_dir = data_split_dir+'/train'
        print(self.data_dir)
        train_dataset = datasets.ImageFolder(self.data_dir, trsfm)
        self.data_dir = data_split_dir+'/val'
        print(self.data_dir)
        val_dataset = datasets.ImageFolder(self.data_dir, val_trsfm)
        #self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(train_dataset, val_dataset, batch_size, shuffle, num_workers)

#for test
class Dataset(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=None, is_valid_file=None):
        super(Dataset, self).__init__(root, transform=transform, target_transform=target_transform, is_valid_file=is_valid_file)
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, path

class TestDataLoader(DataLoader):
    def __init__(self, data_dir, batch_size, shuffle=False, num_workers=0):

        trsfm = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
        print(data_dir)
        #self.data_dir = data_dir
        #test_dataset = datasets.ImageFolder(self.data_dir, trsfm)
        test_dataset = Dataset(data_dir,trsfm)
        super().__init__(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

# 工具类
import os
import random
import shutil
from shutil import copy2


def data_set_split(src_data_folder, target_data_folder_name, train_scale=0.8, val_scale=0.1, test_scale=0.1):
    '''
    读取源数据文件夹，生成划分好的文件夹，分为trian、val、test三个文件夹进行
    :param src_data_folder: 源文件夹 E:/biye/gogogo/note_book/torch_note/data/utils_test/data_split/src_data
    :param target_data_folder: 目标文件夹 E:/biye/gogogo/note_book/torch_note/data/utils_test/data_split/target_data
    :param train_scale: 训练集比例
    :param val_scale: 验证集比例
    :param test_scale: 测试集比例
    :return:
    '''
    print("开始数据集划分")
    target_data_folder = src_data_folder + "/../" + target_data_folder_name
    class_names = os.listdir(src_data_folder)
    # 在目标目录下创建文件夹
    split_names = ['train', 'val', 'test']
    for split_name in split_names:
        split_path = os.path.join(target_data_folder, split_name)
        if os.path.isdir(split_path):
            pass
        else:
            os.mkdir(split_path)
        # 然后在split_path的目录下创建类别文件夹
        for class_name in class_names:
            class_split_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_split_path):
                pass
            else:
                os.mkdir(class_split_path)

    # 按照比例划分数据集，并进行数据图片的复制
    # 首先进行分类遍历
    current_all_data = {}
    for class_name in class_names:
        current_class_data_path = os.path.join(src_data_folder, class_name)
        #current_all_data = os.listdir(current_class_data_path)
        #current_data_length = len(current_all_data)
        current_data_length = 0

        for dirpath, dirnames, filenames in os.walk(current_class_data_path):
            #遍历创建所有子文件夹
            for dirname in dirnames:
                src_dir_path = os.path.join(dirpath,dirname)
                train_folder = src_dir_path.replace(src_data_folder, src_data_folder + '/../' + target_data_folder_name + '/train/',1)
                val_folder = src_dir_path.replace(src_data_folder, src_data_folder + '/../' + target_data_folder_name + '/val/',1)
                test_folder = src_dir_path.replace(src_data_folder, src_data_folder + '/../' + target_data_folder_name + '/test/',1)
                #train_folder
                if os.path.isdir(train_folder):
                    pass
                else:
                    os.mkdir(train_folder)
                #val_folder
                if os.path.isdir(val_folder):
                    pass
                else:
                    os.mkdir(val_folder)
                #test_folder
                if os.path.isdir(test_folder):
                    pass
                else:
                    os.mkdir(test_folder)
            #遍历所有图像路径
            for file in filenames:
                current_all_data[current_data_length] = os.path.join(dirpath,file)
                current_data_length = current_data_length + 1
        print(current_data_length)
        current_data_index_list = list(range(current_data_length))
        random.shuffle(current_data_index_list)

        #train_folder = os.path.join(os.path.join(target_data_folder, 'train'), class_name)
        #val_folder = os.path.join(os.path.join(target_data_folder, 'val'), class_name)
        #test_folder = os.path.join(os.path.join(target_data_folder, 'test'), class_name)
        train_stop_flag = current_data_length * train_scale
        val_stop_flag = current_data_length * (train_scale + val_scale)
        current_idx = 0
        train_num = 0
        val_num = 0
        test_num = 0
        for i in current_data_index_list:
            #src_img_path = os.path.join(current_class_data_path, current_all_data[i])
            src_img_path = current_all_data[i]
            train_folder = src_img_path.replace(src_data_folder, src_data_folder + '/../' + target_data_folder_name + '/train/',1)
            val_folder = src_img_path.replace(src_data_folder, src_data_folder + '/../' + target_data_folder_name + '/val/',1)
            test_folder = src_img_path.replace(src_data_folder, src_data_folder + '/../' + target_data_folder_name + '/test/',1)
            if current_idx <= train_stop_flag:
                copy2(src_img_path, train_folder)
                #print("{}复制到了{}".format(src_img_path, train_folder))
                train_num = train_num + 1
            elif (current_idx > train_stop_flag) and (current_idx <= val_stop_flag):
                copy2(src_img_path, val_folder)
                #print("{}复制到了{}".format(src_img_path, val_folder))
                val_num = val_num + 1
            else:
                copy2(src_img_path, test_folder)
                #print("{}复制到了{}".format(src_img_path, test_folder))
                test_num = test_num + 1

            current_idx = current_idx + 1

        print("*********************************{}*************************************".format(class_name))
        print(
            "{}类按照{}：{}：{}的比例划分完成，一共{}张图片".format(class_name, train_scale, val_scale, test_scale, current_data_length))
        train_folder = os.path.join(os.path.join(target_data_folder, 'train'), class_name)
        val_folder = os.path.join(os.path.join(target_data_folder, 'val'), class_name)
        test_folder = os.path.join(os.path.join(target_data_folder, 'test'), class_name)
        print("训练集{}：{}张".format(train_folder, train_num))
        print("验证集{}：{}张".format(val_folder, val_num))
        print("测试集{}：{}张".format(test_folder, test_num))


#if __name__ == '__main__':
#    src_data_folder = "E:/biye/gogogo/note_book/torch_note/data/utils_test/data_split/src_data"
#    target_data_folder = "E:/biye/gogogo/note_book/torch_note/data/utils_test/data_split/target_data"
#    data_set_split(src_data_folder, target_data_folder)