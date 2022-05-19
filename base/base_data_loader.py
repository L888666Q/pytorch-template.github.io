import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import WeightedRandomSampler
from torch.autograd import Variable


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, train_dataset, val_dataset, batch_size, shuffle, num_workers, collate_fn=default_collate):

        self.shuffle = shuffle

        #self.train_sampler, self.valid_sampler = self._split_sampler(train_dataset,val_dataset)
        self.train_sampler, self.train_classes_weight = self.creater_sampler(train_dataset)
        self.valid_sampler, self.val_classes_weight = self.creater_sampler(val_dataset)
        self.init_kwargs = {
            'dataset': train_dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.train_sampler, **self.init_kwargs)

    #def _split_sampler(self,dataset,val_dataset):

        #train_sampler = self.creater_sampler(self,train_dataset)
        #valid_sampler = self.creater_sampler(self,val_dataset)

        #return None

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)

    def creater_sampler(self,train_set):
        classes_idx = train_set.class_to_idx
        appear_times = Variable(torch.zeros(len(classes_idx), 1))
        for label in train_set.targets:
            appear_times[label] += 1
        classes_weight = (1./(appear_times / len(train_set))).view( -1)
        weight=list(map(lambda x:classes_weight[x],train_set.targets))
        #定义sampler
        #print(len(classes_idx))
        #print(len(train_set))
        num_add=abs(appear_times[0]-appear_times[1])
        #print(appear_times[0])
        #print(appear_times[1])
        #print(num_add)
        #num_sample=int(len(train_set)+num_add)
        num_sample=int(len(train_set))
        print("total:{}".format(num_sample)+",targets0:{},targets1:{}".format(appear_times[0],appear_times[1]))
        sampler = WeightedRandomSampler(weight, num_sample, replacement=True)
        return sampler, classes_weight

    def get_classes_weight(self):
        return self.train_classes_weight, self.val_classes_weight