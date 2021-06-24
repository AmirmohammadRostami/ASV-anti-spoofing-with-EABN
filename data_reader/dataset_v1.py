import numpy as np
import torch
from torch.utils import data
from kaldi_io import read_mat
import psutil

# import h5py
class SimpleCache(dict):
    def __init__(self, limit, name=None):
        super().__init__()
        self.limit = limit
        self.n_keys = 0
        self.name = name
        if not name:
            self.name = SimpleCache.__name__

    def __setitem__(self, key, value):
        if key in self.keys():
            super().__setitem__(key, value)
        elif self.n_keys < self.limit and psutil.virtual_memory().percent < 90:
            self.n_keys += 1
            super().__setitem__(key, value)
        return value
# PyTorch Dataset 

class SpoofDatsetSystemID(data.Dataset):
    ''' multi-class classification for PA: AA, AB, AC, BA, BB, BC, CA, CB, CC --> 10 classes
        (bonafide: 0), (AA: 1), (AB: 2), (AC: 3), (BA: 4), (BB: 5), (BC: 6),
        (CA: 7), (CB: 8), (CC: 9)

        multi-class classification for LA: SS_1, SS_2, SS_4, US_1, VC_1, VC_4 --> 7 classes
        (bonafide: 0), (SS_1: 1), (SS_2: 2), (SS_4: 3), (US_1: 4), (VC_1: 5), (VC_4: 6)
    '''
    
    def __init__(self, scp_file, utt2index_file, binary_class, is_eval = False):
        with open(scp_file) as f:
            temp = f.readlines()
        content = [x.strip() for x in temp]
        self.key_dic = {index: i.split()[0] for (index, i) in enumerate(content)}
        self.ark_dic = {index: i.split()[1] for (index, i) in enumerate(content)}

        with open(utt2index_file) as f:
            temp = f.readlines()
        temp_dic = {index: int(x.strip().split()[1]) for (index, x) in enumerate(temp)}

        # leave one out 
        self.all_idx = {}
        counter = 0 
        for index, label in temp_dic.items():
            self.all_idx[counter] = index 
            counter += 1
        if binary_class:
            self.label_dic = {index: 0 if orig_label == 0 else 1 for (index, orig_label) in temp_dic.items()}
            self.labels = np.array([0 if orig_label == 0 else 1 for (index, orig_label) in temp_dic.items()])
        else: 
            self.label_dic = temp_dic
            self.labels = np.array([orig_label for (index, orig_label) in temp_dic.items()])
        print(f'{np.sum(self.labels)/len(self.labels)} : 0 lbl, {(len(self.labels)-np.sum(self.labels))/len(self.labels)} : 1 lbl')
        self.weights = np.array([np.sum(self.labels)/len(self.labels),(len(self.labels)-np.sum(self.labels))/len(self.labels)],dtype = np.float32)
        assert len(self.all_idx.keys()) == len(self.label_dic.keys())
        # if is_eval:
        #     self._cache = SimpleCache(1)
        # else:
        self._cache = SimpleCache(len(self.labels))

    def __len__(self):
        return len(self.all_idx.keys())

    def __getitem__(self, counter):
        index = self.all_idx[counter]
        utt_id = self.key_dic[index]
        try:
            x = self._cache[self.ark_dic[index]]
        except KeyError:
            x = np.expand_dims(read_mat(self.ark_dic[index]), axis=0)
            self._cache[self.ark_dic[index]] = x
        y = self.label_dic[index]
        return utt_id, x, y
