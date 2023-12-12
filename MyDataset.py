# -*- coding: UTF-8 -*-
import numpy as np
import os
from torch.utils.data import Dataset
import cv2
import csv
import scipy.io as scio
import torchvision.transforms.functional as transF
import torchvision.transforms as transforms
from PIL import Image
from numpy.fft import fft, ifft, rfft, irfft
from torch.autograd import Variable


class Data_DG(Dataset):
    def __init__(self, root_dir, dataName, STMap1, STMap2, frames_num, in_chans=6, transform = None):
        self.root_dir = root_dir
        self.dataName = dataName
        self.STMap_Name1 = STMap1
        self.STMap_Name2 = STMap2
        self.frames_num = int(frames_num)
        self.datalist = os.listdir(root_dir)
        self.datalist = sorted(self.datalist)
        self.num = len(self.datalist)
        self.transform = transform
        self.in_chans = in_chans

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        idx = idx
        img_name = 'STMap'
        STMap_name1 = self.STMap_Name1
        STMap_name2 = self.STMap_Name2
        nowPath = os.path.join(self.root_dir, self.datalist[idx])
        temp = scio.loadmat(nowPath)
        nowPath = str(temp['Path'][0])
        Step_Index = int(temp['Step_Index'])
        STMap_Path = os.path.join(nowPath, img_name)
        # 读取图片序列
        if self.in_chans == 6:
            feature_map1 = cv2.imread(os.path.join(STMap_Path, STMap_name1))
            feature_map2 = cv2.imread(os.path.join(STMap_Path, STMap_name2))
            feature_map1 = feature_map1[:, Step_Index:Step_Index + self.frames_num, :]
            feature_map2 = feature_map2[:, Step_Index:Step_Index + self.frames_num, :]
            feature_map = np.concatenate((feature_map1, feature_map2), axis=2)
            for c in range(feature_map.shape[2]):
              for r in range(feature_map.shape[0]):
                  feature_map[r, :, c] = 255 * ((feature_map[r, :, c] - np.min(feature_map[r, :, c])) \
                              / (0.00001 + np.max(feature_map[r, :,c]) - np.min(feature_map[r, :, c])))
            feature_map1 = Image.fromarray(np.uint8(feature_map[:,:,0:3]))
            feature_map2 = Image.fromarray(np.uint8(feature_map[:,:,3:6]))

            if self.transform:
                feature_map1 = self.transform(feature_map1)
                feature_map2 = self.transform(feature_map2)
                feature_map = np.concatenate((feature_map1, feature_map2), axis = 0)
        else:
            feature_map = cv2.imread(os.path.join(STMap_Path, STMap_name1))
            feature_map = feature_map[:, Step_Index:Step_Index + self.frames_num, :]
            for c in range(feature_map.shape[2]):
                for r in range(feature_map.shape[0]):
                    feature_map[r, :, c] = 255 * ((feature_map[r, :, c] - np.min(feature_map[r, :, c])) \
                                / (0.00001 + np.max(feature_map[r, :,c]) - np.min(feature_map[r, :, c])))
            feature_map = Image.fromarray(np.uint8(feature_map))
            if self.transform:
                feature_map = self.transform(feature_map)
       
        if self.dataName == 'VIPL':
            bvp_name = 'Label_CSI/BVP_Filt.mat'
            bvp_path = os.path.join(nowPath, bvp_name)
            bvp = scio.loadmat(bvp_path)['BVP']
            bvp = np.array(bvp.astype('float32')).reshape(-1)
            bvp = bvp[Step_Index:Step_Index + self.frames_num]
            bvp = (bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp))
            bvp = bvp.astype('float32')

            gt_name = 'Label_CSI/HR.mat'
            gt_path = os.path.join(nowPath, gt_name)
            gt = scio.loadmat(gt_path)['HR']
            gt = np.array(gt.astype('float32')).reshape(-1)
            gt = np.nanmean(gt[Step_Index:Step_Index + self.frames_num])
            gt = gt.astype('float32')

        elif self.dataName == 'PURE':
            bvp_name = 'Label/BVP.mat'
            bvp_path = os.path.join(nowPath, bvp_name)
            bvp = scio.loadmat(bvp_path)['BVP']
            bvp = np.array(bvp.astype('float32')).reshape(-1)
            bvp = bvp[Step_Index:Step_Index + self.frames_num]
            bvp = (bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp))
            bvp = bvp.astype('float32')

            gt_name = 'Label/HR.mat'
            gt_path = os.path.join(nowPath, gt_name)
            gt = scio.loadmat(gt_path)['HR']
            gt = np.array(gt.astype('float32')).reshape(-1)
            gt = np.nanmean(gt[Step_Index:Step_Index + self.frames_num])
            gt = gt.astype('float32')
        elif self.dataName == 'UBFC':
            bvp_name = 'Label/BVP.mat'
            bvp_path = os.path.join(nowPath, bvp_name)
            bvp = scio.loadmat(bvp_path)['BVP']
            bvp = np.array(bvp.astype('float32')).reshape(-1)
            bvp = bvp[Step_Index:Step_Index + self.frames_num]
            bvp = (bvp - np.min(bvp))/(np.max(bvp) - np.min(bvp))
            bvp = bvp.astype('float32')

            gt_name = 'Label/HR.mat'
            gt_path = os.path.join(nowPath, gt_name)
            gt = scio.loadmat(gt_path)['HR']
            gt = np.array(gt.astype('float32')).reshape(-1)
            gt = np.nanmean(gt[Step_Index:Step_Index + self.frames_num])
            gt = gt.astype('float32')


        # 归一化
        return (feature_map, bvp, gt)

def CrossValidation_semi(root_dir, fold_num=5,fold_index=0, semi=10, semi_index=0):
    datalist = os.listdir(root_dir)
    # datalist.sort(key=lambda x: int(x))
    num = len(datalist)
    test_num = round(((num/fold_num) - 2))
    train_num = num - test_num
    test_index = datalist[fold_index*test_num:fold_index*test_num + test_num-1]
    train_index = datalist[0:fold_index*test_num] + datalist[fold_index*test_num + test_num:]
    semi_num = train_num
    semi_withlabel_num = round(((semi_num/semi) - 2))
    semi_withoutlabel_num = semi_num - semi_withlabel_num
    semi_withlabel_index = train_index[semi_index*semi_withlabel_num:semi_index*semi_withlabel_num + semi_withlabel_num-1]
    semi_withoutlabel_index = train_index[0:semi_index*semi_withlabel_num] + train_index[semi_index*semi_withlabel_num + semi_withlabel_num:]
    return test_index, train_index, semi_withlabel_index, semi_withoutlabel_index

def CrossValidation(root_dir, fold_num=5,fold_index=0):
    datalist = os.listdir(root_dir)
    # datalist.sort(key=lambda x: int(x))
    num = len(datalist)
    test_num = round(((num/fold_num) - 2))
    train_num = num - test_num
    test_index = datalist[fold_index*test_num:fold_index*test_num + test_num-1]
    train_index = datalist[0:fold_index*test_num] + datalist[fold_index*test_num + test_num:]
    return test_index, train_index


def getIndex(root_path, filesList, save_path, Pic_path, Step, frames_num):
    Index_path = []
    print('Now processing' + root_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for sub_file in filesList:
        now = os.path.join(root_path, sub_file)
        img_path = os.path.join(now, os.path.join('STMap', Pic_path))
        temp = cv2.imread(img_path)
        Num = temp.shape[1]
        Res = Num - frames_num - 1 -20 # 可能是Diff数据
        Step_num = int(Res/Step)
        for i in range(Step_num):
            Step_Index = i*Step
            temp_path = sub_file + '_' + str(1000 + i) + '_.mat'
            scio.savemat(os.path.join(save_path, temp_path), {'Path': now, 'Step_Index': Step_Index})
            Index_path.append(temp_path)
    return Index_path



