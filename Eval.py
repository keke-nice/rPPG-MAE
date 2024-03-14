import cv2
import os
import numpy as np
import shutil
import pandas as pd
import scipy.io as scio
from scipy import interpolate
import scipy.io as io

# 功能： 对每段视频计算平均HR
# 注释： 1.大约有5个文件时间对不齐需要删除  2.有的检测不到lmk
def MyEval(HR_pr, HR_rel):
    HR_pr = np.array(HR_pr).reshape(-1)
    HR_rel = np.array(HR_rel).reshape(-1)
    temp = HR_pr - HR_rel
    me = np.mean(temp)
    std = np.std(temp)
    mae = np.sum(np.abs(temp))/len(temp)
    rmse = np.sqrt(np.sum(np.power(temp, 2))/len(temp))
    mer = np.mean(np.abs(temp) / HR_rel)
    p = np.sum((HR_pr - np.mean(HR_pr))*(HR_rel - np.mean(HR_rel))) / (
                0.01 + np.linalg.norm(HR_pr - np.mean(HR_pr), ord=2) * np.linalg.norm(HR_rel - np.mean(HR_rel), ord=2))
    print('| me: %.4f' % me,
          '| std: %.4f' % std,
          '| mae: %.4f' % mae,
          '| rmse: %.4f' % rmse,
          '| mer: %.4f' % mer,
          '| p: %.4f' % p
          )
    return me, std, mae, rmse, mer, p


gt_name = 'Label_CSI/HR.mat'
frames_num = 224
gt_av = []
pr_av = []
gt_ps = []
pr_ps = []
# Idex_files = r'/home/haolu/Data/VIPL_Index/VIPL_STMap51250_Test'
Idex_files = r'/scratch/project_2006419/data/VIPL_Index/VIPL_STMap50_Test' #change to your index file path
# pr_path = r'/home/haolu/Code/FG/FG2020/rPPGNetResizen256fn5fi0HR_pr.mat'
pr_path = r'./finetune_VIPLPOSCHROM_CEP_128_8_0.8/HR_pr.mat'
rel_path = r'/finetune_VIPLPOSCHROM_CEP_128_8_0.8/HR_rel.mat'
pr = scio.loadmat(pr_path)['HR_pr']
pr = np.array(pr.astype('float32')).reshape(-1)
rel = scio.loadmat(rel_path)['HR_rel']
rel = np.array(rel.astype('float32')).reshape(-1)
# files_list = os.listdir(Idex_files)
files_list = sorted(os.listdir(Idex_files))
temp = scio.loadmat(os.path.join(Idex_files, files_list[0]))
lastPath = str(temp['Path'][0])
pr_temp = []
gt_temp = []
for HR_index in range(pr.size-1):
    temp = scio.loadmat(os.path.join(Idex_files, files_list[HR_index]))
    nowPath = str(temp['Path'][0])
    Step_Index = int(temp['Step_Index'])
    a = pr[HR_index]
    b = rel[HR_index]

    if lastPath != nowPath:
        print(')************')
        if pr_temp is None:
            print(nowPath)
            print(lastPath)
            pr_temp = []
            gt_temp = []
        else:
            # print('diff_gt', np.array(gt_temp[1:]) - np.array(gt_temp[:-1]))
            # print('diff', np.array(pr_temp[1:]) - np.array(pr_temp[:-1]))
            # print('aaa', np.array(pr_temp)-np.array(gt_temp))
            # print('gt_temp', np.mean(np.array(pr_temp)-np.array(gt_temp)))
            pr_av.append(np.nanmean(pr_temp))
            gt_av.append(np.nanmean(gt_temp))

            print(len(gt_ps))
            print(gt_temp)
            print(pr_temp)
            gt_ps.append(gt_temp)
            pr_ps.append(pr_temp)

            pr_temp = []
            gt_temp = []
    else:
        gt_path = os.path.join(nowPath, gt_name)
        gt = scio.loadmat(gt_path)['HR']
        gt = np.array(gt.astype('float32')).reshape(-1)
        gt = np.nanmean(gt[Step_Index:Step_Index + frames_num])
        gt = gt.astype('float32')
        pr_temp.append(pr[HR_index])
        gt_temp.append(rel[HR_index])
    lastPath = nowPath

io.savemat('gt_ps.mat', {'HR': gt_ps})
io.savemat('pr_ps.mat', {'HR': pr_ps})
io.savemat('HR_rel.mat', {'HR': gt_av})
io.savemat('HR_pr.mat', {'HR': pr_av})
MyEval(gt_av, pr_av)
