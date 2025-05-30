#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
from scipy.io import loadmat
import glob
import numpy as np
import h5py
def generate_patch_from_mat(dir,patch_size,stride):
    file_list = glob.glob(dir + '/*.mat')  # get name list of all .mat files
    file_list = sorted(file_list)  # mcj
    pch_size = patch_size[0]
    stride = stride
    num_patch = 0
    patchs=[]
    for i in range(len(file_list)):
        aa=loadmat(file_list[i])
        keys=aa.keys()
        data = loadmat(file_list[i])[list(keys)[3]]
        data = data[[not np.all(data[i] == 0) for i in range(data.shape[0])], :]
        data = data[:, [not np.all(data[:, i] == 0) for i in range(data.shape[1])]]
        print(data.max())
        print(data.min())
        data =data/abs(data).max()
        H = data.shape[0]
        W = data.shape[1]
        ind_H = list(range(0, H - pch_size + 1, stride[0]))
        if ind_H[-1] < H - pch_size:
            ind_H.append(H - pch_size)
        ind_W = list(range(0, W - pch_size + 1, stride[1]))
        if ind_W[-1] < W - pch_size:
            ind_W.append(W - pch_size)
        for start_H in ind_H:
            for start_W in ind_W:
                patch= data[start_H:start_H + pch_size, start_W:start_W + pch_size, ]
                # patch=patch/abs(patch).max()
                patchs.append(patch)
                num_patch += 1

    print('Total {:d} small images in training set'.format(num_patch))
    print('Finish!\n')
    patchs=np.array(patchs)
    return patchs[:, :, :, np.newaxis]

def generate_patch_from_mat_v2(dir,patch_size,stride):
    file_list = glob.glob(dir + '/*.mat')  # get name list of all .mat files
    exclude_file_names = ['eq-10.mat','eq-69.mat','eq-71.mat','mic-11.mat']
    # 列表推导，过滤掉特定名字的文件
    filtered_files = [file for file in file_list if os.path.basename(file) not in exclude_file_names]
    file_list = sorted(filtered_files)  # mcj

    # filtered_files = [file for file in file_list if os.path.basename(file) in exclude_file_names]
    # file_list = sorted(filtered_files)  # mcj

    pch_size = patch_size[0]
    stride = stride
    num_patch = 0
    patchs=[]
    for i in range(len(file_list)):
        # aa=loadmat(file_list[i])
        # keys=aa.keys()
        # data = loadmat(file_list[i])[list(keys)[3]]
        # 如果读取 v7.3 格式的 .mat 文件：
        with h5py.File(file_list[i], 'r') as f:
            # 列出文件中的变量
            # print("Keys in the file:", list(f.keys()))
            # 假设你要读取变量名为 'data' 的数据
            data = f[list(f.keys())[-1]][:]

        print(data.max())
        print(data.min())
        data =data/abs(data).max()
        H = data.shape[0]
        W = data.shape[1]
        ind_H = list(range(0, H - pch_size + 1, stride[0]))
        if ind_H[-1] < H - pch_size:
            ind_H.append(H - pch_size)
        ind_W = list(range(0, W - pch_size + 1, stride[1]))
        if ind_W[-1] < W - pch_size:
            ind_W.append(W - pch_size)
        for start_H in ind_H:
            for start_W in ind_W:
                patch= data[start_H:start_H + pch_size, start_W:start_W + pch_size, ]
                patchs.append(patch)
                num_patch += 1

    print('Total {:d} small images in training set'.format(num_patch))
    print('Finish!\n')
    patchs = np.array(patchs)
    return patchs[:, :, :, np.newaxis]
def generate_patch_from_noisy_mat(dir,pch_size,stride,sigma=75):
    file_list = glob.glob(dir + '/*.mat')  # get name list of all .mat files
    file_list = sorted(file_list)  # mcj
    pch_size = pch_size
    stride = stride
    sigma=sigma
    num_patch = 0
    patchs=[]
    for i in range(len(file_list)):
        aa=loadmat(file_list[i])
        keys=aa.keys()
        data = loadmat(file_list[i])[list(keys)[3]]
        print(data.max())
        print(data.min())
        data =data/abs(data).max()
        data=data+np.random.normal(0,sigma/255,data.shape)
        H = data.shape[0]
        W = data.shape[1]
        ind_H = list(range(0, H - pch_size + 1, stride[0]))
        if ind_H[-1] < H - pch_size:
            ind_H.append(H - pch_size)
        ind_W = list(range(0, W - pch_size + 1, stride[1]))
        if ind_W[-1] < W - pch_size:
            ind_W.append(W - pch_size)
        for start_H in ind_H:
            for start_W in ind_W:
                patch= data[start_H:start_H + pch_size, start_W:start_W + pch_size, ]
                patchs.append(patch)
                num_patch += 1

    print('Total {:d} small images in training set'.format(num_patch))
    print('Finish!\n')
    patchs=np.array(patchs)
    return patchs[:, :, :, np.newaxis]


# def generate_patch_from_3D_mat(dir,pch_size,stride):
#     file_list = glob.glob(dir + '/*.mat')  # get name list of all .mat files
#     file_list = sorted(file_list)  # mcj
#     pch_size = pch_size
#     stride = stride
#     num_patch = 0
#     patchs=[]
#     for i in range(len(file_list)):
#         aa=loadmat(file_list[i])
#         keys=aa.keys()
#         data = loadmat(file_list[i])[list(keys)[3]]
#         print(data.max())
#         print(data.min())
#         data =data/abs(data).max()
#         H = data.shape[0]
#         W = data.shape[1]
#         ind_H = list(range(0, H - pch_size + 1, stride[0]))
#         if ind_H[-1] < H - pch_size:
#             ind_H.append(H - pch_size)
#         ind_W = list(range(0, W - pch_size + 1, stride[1]))
#         if ind_W[-1] < W - pch_size:
#             ind_W.append(W - pch_size)
#         for start_H in ind_H:
#             for start_W in ind_W:
#                 patch= data[start_H:start_H + pch_size, start_W:start_W + pch_size, ]
#                 patchs.append(patch)
#                 num_patch += 1
#
#     print('Total {:d} small images in training set'.format(num_patch))
#     print('Finish!\n')
#     patchs=np.array(patchs)
#     return patchs[:, :, :, np.newaxis]

if __name__ == '__main__':
    # train_im_list = generate_patch_from_mat(dir="/home/shendi_mcj/datasets/seismic/marmousiShot", pch_size=32, stride=[24, 24])
    train_im_list = generate_patch_from_noisy_mat(dir="/home/shendi_mcj/datasets/seismic/marmousi/marmousi20", pch_size=32,
                                            stride=[24, 24],sigma=75)