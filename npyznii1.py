#coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.transform import resize
from glob import glob
import SimpleITK as sitk
file_dir = r"F:\2challenge\rule\datasets\chest\train\img\*"  # npy文件路径
dest_dir = r"F:\2challenge\rule\datasets\chest\train\imgnii"  # 文件存储的路径


def npy_png(file_dir, dest_dir):
    # 如果不存在对应文件，则创建对应文件
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    path = glob(file_dir)
    k=0
    for file in path:
        arr = np.load(file)
        file1 = file.split('\\')[-1]
        file1 = file1.split('.')[0]
        z = arr.shape[0] # 获取Z轴大小


        arr1 = arr #每次增长1 slice
        arr2 = arr1[0, ...] # 将其转换为两维，因为Z轴当前为1，可以省略。
        # disp_to_img = resize(arr2, [128, 128])
        plt.imsave(os.path.join(dest_dir, "{}.png".format(str(file1))), arr2, cmap='gray')  # 定义命名规则，保存图片为彩色模式
        # ## npy文件转换为nii文件
        # ar, num = np.unique(arr, return_counts=True)
        # arr = arr.astype(np.int16)
        # # arr = arr[np.newaxis, :]
        # print(arr.shape)
        # sitk_img = sitk.GetImageFromArray(arr, isVector=False)
        # ar, num = np.unique(sitk_img, return_counts=True)
        # sitk.WriteImage(sitk_img, os.path.join(dest_dir, str(file1) + ".nii.gz"))
        # print('file_name:{}'.format(file))
        # sitk_img=[]
        # arr=[]


if __name__ == "__main__":
    npy_png(file_dir, dest_dir)