import os
from pathlib import Path
import SimpleITK as sitk

path = './data/'
# 分别令train或test为True，对训练数据和测试数据进行处理
train = True
test = False
if train:
    ###########datatrain
    datasplit = 'train'
    datasplitannot = 'trainannot'
    #########savetrain
    savesplit = 'imagesTr'
    savesplitannot = 'labelsTr'
if test:
    ##########datatest
    datasplit = 'test'
    datasplitannot = 'testannot'
    #########savetest
    savesplit = 'imagesTs'
    savesplitannot = 'labelsTs'

savepath = Path('./Task100_PapSmear/')
savepath_img = os.path.join(savepath, savesplit)
savepath_img = Path(savepath_img)
savepath_mask = os.path.join(savepath, savesplitannot)
savepath_mask = Path(savepath_mask)
savepath.mkdir(exist_ok=True)
savepath_img.mkdir(exist_ok=True)
savepath_mask.mkdir(exist_ok=True)

ImgPath = os.path.join(path, datasplit)
MaskPath = os.path.join(path, datasplitannot)
ImgList = os.listdir(ImgPath)
print('ImgList Num:', len(ImgList))

import cv2
import numpy as np

# print('ImgList:', ImgList)

for NumIndex in range(len(ImgList)):
    # print('ImgName:', os.path.join(ImgPath, ImgList[NumIndex]))
    Img = cv2.imread(os.path.join(ImgPath, ImgList[NumIndex]))
    Mask = cv2.imread(os.path.join(MaskPath, ImgList[NumIndex]), 0)
    # print('Img.shape', Img.shape, Mask.shape)
    Img_Transposed = np.transpose(Img, (2, 0, 1))
    # print('Img_Transposed.shape', Img_Transposed.shape)
    Img_0 = Img_Transposed[0].reshape(1, Img_Transposed[0].shape[0], Img_Transposed[0].shape[1])
    Img_1 = Img_Transposed[1].reshape(1, Img_Transposed[1].shape[0], Img_Transposed[1].shape[1])
    Img_2 = Img_Transposed[2].reshape(1, Img_Transposed[2].shape[0], Img_Transposed[2].shape[1])
    Mask = Mask.reshape(1, Mask.shape[0], Mask.shape[1])

    Img_0_name = ImgList[NumIndex].split('.')[0] + '_0000.nii.gz'
    Img_1_name = ImgList[NumIndex].split('.')[0] + '_0001.nii.gz'
    Img_2_name = ImgList[NumIndex].split('.')[0] + '_0002.nii.gz'
    Mask_name = ImgList[NumIndex].split('.')[0] + '.nii.gz'
    # print('Img_012.shape', Img_0.shape, Img_1.shape, Img_2.shape, Mask.shape, Img_0_name, Mask_name)

    Img_0_nii = sitk.GetImageFromArray(Img_0)
    Img_1_nii = sitk.GetImageFromArray(Img_1)
    Img_2_nii = sitk.GetImageFromArray(Img_2)
    Mask_nii = sitk.GetImageFromArray(Mask)
    # print('save', os.path.join(savepath_img, Img_0_name))

    sitk.WriteImage(Img_0_nii, os.path.join(savepath_img, Img_0_name))
    sitk.WriteImage(Img_1_nii, os.path.join(savepath_img, Img_1_name))
    sitk.WriteImage(Img_2_nii, os.path.join(savepath_img, Img_2_name))
    sitk.WriteImage(Mask_nii, os.path.join(savepath_mask, Mask_name))