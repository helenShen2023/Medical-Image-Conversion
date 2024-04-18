import numpy as np
import nrrd
import os
import cv2
import pydicom
import SimpleITK as sitk


def extra_same_elem(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    iset = set1.intersection(set2)
    return list(iset)


def nrrd_to_png(nrrd_filename, nrrd1, patient_id):
    # nrrd_filename = nrrd_filename
    # nrrd_data, nrrd_options = nrrd.read(nrrd_filename)
    nrrd_data = nrrd1
    h, w, slides_num = nrrd_data.shape
    # 存nii
    ar, num = np.unique(nrrd_data, return_counts=True)
    print(nrrd_data.shape)
    sitk_img = sitk.GetImageFromArray(nrrd_data, isVector=False)
    sitk.WriteImage(sitk_img, os.path.join(nrrd_filename,str(patient_id) + ".nii.gz"))
    # 存png
    # for i in range(slides_num):
    #     img = nrrd_data[:, :, slides_num - i - 1]
    #     # if np.sum(img) == 0:
    #     #     continue
    #     image = cv2.transpose(img)
    #     # image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    #     # cv2.imwrite(nrrd_filename + '/' + str(patient_id) + '-' + str(slides_num-i-1) + '.png', image)200用的
    #     cv2.imwrite(nrrd_filename + '/' + str(patient_id) + '-' + str(slides_num - i) + '.png', image)  # 200其他用的
    # return nrrd1


def rotation(path):
    image = cv2.imread(path)
    image = cv2.transpose(image)
    # image = cv2.flip(image,1)
    cv2.imwrite(path, image)


if __name__ == '__main__':
    root00 = r'F:\Datasets\NPC'
    a0 = os.listdir(root00)
# ignore_cls = [1,4,6,7,10,11,15,17,19,20,21,22,23,27,28,29,30,31,36,38,39,40,44,45]
i0 = 28  # 定义第几个名字
root00 = r'F:/Datasets/NPC/' + str(a0[i0])
# data1 = pydicom.dcmread(root0)
a = os.listdir(root00)
root0 = r'F:/Datasets/NPC/' + str(a0[i0]) + '/' + str(a[1])
nrrd_data, nrrd_options = nrrd.read(root0)
ar, num = np.unique(nrrd_data, return_counts=True)
nrrd1 = nrrd_data + 30  # 定义新的名字
# 对标记颜色进行替换
nrrd1[np.isin(nrrd1, 30)] = 0
nrrd1[np.isin(nrrd1, 31)] = 30
nrrd1[np.isin(nrrd1, 32)] = 19
nrrd1[np.isin(nrrd1, 33)] = 18
nrrd1[np.isin(nrrd1, 34)] = 29
nrrd1[np.isin(nrrd1, 35)] = 28
nrrd1[np.isin(nrrd1, 36)] = 10
nrrd1[np.isin(nrrd1, 37)] = 9
nrrd1[np.isin(nrrd1, 38)] = 24
nrrd1[np.isin(nrrd1, 39)] = 23
nrrd1[np.isin(nrrd1, 40)] = 20
nrrd1[np.isin(nrrd1, 41)] = 11
nrrd1[np.isin(nrrd1, 42)] = 13
nrrd1[np.isin(nrrd1, 43)] = 12
nrrd1[np.isin(nrrd1, 44)] = 26
nrrd1[np.isin(nrrd1, 45)] = 7
nrrd1[np.isin(nrrd1, 46)] = 6
nrrd1[np.isin(nrrd1, 47)] = 8
nrrd1[np.isin(nrrd1, 48)] = 22
nrrd1[np.isin(nrrd1, 49)] = 21
nrrd1[np.isin(nrrd1, 50)] = 5
nrrd1[np.isin(nrrd1, 51)] = 4
nrrd1[np.isin(nrrd1, 52)] = 27
nrrd1[np.isin(nrrd1, 53)] = 15
nrrd1[np.isin(nrrd1, 54)] = 14
nrrd1[np.isin(nrrd1, 55)] = 3
nrrd1[np.isin(nrrd1, 56)] = 2
nrrd1[np.isin(nrrd1, 57)] = 1

ar, num = np.unique(nrrd1, return_counts=True)
root1 = r'F:/pytorch-medical-image-segmentation-master/pytorchmaster/media/Datasets/Bladder/raw_data/labelspddca'
nrrd_to_png(root1, nrrd1, str(228))
# rotation(root1)