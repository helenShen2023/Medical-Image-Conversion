import cv2
import os
import numpy as np
from datasets import bladder
def read_directory1(mask):
    sum1=0
    sum2=0
    sum3=0
    sum4=0
    sum5=0
    sum6=0
    sum7=0
    sum8=0
    sum9=0
    img=np.array(mask)
    # if np.sum(img) == 0:
    #         continue
    ar, num = np.unique(img, return_counts=True)
    # c=(np.array(bladder.palette).T)[0]
    c1 = [1]
    c2 = [6]
    c3 = [7]
    c4 = [8]
    c5 = [12]
    c6 = [13]
    c7 = [23]
    c8 = [24]
    c9 = [25]
    if len(set(ar) & set(c1))>0:
            sum1+=1
            print(sum1)
            # print(filename)
    if len(set(ar) & set(c2))>0:
            sum2+=1
            print(sum2)
    if len(set(ar) & set(c3)) > 0:
            sum3 += 1
            print(sum3)
    if len(set(ar) & set(c4)) > 0:
            sum4 += 1
            print(sum4)
    if len(set(ar) & set(c5)) > 0:
            sum5 += 1
            print(sum5)
    if len(set(ar) & set(c6)) > 0:
            sum6 += 1
            print(sum6)
    if len(set(ar) & set(c7)) > 0:
            sum7 += 1
            print(sum7)
    if len(set(ar) & set(c8)) > 0:
            sum8 += 1
            print(sum8)
    if len(set(ar) & set(c9)) > 0:
            sum9 += 1
            print(sum9)
    return sum1,sum2,sum3,sum4,sum5,sum6,sum7,sum8,sum9

# 读取函数，用来读取文件夹中的所有函数，输入参数是文件名
def read_directory2(directory_name):
    for filename in os.listdir(directory_name):
        # print(filename)  # 仅仅是为了测试
        img = cv2.imread(directory_name + "/" + filename)
        img=np.array(img)
        if np.sum(img) == 0:
              continue
        ar, num = np.unique(img, return_counts=True)
        # c=(np.array(bladder.palette).T)[0]
        # c=[6,7,8]
        # if len(set(ar) & set(c))>0:
        print(ar,num)
        # print(filename)
        # #####显示图片#######
        # cv2.imshow(filename, img)
        # cv2.waitKey(0)
        # #####################
        #
        # #####保存图片#########
        # cv2.imwrite("D://wangyang//face1" + "/" + filename, img)

# 读取函数，用来读取文件夹中的所有函数，输入参数是文件名
def read_directory(directory_name):
    sum1=0
    sum2=0
    sum3=0
    sum4=0
    sum5=0
    sum6=0
    sum7=0
    sum8=0
    sum9=0
    for filename in os.listdir(directory_name):
        # print(filename)  # 仅仅是为了测试
        img = cv2.imread(directory_name + "/" + filename)
        img=np.array(img)
        # if np.sum(img) == 0:
        #         continue
        ar, num = np.unique(img, return_counts=True)
        # c=(np.array(bladder.palette).T)[0]
        c1 = [1]
        c2 = [6]
        c3 = [7]
        c4 = [8]
        c5 = [12]
        c6 = [13]
        c7 = [23]
        c8 = [24]
        c9 = [25]
        if len(set(ar) & set(c1))>0:
            sum1+=1
            print(sum1)
            # print(filename)
        if len(set(ar) & set(c2))>0:
            sum2+=1
            print(sum2)
        if len(set(ar) & set(c3)) > 0:
            sum3 += 1
            print(sum3)
        if len(set(ar) & set(c4)) > 0:
            sum4 += 1
            print(sum4)
        if len(set(ar) & set(c5)) > 0:
            sum5 += 1
            print(sum5)
        if len(set(ar) & set(c6)) > 0:
            sum6 += 1
            print(sum6)
        if len(set(ar) & set(c7)) > 0:
            sum7 += 1
            print(sum7)
        if len(set(ar) & set(c8)) > 0:
            sum8 += 1
            print(sum8)
        if len(set(ar) & set(c9)) > 0:
            sum9 += 1
            print(sum9)
    return sum1,sum2,sum3,sum4,sum5,sum6,sum7,sum8,sum9
                # print(filename)
            # print(filename)
        # #####显示图片#######
        # cv2.imshow(filename, img)
        # cv2.waitKey(0)
        # #####################
        #
        # #####保存图片#########
        # cv2.imwrite("D://wangyang//face1" + "/" + filename, img)

if __name__ == '__main__':
  read_directory2(r"F:/bishe/testt")
  #这里传入所要读取文件夹的绝对路径，加引号（引号不能省略！）
  # [sum1,sum2,sum3,sum4,sum5,sum6,sum7,sum8,sum9]=read_directory(r"F:\pytorch-medical-image-segmentation-master\med\med_unet\media\Datasets\Bladder\raw_data\labelspddca")
  # read_directory2(r"F:\pytorch-medical-image-segmentation-master\med\med_unet\media\Datasets\Bladder\raw_data\labelspddca")

  # print(sum1)
  # print(sum2)
  # print(sum3)
  # print(sum4)
  # print(sum5)
  # print(sum6)
  # print(sum7)
  # print(sum8)
  # print(sum9)
