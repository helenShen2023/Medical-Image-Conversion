import os
import cv2
import numpy as np
import pandas as pd

img_size = 512
fixed_size = [img_size, img_size, 3]

# 更换路径 dataset = ['train', 'test_a', 'test_b']
path_npy = 'F:/2challenge/rule/datasets/chest/train/maskss/'
path_csv = 'F:/2challenge/rule/datasets/chest/train/train.csv'
df = pd.read_csv(path_csv)

for index, row in df.iterrows():
    scapula = eval(row['scapula'])
    vertebra = eval(row['vertebra'])
    rib = eval(row['rib'])
    lung = eval(row['lung'])
    heart = eval(row['heart'])
    trachea = eval(row['trachea'])
    clavicle = eval(row['clavicle'])

    # 肩胛骨
    mask_scapula = np.zeros((len(scapula), img_size, img_size))
    for j in range(len(scapula)):
        temp = np.zeros(fixed_size)
        pts_scapula = np.array(scapula[j]) / 4
        temp_scapula = cv2.fillPoly(temp, np.int32([pts_scapula]), (1, 1, 1))
        mask_scapula[j] = temp_scapula[:, :, 0]

    # 胸椎
    mask_vertebra = np.zeros((len(vertebra), img_size, img_size))
    for j in range(len(vertebra)):
        temp = np.zeros(fixed_size)
        pts_vertebra = np.array(vertebra[j]) / 4
        temp_vertebra = cv2.fillPoly(temp, np.int32([pts_vertebra]), (2, 2, 2))
        mask_vertebra[j] = temp_vertebra[:, :, 0]

    # 肋骨
    mask_rib = np.zeros((len(rib), img_size, img_size))
    for j in range(len(rib)):
        temp = np.zeros(fixed_size)
        pts_rib = np.array(rib[j]) / 4
        temp_rib = cv2.fillPoly(temp, np.int32([pts_rib]), (3, 3, 3))
        mask_rib[j] = temp_rib[:, :, 0]

    # 肺野
    mask_lung = np.zeros((len(lung), img_size, img_size))
    for j in range(len(lung)):
        temp = np.zeros(fixed_size)
        pts_lung = np.array(lung[j]) / 4
        temp_lung = cv2.fillPoly(temp, np.int32([pts_lung]), (4, 4, 4))
        mask_lung[j] = temp_lung[:, :, 0]

    # 心脏
    mask_heart = np.zeros((1, img_size, img_size))
    temp = np.zeros(fixed_size)
    pts_heart = np.array(heart) / 4
    temp_heart = cv2.fillPoly(temp, np.int32([pts_heart]), (5, 5, 5))
    mask_heart[0] = temp_heart[:, :, 0]

    # 气管
    mask_trachea = np.zeros((1, img_size, img_size))
    temp = np.zeros(fixed_size)
    pts_trachea = np.array(trachea) / 4
    temp_trachea = cv2.fillPoly(temp, np.int32([pts_trachea]), (6, 6, 6))
    mask_trachea[0] = temp_trachea[:, :, 0]

    # 锁骨
    mask_clavicle = np.zeros((len(clavicle), img_size, img_size))
    for j in range(len(clavicle)):
        temp = np.zeros(fixed_size)
        pts_clavicle = np.array(clavicle[j]) / 4
        temp_clavicle = cv2.fillPoly(temp, np.int32([pts_clavicle]), (7, 7, 7))
        mask_clavicle[j] = temp_clavicle[:, :, 0]

    # 汇总
    # target = np.zeros((40, img_size, img_size), np.float16)
    # target[0:2] = mask_scapula #肩胛骨
    # target[2:14] = mask_vertebra #椎体
    # target[14:34] = mask_rib#肋骨
    # target[34:36] = mask_lung#肺
    # target[36:37] = mask_heart#心脏
    # target[37:38] = mask_trachea#气管
    # target[38:40] = mask_clavicle#锁骨
    # ar, num = np.unique(target, return_counts=True)
    # print(target2.shape)
    # target2=np.zeros((7, img_size, img_size), np.int16)
    # target2[0]=mask_scapula[0]+mask_scapula[1]
    # target2[1]=mask_vertebra[0]+mask_vertebra[1]+mask_vertebra[2]+\
    #         mask_vertebra[3]+mask_vertebra[4]+mask_vertebra[5]+mask_vertebra[6]+mask_vertebra[7]+\
    #         mask_vertebra[8]+mask_vertebra[9]+mask_vertebra[10]+mask_vertebra[11]
    # target2[2]=mask_rib[0]+\
    #         mask_rib[1]+mask_rib[2]+mask_rib[3]+mask_rib[4]+mask_rib[5]+mask_rib[6]+mask_rib[7]+mask_rib[8]+\
    #         mask_rib[9]+mask_rib[10]+mask_rib[11]+mask_rib[12]+mask_rib[13]+mask_rib[14]+mask_rib[15]+mask_rib[16]+mask_rib[17]+mask_rib[18]+mask_rib[19]
    # target2[3]=mask_lung[0]+mask_lung[1]
    # target2[4]=mask_heart
    # target2[5]=mask_trachea
    # target2[6]=mask_clavicle[0]+mask_clavicle[1]
    target1=np.zeros((img_size,img_size),np.int16)
    # target1=mask_scapula[0]+mask_scapula[1]+mask_vertebra[0]+mask_vertebra[1]+mask_vertebra[2]+\
    #         mask_vertebra[3]+mask_vertebra[4]+mask_vertebra[5]+mask_vertebra[6]+mask_vertebra[7]+\
    #         mask_vertebra[8]+mask_vertebra[9]+mask_vertebra[10]+mask_vertebra[11]+mask_rib[0]+\
    #         mask_rib[1]+mask_rib[2]+mask_rib[3]+mask_rib[4]+mask_rib[5]+mask_rib[6]+mask_rib[7]+mask_rib[8]+\
    #         mask_rib[9]+mask_rib[10]+mask_rib[11]+mask_rib[12]+mask_rib[13]+mask_rib[14]+mask_rib[15]+mask_rib[16]+mask_rib[17]+mask_rib[18]+mask_rib[19]+\
    #         +mask_lung[0]+mask_lung[1]+mask_heart+mask_trachea+mask_clavicle[0]+mask_clavicle[1]
    mask_scapula1=mask_scapula[0]+mask_scapula[1]
    mask_scapula1=np.where(mask_scapula1>0,1,0)
    mask_vertebra1=mask_vertebra[0]+mask_vertebra[1]+mask_vertebra[2]+\
            mask_vertebra[3]+mask_vertebra[4]+mask_vertebra[5]+mask_vertebra[6]+mask_vertebra[7]+\
            mask_vertebra[8]+mask_vertebra[9]+mask_vertebra[10]+mask_vertebra[11]
    mask_vertebra1=np.where(mask_vertebra1 > 0, 2, 0)
    target1=mask_scapula1+mask_vertebra1
    target1[target1==3] = 100
    ar, num = np.unique(target1, return_counts=True)
    mask_rib1=mask_rib[0]+mask_rib[1]+mask_rib[2]+mask_rib[3]+mask_rib[4]+mask_rib[5]+mask_rib[6]+mask_rib[7]+mask_rib[8]+\
            mask_rib[9]+mask_rib[10]+mask_rib[11]+mask_rib[12]+mask_rib[13]+mask_rib[14]+mask_rib[15]+mask_rib[16]+mask_rib[17]+mask_rib[18]+mask_rib[19]
    mask_rib1=np.where(mask_rib1 > 0, 3, 0)
    target1=target1+mask_rib1
    ar, num = np.unique(target1, return_counts=True)
    target1[target1 == 4] = 200
    target1[target1 == 5] = 300
    target1 = target1 +mask_lung[0]+mask_lung[1]
    ar, num = np.unique(target1, return_counts=True)
    target1[target1 == 5] = 400
    target1[target1 == 6] = 500
    target1[target1 == 7] = 600
    target1 = target1 +mask_heart
    ar, num = np.unique(target1, return_counts=True)
    target1[target1 == 6] = 700
    target1[target1 == 7] = 800
    target1[target1 == 8] = 900
    target1[target1 == 9] = 1000
    target1 = target1 +mask_trachea
    ar, num = np.unique(target1, return_counts=True)
    target1[target1 == 7] = 1100
    target1[target1 == 8] = 1200
    target1[target1 == 9] = 1300
    target1[target1 == 10] = 1400
    target1[target1 == 11] = 1500
    mask_clavicle1=mask_clavicle[0]+mask_clavicle[1]
    mask_clavicle1=np.where(mask_clavicle1 > 0, 7, 0)
    target1 = target1 +mask_clavicle1
    # set_c = set(target1) & set(list_b)
    # list_c = list(set_c)
    ar, num = np.unique(target1, return_counts=True)
    # 保存成npy
    for ii in range(len(ar)):
     target1[target1 == ar[ii]] = ii

    np.save(os.path.join(path_npy, row['filename']), target1)