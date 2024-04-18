import os
import glob
import random
import shutil
from os.path import join
from pathlib import Path
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm

# from utils import read_list, read_nifti, config
np.random.seed(1337)
random.seed(1337)


class Data_Preprocess:
    def __init__(self, Base_path):
        self.Base_path = Base_path


    def write_txt(self,list, path):
        with open(path, 'w') as f:
            for data in list:
                f.write(str(data) + '\n')

    def do_default(self):
        self.do_process(self.val_ids, 'Va')
        self.do_process(self.test_ids, 'Ts')
        self.do_process(self.unlabeled_ids, 'Un')
        self.do_process(self.labeled_ids, 'La')

    def split_dataset(self):
		# 当前函数注意第一个labelsTr是真实数据的路径，是对主目录的下级路径添加， base_dir + labelsTr
        self.img_ids = []
        for path in glob.glob(os.path.join(base_dir, f'labelsTr', '*.nii.gz')):
            img_id = path.split('/')[-1].split('.nii.gz')[0]
            self.img_ids.append(img_id)

        self.test_ids = []
        for i in range(0, 0):
            img_idx = random.choices(list(range(len(self.img_ids))), k=1)[0]
            self.test_ids.append(self.img_ids[img_idx])
            self.img_ids.pop(int(img_idx))
        self.write_txt(
            self.test_ids,
            os.path.join(base_dir, './test.txt')
        )

        self.val_ids = []
        for i in range(0, 50):
            img_idx = random.choices(list(range(len(self.img_ids))), k=1)[0]
            self.val_ids.append(self.img_ids[img_idx])
            self.img_ids.pop(int(img_idx))
        self.write_txt(
            self.val_ids,
            os.path.join(base_dir, './val.txt')
        )

        self.labeled_ids = []
        for i in range(0, 4):
            img_idx = random.choices(list(range(len(self.img_ids))), k=1)[0]
            self.labeled_ids.append(self.img_ids[img_idx])
            self.img_ids.pop(int(img_idx))
        self.write_txt(
            self.labeled_ids,
            os.path.join(base_dir, './labeled.txt')
        )
        self.write_txt(
            self.labeled_ids,
            os.path.join(base_dir, './train.txt')
        )

        self.unlabeled_ids = self.img_ids
        self.write_txt(
            self.unlabeled_ids,
            os.path.join(base_dir, './unlabeled.txt')
        )
        self.write_txt(
            self.unlabeled_ids,
            os.path.join(base_dir, './train.txt')
        )

    def do_process(self,data_list,tag):
    	# 当前函数注意imagesTr \labelsTr是真实数据的路径，是对主目录的下级路径添加， base_dir + imagesTr\labelsTr
        data_num = len(data_list)
        print(tag,'set has {} images'.format(data_num))

        if not os.path.exists(join(self.Base_path, f'image{tag}')):    # 创建保存目录
            os.makedirs(join(join(self.Base_path, f'image{tag}')))
        if not os.path.exists(join(self.Base_path, f'label{tag}')):  # 创建保存目录
            os.makedirs(join(join(self.Base_path, f'label{tag}')))

        for i,img_id in enumerate(data_list):
            print("==== {}/{} ====".format(i + 1, data_num))
            image_path = os.path.join(base_dir, f'imagesTr', f'{img_id}.nii.gz')
            label_path = os.path.join(base_dir, f'labelsTr', f'{img_id}.nii.gz')
            shutil.copy(image_path,join(self.Base_path, f'image{tag}', f'{img_id}.nii.gz'))
            # print(image_path, join(self.Base_path, f'image{tag}', f'{img_id}.nii.gz'))
            label = sitk.ReadImage(label_path)
            label_array = sitk.GetArrayFromImage(label)

            label_array[label_array == 1] = 0
            label_array[label_array == 3] = 0
            # label_array[label_array==5]=4
            temp_array = label_array
            temp_image = sitk.GetImageFromArray(temp_array.astype(np.uint8))
            temp_image.SetOrigin(label.GetOrigin())
            temp_image.SetSpacing(label.GetSpacing())
            temp_image.SetDirection(label.GetDirection())
            sitk.WriteImage(temp_image, join(self.Base_path, f'label{tag}', f'{img_id}.nii.gz'))
            # print(image_path,label_path)


if __name__ == '__main__':
	# 自己修改路径
    base_dir = r'/home/dluser/dataset/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task66_k'
    data_preprocess = Data_Preprocess(base_dir)
    # 把数据集分成了几个文件夹，主要为do_default函数中定义的4个，先自动写txt文件
    # 然后从源数据文件夹中开始创建新文件夹并划分数据
    data_preprocess.split_dataset()
    # # data_preprocess.split_dataset()
    # # data_preprocess.do_npy()
    data_preprocess.do_default()

