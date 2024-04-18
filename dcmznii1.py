from dcmrtstruct2nii import dcmrtstruct2nii, list_rt_structs
import SimpleITK as sitk
import os
import numpy as np
from pathlib import Path
import pandas as df

def setMetaMessage(target, origin):

    target.SetDirection(origin.GetDirection())
    target.SetOrigin(origin.GetOrigin())
    target.SetSpacing(origin.GetSpacing())
    return target

path = Path(r'F:/Datasets/NPC/')
a0 = [e for e in path.iterdir() if e.is_dir()]
m=0
for i0 in a0:
 root00 =i0
 dirs = os.listdir(root00)
 for i in dirs:  # 循环读取路径下的文件并筛选输出
     # if df[df[os.path.splitext(i)[0]].str.contains('RS')]: # 筛选csv文件 os.path.splitext(i)[0] == "RS*":
  if 'RS.1' in os.path.splitext(i)[0] and 'dcm' in os.path.splitext(i)[1]:
      iii=i
  else:
    continue

  path = r'F:/Datasets/NPC/' + str(i0.name) + '/' + iii
  oars = ['brain stem', 'eye-L', 'eye-R', 'lens-L', 'lens-R', 'optic nerve-L', 'optic nerve-R', 'optic chiasma',
             'temp lobe-L', 'temp lobe-R', 'pituitary', 'parotid-L', 'parotid-R', 'inner ear-L', 'inner ear-R',
             'Mid_Ear-L', 'Mid_Ear-R', 'TM joint-L', 'TM joint-R', 'SpinalCord', 'mandibular-L', 'mandibular-R',
             'Submandibular-L', 'Submandibular-R', 'Mandible', 'oral cavity', 'larynx', 'thyroid',
             'trachea']  # 'GTVnx','GTVnd-L','GTVnd-R']#thyroid-R
  # oars=['brain stem','eye-L','eye-R','lens-L','lens-R','optic nerve-L','optic nerve-R','optic chiasma',
  # 'temp lobe-L','temp lobe-R','pituitary','parotid-L','parotid-R','inner ear-L','inner ear-R',
  # 'Mid_Ear-L','Mid_Ear-R','TM joint-L',	'TM joint-R','spine','mandibular-L','mandibular-R',
  # 'Submandibular-L','Submandibular-R','Mandible','oral cavity','larynx','thyroid','trachea']#'GTVnx','GTVnd-L','GTVnd-R']#thyroid-R
  # oars = ['BrainStem', 'Eye_L', 'Eye_R', 'Lens_L', 'Lens_R', 'OpticNerve_L', 'OpticNerve_R', 'OpticChiasm',
  #         'TemporalLobe_L', 'TemporalLobe_R', 'Pituitary', 'Parotid_L', 'Parotid_R', 'inner ear-L', 'inner ear-R',
  #         'Mid_Ear-L', 'Mid_Ear-R', 'TMJ_L', 'TMJ_R', 'SpinalCord', 'Mandible_L', 'mandible_R',
  #         'Submandibular_L', 'Submandibular_R', 'Mandible', 'OralCavity', 'Larynx', 'Thyroid', #Thyroid_R
  #         'Trachea']#'GTVp', 'GTVnd_L', 'GTVnd_R'
  listname=list_rt_structs(path,oars)
  print(listname)
  name, image = dcmrtstruct2nii(path, root00)
  # aq=sitk.GetArrayFromImage(img)
  # s11 = sitk.GetArrayFromImage(image[0])
  # sitk.WriteImage(image, os.path.join('F:/Datasets/NPC/'+str(a0[i0])+'/'+str(os.path.splitext(iii)[0])+'.nii.gz'))
  size=sitk.GetArrayFromImage(image[0]).shape
  nn=0.
  img=np.zeros(size)
  # 参数分别为struct文件、病人图像文件夹，输出文件夹
  for i in listname:
    nn = nn + 1.
    if i==255:
      continue
    img1 = sitk.GetArrayFromImage(image[i])
    img1[img1 > 1] = nn
    img=img+img1
    # ar, num = np.unique(img, return_counts=True)
    img[img > nn] = nn
  # for i in tqdm(range(image.shape[0])):
  #     # img_path = "F:/Datasets/NPC/" + str(i).rjust(3, '0') + ".png"
  #
  #     org_img = normalize_hu(image[i])  # 将像素值归一化到[0,1]区间
  #
  #     # cv2.imshow('imshow', org_img* 255)
  #     # cv2.waitKey(0)
  #     # cv2.destroyAllWindows()
  #     # cv2.imwrite(img_path, org_img * 255)  # 保存图像数组为灰度图(.png)
  #     img=org_img * 255
  #     cv2.imwrite( 'F:\pytorch-medical-image-segmentation-master\med\med_unet\media\Datasets\Bladder/raw_data\imagespddca/' + str(k)+'-'+ str(i+1)+ '.png', img)
  sitk_img = sitk.GetImageFromArray(img, isVector=False)
  sitk_img = setMetaMessage(sitk_img, image[0])
  sitk.WriteImage(sitk_img, os.path.join('F:/Datasets/NPCmask/'+str(m+200)+'.nii.gz'))  #+str(os.path.splitext(i)[0])
  m=m+1
# import matplotlib.pyplot as plt
#
# plt.imshow(img[60])
# plt.show()