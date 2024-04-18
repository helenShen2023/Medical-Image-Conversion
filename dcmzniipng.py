from miscnn.data_loading.interfaces.dicom_io import DICOM_interface
import SimpleITK as sitk
# 创建需要的标记的interface
structure_dict = {"Lung_L": 1, "Lung_R": 1}
interface = DICOM_interface(structure_dict = structure_dict, classes=20, annotation_tag="1.000000-.simplified")

# 获取mask文件列表
from miscnn.data_loading.data_io import Data_IO
data_path='F:/Datasets/NPC/0003Chenshaoqing/RS.1.2.246.352.205.5733944302949213325.7931378431163648896.dcm'
data_io = Data_IO(interface, data_path)
sample_list = data_io.get_indiceslist()
sample_list.sort() # sample_list中有 LICENSE 记得处理删除掉
sample_list=[i for i in sample_list if "LICENSE" not in i ]

# 获取原图的坐标系信息
single_mask_path="../NiiGZ/label255/Test-S1-101.nii.gz"
mask_img=sitk.ReadImage(single_mask_path)

origin =  mask_img.GetOrigin()
spacing = mask_img.GetSpacing()
direction = mask_img.GetDirection()

# 开始转换
for i in sample_list[:3]:
    sample = data_io.sample_loader(i)
    segmentations = sample.seg_data

    mask_img_convert=sitk.GetImageFromArray(segmentations)
    # 恢复到世界坐标系
    mask_img_convert.SetOrigin(origin)
    mask_img_convert.SetSpacing(spacing)
    mask_img_convert.SetDirection(direction)

    name="../NiiGZ/lunglabel/"+i[6:]+'.nii.gz'
    print(name)
    sitk.WriteImage(mask_img_convert,name)
