from dcmrtstruct2nii.adapters.convert.rtstructcontour2mask import DcmPatientCoords2Mask
from dcmrtstruct2nii.adapters.convert.filenameconverter import FilenameConverter
from dcmrtstruct2nii.adapters.input.contours.rtstructinputadapter import RtStructInputAdapter
from dcmrtstruct2nii.adapters.input.image.dcminputadapter import DcmInputAdapter

import os.path

from dcmrtstruct2nii.adapters.output.niioutputadapter import NiiOutputAdapter
from dcmrtstruct2nii.exceptions import PathDoesNotExistException, ContourOutOfBoundsException

import logging
import difflib

# cite
from dcmrtstruct2nii.cite import cite
from termcolor import cprint

cprint('\nPlease cite:', attrs=["bold"])
cprint(f'{cite}\n')


def list_rt_structs(rtstruct_file,oars):
    """
    Lists the structures in an DICOM RT Struct file by name.

    :param rtstruct_file: Path to the rtstruct file
    :return: A list of names, if any structures are found
    """
    if not os.path.exists(rtstruct_file):
        raise PathDoesNotExistException(f'rtstruct path does not exist: {rtstruct_file}')
    rtreader = RtStructInputAdapter()
    rtstructs = rtreader.ingest(rtstruct_file, True)
    a1=[struct['name'] for struct in rtstructs]
    a4=[]
    a2=[struct['roi_number'] for struct in rtstructs]
    a3 = [struct['display_color'] for struct in rtstructs]
    l3=[list(t) for t in zip(a1, a2,a3)]
    print(l3)
    for ii in range(len(oars)):
        print(difflib.get_close_matches(oars[ii], a1, 1, cutoff=0.9))
        if difflib.get_close_matches(oars[ii], a1, 1, cutoff=0.9) == []:
            a4.append(255)
        else:
            a4.append(a1.index(' '.join(str(i) for i in difflib.get_close_matches(oars[ii], a1, 1, cutoff=0.9))))
        # a4.append(difflib.get_close_matches(oars[ii], l3[ii][0], 1, cutoff=0.9))
    return a4


def dcmrtstruct2nii(rtstruct_file, dicom_file, output_path='', structures=None, gzip=True,
                    mask_background_value=0, mask_foreground_value=255,
                    convert_original_dicom=True):
    """
    Converts A DICOM and DICOM RT Struct file to nii
    :param rtstruct_file: Path to the rtstruct file
    :param dicom_file: Path to the dicom file
    :param output_path: Output path where the masks are written to
    :param structures: Optional, list of structures to convert
    :param gzip: Optional, output .nii.gz if set to True, default: True
    :param save：Optional, whether to save nii file default:True
    :raise InvalidFileFormatException: Raised when an invalid file format is given.
    :raise PathDoesNotExistException: Raised when the given path does not exist.
    :raise UnsupportedTypeException: Raised when conversion is not supported.
    :raise ValueError: Raised when mask_background_value or mask_foreground_value is invalid.
    """
    output_path = os.path.join(output_path, '')  # make sure trailing slash is there

    if not os.path.exists(rtstruct_file):
        raise PathDoesNotExistException(f'rtstruct path does not exist: {rtstruct_file}')

    if not os.path.exists(dicom_file): #20230517
        raise PathDoesNotExistException(f'DICOM path does not exists: {dicom_file}')

    if mask_background_value < 0 or mask_background_value > 255:
        raise ValueError(f'Invalid value for mask_background_value: {mask_background_value}, must be between 0 and 255')

    if mask_foreground_value < 0 or mask_foreground_value > 255:
        raise ValueError(f'Invalid value for mask_foreground_value: {mask_foreground_value}, must be between 0 and 255')

    if structures is None:
        structures = []
    #### 如果输出路径为空，不进行文件保存
    if output_path != '':
        os.makedirs(output_path, exist_ok=True)

    filename_converter = FilenameConverter()
    rtreader = RtStructInputAdapter()

    rtstructs = rtreader.ingest(rtstruct_file)
    dicom_image = DcmInputAdapter().ingest(dicom_file)

    dcm_patient_coords_to_mask = DcmPatientCoords2Mask()
    nii_output_adapter = NiiOutputAdapter()
    mask_name = []  ##定义保存结构体名字的列表
    mask_Image = []  ##定义保存Image的列表
    for rtstruct in rtstructs:
        if len(structures) == 0 or rtstruct['name'] in structures:
            if not 'sequence' in rtstruct:
                logging.info('Skipping mask {} no shape/polygon found'.format(rtstruct['name']))
                continue

            logging.info('Working on mask {}'.format(rtstruct['name']))
            try:
                mask = dcm_patient_coords_to_mask.convert(rtstruct['sequence'], dicom_image, mask_background_value,
                                                          mask_foreground_value)
            except ContourOutOfBoundsException:
                logging.warning(f'Structure {rtstruct["name"]} is out of bounds, ignoring contour!')
                continue

            mask.CopyInformation(dicom_image)#20230527

            mask_filename = filename_converter.convert(f'mask_{rtstruct["name"]}')
            mask_name.append(mask_filename)
            mask_Image.append((mask))
            if output_path != '':  #### 如果输出路径为空，不进行文件保存
                nii_output_adapter.write(mask, f'{output_path}{mask_filename}', gzip)
    return mask_name, mask_Image  ##返回对应的两个列表

    if convert_original_dicom:
        logging.info('Converting original DICOM to nii')
        nii_output_adapter.write(dicom_image, f'{output_path}image', gzip)

    logging.info('Success!')
