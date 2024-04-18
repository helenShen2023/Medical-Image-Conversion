import numpy as np
import os
import SimpleITK as sitk
import shutil
from batchgenerators.utilities.file_and_folder_operations import save_json, subfiles
from typing import Tuple

def get_identifiers_from_splitted_files(folder: str):
    uniques = np.unique([i[:-7] for i in subfiles(folder, suffix='.nii.gz', join=False)])
    return uniques

def generate_dataset_json(output_file: str, imagesTr_dir: str, imagesTs_dir: str, modalities: Tuple,
                          labels: dict, dataset_name: str, license: str = "", dataset_description: str = "",
                          dataset_reference="", dataset_release='1.1 14/08/2018'):
    """
    :param output_file: This needs to be the full path to the dataset.json you intend to write, so
    output_file='DATASET_PATH/dataset.json' where the folder DATASET_PATH points to is the one with the
    imagesTr and labelsTr subfolders
    :param imagesTr_dir: path to the imagesTr folder of that dataset
    :param imagesTs_dir: path to the imagesTs folder of that dataset. Can be None
    :param modalities: tuple of strings with modality names. must be in the same order as the images (first entry
    corresponds to _0000.nii.gz, etc). Example: ('T1', 'T2', 'FLAIR').
    :param labels: dict with int->str (key->value) mapping the label IDs to label names. Note that 0 is always
    supposed to be background! Example: {0: 'background', 1: 'edema', 2: 'enhancing tumor'}
    :param dataset_name: The name of the dataset. Can be anything you want
    :param license:
    :param dataset_description:
    :param dataset_reference: website of the dataset, if available
    :param dataset_release:
    :return:
    """
    train_identifiers = get_identifiers_from_splitted_files(imagesTr_dir)

    if imagesTs_dir is not None:
        test_identifiers = get_identifiers_from_splitted_files(imagesTs_dir)
    else:
        test_identifiers = []

    json_dict = {}
    json_dict['name'] = dataset_name
    json_dict['description'] = dataset_description
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = dataset_reference
    json_dict['licence'] = license
    json_dict['release'] = dataset_release
    json_dict['modality'] = {str(i): modalities[i] for i in range(len(modalities))}
    json_dict['labels'] = {str(i): labels[i] for i in labels.keys()}

    json_dict['numTraining'] = len(train_identifiers)
    json_dict['numTest'] = len(test_identifiers)
    json_dict['training'] = [
        {'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in train_identifiers]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in test_identifiers]

    output_file += "dataset.json"
    if not output_file.endswith("dataset.json"):
        print("WARNING: output file name is not dataset.json! This may be intentional or not. You decide. "
              "Proceeding anyways...")
    save_json(json_dict, os.path.join(output_file))


def SaveJason():
    print("------开除保存jason文件------")

    dataset_name = 'lung'
    dataset_description = "Task099_lung"
    dataset_reference = "Memorial Sloan Kettering Cancer Center"
    license = "CC-BY-SA 4.0"

    modalities = ["ray"]
    labels = {"0": "Background",
        "1": "scapula",
        "2": "vertebra",
        "3": "rib",
        "4": "lung",
        "5": "heart",
        "6": "trachea",
        "7": "clavicle"}


    imagesTr = os.path.join(outPath, 'train/imgnii')
    imagesTs = os.path.join(outPath, 'test_a/imgnii')
    generate_dataset_json(outPath, imagesTr, imagesTs, modalities, labels, dataset_name, license,
                          dataset_description, dataset_reference, dataset_release=' 25/08/2022')

if __name__ == '__main__':
    outPath = r'F:\2challenge\rule\datasets\chest'
    SaveJason()