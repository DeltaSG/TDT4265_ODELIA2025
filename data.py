import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import torch
from pathlib import Path

from monai.transforms import (
    Compose,
    RandScaleIntensity,
    RandAdjustContrast,
    RandGaussianNoise,
    RandBiasField,
    RandAffine,
    RandFlip,
    NormalizeIntensity,
    ScaleIntensityRangePercentiles
)

# ALL data augmentations tried
train_transforms = Compose([
    # ScaleIntensityRangePercentiles(0.5,99.5,0,1,channel_wise=True),
    # RandBiasField(coeff_range=(0.0,0.2),prob=0.4),
    # RandScaleIntensity(factors=0.15,prob=0.5),
    # RandAdjustContrast(gamma=(0.8,1.2),prob=0.5),
    # RandGaussianNoise(mean=0.0,std=0.03,prob=0.3),
    # RandAffine(prob=0.5,rotate_range=(0.1,0.1,0.1),scale_range=(0.1,0.1,0.1),translate_range=(10,10,5),padding_mode="border"),
    # RandFlip(spatial_axis=0,prob=0.5),
    NormalizeIntensity(nonzero=False,channel_wise = True)
])

test_transforms = Compose([NormalizeIntensity(nonzero=False,channel_wise = True)])

#Creates a dictionary for all datapoints including paths to all 3D volumes, label, uid and which center the datapoint belong to.
def collect_data(centers = ["CAM","MHA","RUMC","UKA"]):

    data_list = []
    
    for center in centers:
        data_path = "/cluster/projects/vc/courses/TDT17/mic/ODELIA2025/data/"
        annotation_path = data_path + center + "/metadata_unilateral/annotation.csv"
        annotation = pd.read_csv(annotation_path)

        for _,row in annotation.iterrows():
            uid = row["UID"]
            label = int(row["Lesion"])
            data_dir = data_path + center + "/data_unilateral/" + uid

            data_list.append({"image":[
                data_dir + "/Post_1.nii.gz",
                data_dir + "/Post_2.nii.gz",
                data_dir + "/Pre.nii.gz",
                data_dir + "/Sub_1.nii.gz",
                data_dir + "/T2.nii.gz"
            ],"label":label,"uid":uid,"center":center})

    return data_list

# Class created in order for dataloader to create batches and feed them to the neural networks.
class MRIdataset():
    def __init__(self,data_list,validation = False):
        self.data_list = data_list
        self.validation = validation

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self,idx):
        sample = self.data_list[idx]

        images = []
        for path in sample["image"]:
            img = nib.load(path).get_fdata()
            images.append(img)

        image = np.stack(images, axis = 0)

        image = torch.tensor(image,dtype=torch.float32)
        label = torch.tensor(sample["label"],dtype=torch.long)
        if self.validation == False:
            image = train_transforms(image)
        else:
            image = test_transforms(image)

        return image, label

# Creates dictionaries for each datapoint in test dataset, containing paths to all images and uid.
def collect_test_data():

    data_list = []
    folder = Path("/cluster/projects/vc/courses/TDT17/mic/ODELIA2025/data/RSH/data_unilateral")
    for file in folder.iterdir():
        data_list.append({"image":[
            str(file) + "/Post_1.nii.gz",
            str(file) + "/Post_2.nii.gz",
            str(file) + "/Pre.nii.gz",
            str(file) + "/Sub_1.nii.gz",
            str(file) + "/T2.nii.gz"
        ],"uid":file.name})
    return data_list

# Splits the training dataset into a smaller training dataset and a validation dataset.
def make_validation_set(data_set):
    length = len(data_set)
    data_set = np.array(data_set)
    rng = np.random.default_rng(seed=42)
    rng.shuffle(data_set)
    split_idx = int((length-1) * 0.18)
    validation_set = data_set[:split_idx]
    training_set = data_set[split_idx:]
    return training_set , validation_set





