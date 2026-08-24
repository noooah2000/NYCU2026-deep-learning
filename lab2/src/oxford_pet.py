import os
from PIL import Image
import torch
import random
import numpy as np
from torchvision.datasets.utils import download_url, extract_archive

class OxfordPetDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split_dir, mode="train", model_name="unet", transform=None):
        assert mode in {"train", "valid", "test"}
        self.data_dir = data_dir
        self.split_dir = split_dir 
        self.mode = mode
        self.model_name=model_name
        self.transform = transform
        self.images_directory = os.path.join(self.data_dir, "images")
        self.masks_directory = os.path.join(self.data_dir, "annotations", "trimaps")
        self._download()
        self.filenames = self._read_split()

    def __len__(self):
        return len(self.filenames)

    def _read_split(self):
        match (self.mode, self.model_name):
            case ("train", _): 
                txt_name = "train.txt"
            case ("valid", _):
                txt_name = "val.txt"
            case ("test", "unet"):
                txt_name = "test_unet.txt"
            case ("test", "resnet"):
                txt_name = "test_res_unet.txt"
            case _:
                raise ValueError(f"Wrong parameters! mode={self.mode}, model_name={self.model_name}")
            
        split_file = os.path.join(self.split_dir, txt_name)
        filenames = []

        with open(split_file, "r") as f:
            for line in f:
                filename = line.strip() 
                if filename: filenames.append(filename)
        
        return filenames
    
    def __getitem__(self, index):
        filename = self.filenames[index]
        image_path = os.path.join(self.images_directory, f"{filename}.jpg")
        image = np.array(Image.open(image_path).convert("RGB"))
        
        ori_height, ori_weight = image.shape[:2]

        if self.mode == "test":
            if self.transform is not None:
                transformed = self.transform(image=image)
                image = transformed["image"]
            return image, filename, ori_height, ori_weight
        else:
            mask_path = os.path.join(self.masks_directory,  f"{filename}.png")
            trimap = Image.open(mask_path)
            mask = (np.array(trimap) == 1).astype(np.float32)

            if self.transform is not None:
                transformed = self.transform(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]
                mask = mask.unsqueeze(0)
            return image, mask

    def _download(self):
        os.makedirs(self.data_dir, exist_ok=True)

        self._download_data(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            zip_file="images.tar.gz",
            check_dir=self.images_directory
        )
        self._download_data(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            zip_file="annotations.tar.gz",
            check_dir=os.path.join(self.data_dir, "annotations")
        )
    
    def _download_data(self, url, zip_file, check_dir):
        if os.path.exists(check_dir):
            print(f"The file is already exist")
            return
        
        zipfile_path = os.path.join(self.data_dir, zip_file)
        if not os.path.exists(zipfile_path):
            print(f"Downing {zip_file} ...")
            download_url(url=url, root=self.data_dir, filename=zip_file)
        print(f"Unzipping {zip_file} ...")
        extract_archive(from_path=zipfile_path, to_path=self.data_dir)
        print(f"Successful Unzipping {zip_file}")
        