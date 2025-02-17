from PIL import Image 
import numpy as np 
import os 
from torch.utils.data import Dataset
import config 
import ssl
import certifi

ssl._create_default_https_context = ssl._create_unverified_context

class MapDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        print(f"Looking for dataset in: {os.path.abspath(self.root_dir)}")
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Dataset path {self.root_dir} does not exist!")
            
        self.list_files = os.listdir(self.root_dir)
        print(f"Found {len(self.list_files)} files in {self.root_dir}")
    
    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        input_image = image[:, :600, :]
        target_image = image[:, 600:, :]

        augmentations = config.both_transform(image = input_image, image0= target_image)
        input_image, target_image = augmentations["image"], augmentations["image0"]

        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_input(image=target_image)["image"]

        return input_image, target_image
