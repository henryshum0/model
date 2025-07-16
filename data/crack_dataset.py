from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import cv2
import numpy as np
import data.data_generation as D
import torch

class CrackDataset(Dataset):
    def __init__(self, img_path:str=None, mask_path:str=None, transforms=[D.Resize((448,448))]
                 , mask_suffix:str=''):
        
        # checking if image and mask paths are valid
        self.img_path = Path(img_path)
        self.mask_path = Path(mask_path)
        if not self.img_path.exists():
            raise FileNotFoundError(f"Image path {self.img_path} does not exist.")
        if not self.mask_path.exists():
            raise FileNotFoundError(f"Mask path {self.mask_path} does not exist.")
        
        # getting all image and mask files
        exts = ['.jpg', '.jpeg', '.png']
        self.img_files = sorted([f for f in self.img_path.rglob("*") if f.suffix.lower() in exts and f.stem.isdigit()])
        self.mask_files = sorted([f for f in self.mask_path.rglob("*") if f.suffix.lower() in exts and f.stem.isdigit()])
        if len(self.img_files) == 0:
            raise FileExistsError(f"No image files found in {self.img_path}.")
        if len(self.mask_files) == 0:
            raise FileExistsError(f"No mask files found in {self.mask_path}.")
        if len(self.img_files) != len(self.mask_files):
            raise FileExistsError(f"Number of images ({len(self.img_files)}) does not match number of masks ({len(self.mask_files)}).")
        
        self.data = list(zip(self.img_files, self.mask_files))
        self.data = sorted(self.data, key=lambda x: int(x[0].stem))
        self.transform_pipeline = D.DataGenPipeline(save=False, load=False, transforms=transforms)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #assume input image are in RGB[0, 255]
        #input masks are in Binary[0, 255]
        img_file, mask_file = self.data[idx]
        image = cv2.imread(str(img_file), cv2.IMREAD_COLOR_RGB)
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)

        if self.transform_pipeline:
            image, mask = self.transform_pipeline(image, mask)
            image = image.astype(np.float32).transpose((2, 0, 1)) / 255
            mask = mask.astype(np.float32) / 255
            if mask.ndim == 2:  # Ensure mask is 2D
                mask = np.expand_dims(mask, axis=0)  # shape (1, H, W)
        return {
            'image': torch.from_numpy(image),
            'mask': torch.from_numpy(mask)
        }
    
if __name__ == "__main__":
    
    #run the formatting.py first
    
    img_path = "data_proccess/train_n_test/crop_img_tr"
    mask_path = "data_proccess/train_n_test/crop_msk_tr"
    transforms_list = [
        
    ]
    dataset = CrackDataset(img_path=img_path, mask_path=mask_path, transforms=transforms_list)
    print(f"Dataset size: {len(dataset)}")
    for i in range(5):
        img, mask = dataset[i]['image'], dataset[i]['mask']
        print(f"Image {i} shape: {img.shape}, Mask {i} shape: {mask.shape}")
        # Optionally, you can visualize the images and masks using matplotlib or any other library.
        # For example:
        # print(np.unique(img.numpy()))
        # cv2.imshow("Image", img.numpy())
        # cv2.imshow("Mask", np.stack([mask.numpy()]*3, axis=-1) ) # Convert mask to 3 channels for visualization
        # cv2.waitKey(0)
        # Note: The above visualization code is commented out to avoid unnecessary imports.
        # Uncomment it if you want to visualize the images and masks.  