import os
import torch
import numpy as np
import albumentations as A
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FaceDataLoader(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.path_array = []

        for files in os.listdir(self.path):
            if files.endswith(".txt"):
                txt_path = os.path.join(self.path, files)
                with open(txt_path) as f:
                    content = f.readlines()

                    for line in content:
                        line = line.strip().split()
                        if len(line) > 1:   
                            img_path = os.path.join(self.path, line[0].replace("\\","/"))
                            landmark = list(map(float, line[1:]))
                            self.path_array.append((img_path, landmark))

        # Albumentations transform to resize image and landmarks
        self.albumentations_transform = A.Compose([
            A.LongestMaxSize(max_size=224),
            A.PadIfNeeded(min_height=224, min_width=224, pad_mode='constant', pad_cval=0),
        ], keypoint_params=A.KeypointParams(format='xy'))

        
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.path_array)

    def __getitem__(self, idx):
        image_path, landmark = self.path_array[idx]
        image = Image.open(image_path).convert("RGB")
        landmark = np.array(landmark).reshape(-1, 2)  # reshape landmarks to Nx2

        image_np = np.array(image)

        augmented = self.albumentations_transform(image=image_np, keypoints=landmark)
        resized_image = augmented['image']
        resized_landmarks = np.array(augmented['keypoints'])


        # Flatten landmarks back to 1D tensor
        landmarks_tensor = torch.tensor(resized_landmarks, dtype=torch.float32).view(-1)

        image_pil = Image.fromarray(resized_image)

       
        if self.transform:
            image_tensor = self.transform(image_pil)
        else:
            image_tensor = transforms.ToTensor()(image_pil)


        return image_tensor, landmarks_tensor
