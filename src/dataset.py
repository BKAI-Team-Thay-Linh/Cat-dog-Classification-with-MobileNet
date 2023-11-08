import os 
import sys
sys.path.append(os.getcwd())

from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image

class DogCatDataset(Dataset):
    def __init__(self, type, root_dir):
        self.root_dir = root_dir
        self.type = type
        self.transform = self._get_transforms()
        
        # Prepare list of data
        cats_images_dir = os.path.join(self.root_dir, 'train', 'train', 'cats')
        dogs_images_dir = os.path.join(self.root_dir, 'train', 'train', 'dogs')
        
        # All images
        self.all_images_path = [os.path.join(cats_images_dir, img_name) for img_name in os.listdir(cats_images_dir)] + \
                                 [os.path.join(dogs_images_dir, img_name) for img_name in os.listdir(dogs_images_dir)]
    
    def __len__(self):
        return len(os.listdir(os.path.join(self.root_dir)))

    def __getitem__(self, idx):
        
        image_path = self.all_images_path[idx]
        
        image_name = os.path.basename(image_path)
        
        image = Image.open(image_path, mode="r")
        
        image = self.transform(image)
        
        # Get the label 
        label = 0 if image_name.split('.')[0] == 'cat' else 1
        
        return image, label

    def _get_transforms(self):
        if self.type == 'train':
            return T.Compose([
                T.Resize((112, 112)),
                T.RandomChoice([T.RandomRotation(degrees=15)], p=[0.3]),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor()
            ])
        else:
            return T.Compose([
                T.ToTensor()
            ])

        
            
            