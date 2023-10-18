import os 
import sys
sys.path.append(os.getcwd())

from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image

class DogCatDataset(Dataset):
    def __init__(self, root_dir, type = 'train'):
        self.root_dir = root_dir
        self.type = type
        self.transform = self._get_transforms()
    
    def __len__(self):
        return len(os.listdir(os.path.join(self.root_dir, self.type)))

    def __getitem__(self, idx):
        img_name = os.listdir(os.path.join(self.root_dir, self.type))[idx]
        
        img_path = os.path.join(self.root_dir, self.type, img_name)
        
        image = Image.open(img_path)
        
        image = self.transform(image)
        
        # Get the label 
        label = 0 if img_name.split('.')[0] == 'cat' else 1
        
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

        
            
            