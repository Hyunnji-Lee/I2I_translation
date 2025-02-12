from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import numpy as np

class ImagePathDataset(Dataset):
    def __init__(self, image_paths, image_size=(256, 256), max_pixel=255.0, flip=False, to_normal=False):
        self.image_size = image_size
        self.image_paths = image_paths
        self._length = len(image_paths)
        self.max_pixel = float(max_pixel)
        self.flip = flip
        self.to_normal = to_normal

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = 0.0
        
        if index >= self._length:
            index = index - self._length
            p = 1.0

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=p),
            transforms.Resize(self.image_size)
        ])
        
        img_path = self.image_paths[index]
        image = None
        
        image_name = Path(img_path).stem

        try:
            np_image = np.load(img_path, allow_pickle=True)
            
            if image_name.startswith('cbct'):
                # cbct norm -> [0,1]
                if image_name.startswith('cbct_2BA'):
                    cbct_max, cbct_min = 3000.0, 0
                elif image_name.startswith('cbct_2BB'):
                    cbct_max, cbct_min = 2000.0, -1000.0
                elif image_name.startswith('cbct_2BC'):
                    cbct_max, cbct_min = 3000.0, -1024.0
                np_image =  (np_image - cbct_min) / (cbct_max - cbct_min)
                
            elif image_name.startswith('ct'):
                # ct norm -> [0,1]
                np_image = (np_image - (-1024)) / (3000 - (-1024))
            else: 
                raise ValueError("Unsupported image type.")
                            
#             image = Image.fromarray(np_image) 
            
#             if not image.mode == 'RGB':
#                 image = image.convert('RGB')

            np_image = np_image.astype(np.float32)

            if np_image.ndim == 2:
                np_image = np.stack([np_image]*3, axis=-1)
            
            image = transform(np_image) 

        except BaseException as e:
            print(img_path)    
            print(e)
        
        if self.to_normal: # [0,1] -> [-1,1]
            image = (image - 0.5) * 2.

        return image, image_name
      