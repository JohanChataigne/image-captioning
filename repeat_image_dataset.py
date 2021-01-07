import pandas as pd
import torch
from skimage import io
import os
from torch.utils.data import Dataset

ANNOTATIONS = './flickr8k/annotations/annotations_image_id.csv'

class RepeatImageDataset(Dataset):
    """Image captioning dataset"""
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file(string): captions file
            root_dir(string): images directory
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.df_captions = pd.read_csv(ANNOTATIONS, sep=';')
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        """Length is nb_captions / captions_per_image"""
        return len(self.df_captions) / 5
        
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
            
        im_name = self.df_captions.iloc[index, 0]
        im_path = os.path.join(self.root_dir, im_name)
        image = io.imread(im_path)
        
        caption = self.df_captions.iloc[index, 1]
        
        sample = {'image': image, 'caption': "<start>" + caption + "<stop>"}

        if self.transform:
            sample = self.transform(sample)

        return sample

    
if __name__ == "__main__":
    
    dataset = RepeatImageDataset('./flickr8k/images/train/')
    print(dataset[0])
    print(dataset[0]['image'].shape)