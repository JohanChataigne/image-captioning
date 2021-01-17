from skimage import transform
import numpy as np
import torch
from torchvision import transforms

class Rescale(object):
    """Rescale the image in a sample to a given size. Usefull to have all samples of same shape in input of a CNN

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, caption, im_path = sample['image'], sample['caption'], sample['im_path']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'caption': caption, 'im_path': im_path}

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, caption, im_path = sample['image'], sample['caption'], sample['im_path']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]


        return {'image': image,
                'caption': caption,
                'im_path': im_path}
    
class ToTensor(object):
    """Convert ndarrays in a sample to Tensors."""

    def __call__(self, sample):
        image, caption, im_path = sample['image'], sample['caption'], sample['im_path']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'caption': caption,
                'im_path': im_path}
    
class Normalize(object):
    """Normalize an image. Image need to be a tensor"""
    
    def __call__(self, sample):
        image, caption, im_path = sample['image'], sample['caption'], sample['im_path']
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)
        
        return {'image': image,
                'caption': caption,
                'im_path': im_path}
    
class OneHotEncode(object):
    """One Hot Encoding the caption of a sample given the preprocessor object"""
    
    
    def __init__(self, text_preprocessor):
        
        self.preprocessor = text_preprocessor
        
    def __call__(self, sample):
        
        image, caption, im_path = sample['image'], sample['caption'], sample['im_path']
        
        caption = self.preprocessor.caption_to_vect(caption)
        
        return {'image': image,
                'caption': caption,
                'im_path': im_path}
    
class PadSentence(object):
    """Add padding to the caption of a sample given the preprocessor object"""
    
    def __init__(self, text_preprocessor):
        
        self.preprocessor = text_preprocessor
        
    def __call__(self, sample):
        
        image, caption, im_path = sample['image'], sample['caption'], sample['im_path']
        
        caption = self.preprocessor.pad_sentence(caption)
        
        return {'image': image,
                'caption': caption,
                'im_path': im_path}
    
class AddDelimiters(object):
    """Add <start> and <stop> to the caption of a sample given the preprocessor object"""
        
    def __call__(self, sample):
        
        image, caption, im_path = sample['image'], sample['caption'], sample['im_path']
        
        caption = '<start> ' + caption + ' <stop>'
        
        return {'image': image,
                'caption': caption,
                'im_path': im_path}
    
    

class RemoveUnpopulars(object):
    """Remove in a string caption all the words not matching the occurences criteria"""
        
    def __init__(self, text_preprocessor, n):
        
        assert (n > 0)
        
        self.text_preprocessor = text_preprocessor
        self.n = n
        
        
    def __call__(self, sample):
        
        image, caption, im_path = sample['image'], sample['caption'], sample['im_path']
        
        caption = self.text_preprocessor.remove_unpopulars(caption, self.n)
        
        return {'image': image,
                'caption': caption,
                'im_path': im_path}
    
        
        
        
        
