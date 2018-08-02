"""Loads image datasets from folders of images."""
from PIL import Image
from os import listdir
from os.path import join
import torch.utils.data as data
import numpy as np
import random
from scipy.misc import imread, imresize
import torch
from torchvision.transforms import Compose, ToTensor

def is_image_file(filename, extensions):
    return any(filename.lower().endswith(ext) for ext in extensions)

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

def centeredCrop(img, new_height = 128, new_width= 128):
    width =  np.size(img,1)
    height =  np.size(img,0)
    left = ((width - new_width)//2)
    top = ((height - new_height)//2)
    right = ((width + new_width)//2)
    bottom =((height + new_height)//2)
    #cImg = img[top: bottom, left: right, :]
    cImg = img.crop((left, top, right, bottom))
    return cImg

def _scale_and_crop(img, cropSize, is_train = True):
    #h, w = img.shape[0], img.shape[1]
    h, w = np.size(img,0), np.size(img,1)
    if is_train:
        # random scale
        scale = random.random() + 0.5     # 0.5-1.5
        scale = max(scale, 1. * cropSize / (min(h, w) - 1))
    else:
        # scale to crop size
        scale = 1. * cropSize / (min(h, w) - 1)

    img_scale = imresize(img, scale, interp='bicubic')
    h_s, w_s = img_scale.shape[0], img_scale.shape[1]
    if is_train:
        # random crop
        x1 = random.randint(0, w_s - cropSize)
        y1 = random.randint(0, h_s - cropSize)
    else:
        # center crop
        x1 = (w_s - cropSize) // 2
        y1 = (h_s - cropSize) // 2
    img_crop = img_scale[y1: y1 + cropSize, x1: x1 + cropSize, :]
    return img_crop

def target_transform():
    return Compose([ToTensor()])

class SISR_Dataset(data.Dataset):
    def __init__(self, path, cropsize, upscale,
                 image_extensions=(".png", ".jpg", ".jpeg"), bicubic = False):
        super(SISR_Dataset, self).__init__()

        self.cropsize = cropsize
        self.upscale = upscale
        self.bicubic = bicubic
        self.image_filenames = [join(path, fn)
                                for fn in listdir(path)
                                if is_image_file(fn, image_extensions)]

        self.target_transform = target_transform()

    def _flip(self, img):
        img_flip = img[:, ::-1, :]
        return img_flip.copy()       

    def __getitem__(self, index):
        data_in = load_img(self.image_filenames[index])

        data_in = _scale_and_crop(data_in, self.cropsize)
        target = data_in.copy()
        data_in = imresize(data_in, (self.cropsize//self.upscale,self.cropsize//self.upscale), interp='bicubic')
        if self.bicubic:
            bicubic = imresize(data_in, (self.cropsize,self.cropsize), interp='bicubic')
        if random.choice([-1, 1]) > 0:
            data_in = self._flip(data_in)
            target = self._flip(target) 
            if self.bicubic:
                bicubic = self._flip(bicubic)                  
        data_input = self.target_transform(data_in)
        data_target = self.target_transform(target)
        if self.bicubic:
            bicubic = self.target_transform(bicubic)
            return data_input, data_target, bicubic 
        else:
            return data_input, data_target

    def __len__(self):
        return len(self.image_filenames)
