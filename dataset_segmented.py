#!/usr/bin/env python3
"""Loads image datasets from folders of images."""
from PIL import Image
from os import listdir
from os.path import join
import torch.utils.data as data
import numpy as np
import random
from scipy.misc import imread, imresize
import torch
from pycocotools.coco import COCO
import os
from torchvision.transforms import Compose, CenterCrop, ToTensor

def is_image_file(filename, extensions):
    return any(filename.lower().endswith(ext) for ext in extensions)

def load_img(filepath):
    return Image.open(filepath).convert('RGB')

def _scale_and_crop(img, seg_masks, cropSize, is_train):
    h, w = np.size(img,0), np.size(img,1)
    if is_train:
        # random scale
        scale = random.random() + 0.2     # 0.5-1.5
        scale = max(scale, 1. * cropSize / (min(h, w) - 1))
    else:
        # scale to crop size
        scale = 1. * cropSize / (min(h, w) - 1)

    img = imresize(img, scale, interp='bicubic')
    seg = [imresize(seg, scale, interp='nearest') for seg in seg_masks]

    h_s, w_s = img.shape[0], img.shape[1]
    x1 = random.randint(0, w_s - cropSize)
    y1 = random.randint(0, h_s - cropSize)

    img_crop = img[y1: y1 + cropSize, x1: x1 + cropSize, :]
    seg_crop = [seg[y1: y1 + cropSize, x1: x1 + cropSize] for seg in seg_masks]
    return img_crop, seg_crop

def return_segMasks(anns, coco):
    '''Return 6 segmentaion masks sorted on the basis of their area. The 7th mask belongs to others class'''
    _masks = [int (m['area']) for m in anns]
    # Sorting Top 6 masks on the basis of area
    temp = np.array(_masks)
    idx = (-temp).argsort()[:6]
    seg_masks =[]
    if len(_masks) == 0:
        for i in range(7):
            seg_masks.append(np.full((500,500), 255).astype(np.uint8))        
        return seg_masks
    else:
        others_class = np.full(coco.annToMask(anns[0]).shape, 255)
        for i in range(len(idx)):
            mask = coco.annToMask(anns[idx[i]])*255
            others_class -=mask
            seg_masks.append(mask.astype(np.uint8))
        seg_masks.append(others_class.astype(np.uint8))
        # Avoid sending inconsistent seg lengths
        if len(seg_masks)!=7:
            #print (len(seg_masks))
            for i in range(7-len(seg_masks)):
                seg_masks.append(others_class.astype(np.uint8))
        return seg_masks

def target_transform(crop_size):
    return Compose([CenterCrop(crop_size), ToTensor()])

class SISR_Dataset_Segment(data.Dataset):
    def __init__(self, path, cropsize, upscale, annotationFile,
                 image_extensions=(".png", ".jpg", ".jpeg"), is_train = False, bicubic = False):
        super(SISR_Dataset_Segment, self).__init__()
        self.is_train = is_train
        self.cropsize = cropsize
        self.upscale = upscale
        self.bicubic = bicubic
        self.root = path
        self.annFile = annotationFile #'/home/wgondal/coco_stuff/stuff_train2017.json'
        self.coco = COCO(self.annFile)
        self.imgIds = self.coco.getImgIds()
        self.target_transform = target_transform(cropsize)      

    def __getitem__(self, index):
        img_id = self.imgIds[index]
        annIds = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(annIds)
        seg_masks = return_segMasks(anns, self.coco)
        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = load_img(os.path.join(self.root, path))

        img, seg_masks = _scale_and_crop(img, seg_masks, self.cropsize, is_train = True)
        target_masks = []
        target = img.copy()
        data_in = imresize(img, (self.cropsize//self.upscale,self.cropsize//self.upscale), interp='bicubic')
        if self.bicubic:
            bicubic = imresize(data_in, (self.cropsize,self.cropsize), interp='bicubic')               

        data_in = data_in.astype(np.float32) / 255.
        data_in = data_in.transpose((2, 0, 1))
        if self.bicubic:
            bicubic_in = bicubic.astype(np.float32) / 255.
            bicubic_in = bicubic_in.transpose((2, 0, 1))            

        target = target.astype(np.float32) / 255.
        target = target.transpose((2, 0, 1))

        seg_masks = [self.target_transform(Image.fromarray(mask)) for mask in seg_masks]

        data_in = torch.from_numpy(data_in)
        target = torch.from_numpy(target)
        if self.bicubic:
            bicubic_in = torch.from_numpy(bicubic_in)
            return data_in, target, seg_masks, bicubic_in            
        else:
            return data_in, target, seg_masks     

    def __len__(self):
        return len(self.imgIds)