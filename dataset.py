import glob
import os
import random

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class MyTrainDataSet(Dataset):
    def __init__(self,
                 type_name,
                 set_name,
                 shot_num=2,
                 expand=50,
                 resize=256,
                 imagesize=224):

        # 初始化参数
        self._set_name = set_name
        self._type_name = type_name
        self._shot_num = shot_num
        self._expand = expand
        self._resize = resize
        self._imagesize = imagesize

        # 路径配置
        self._key = {'MVTec-AD': 'mvtec', 'MPDD': 'mpdd', 'BTAD': 'btad', 'VisA': 'visa'}
        root_path = os.path.join('./data', set_name, self._key[set_name], type_name)
        self._image_paths = sorted(glob.glob(os.path.join(root_path, 'train', '*', '*.jpg')))

        # 预加载所有采样图像到内存
        self._sampled_paths = random.sample(self._image_paths, self._shot_num)
        self._sampled_images = [Image.open(path).convert('RGB') for path in self._sampled_paths]

        # 合并动态增强的Tensor转换和归一化
        self._dynamic_aug = transforms.Compose([
            transforms.Resize(resize),
            transforms.RandomAffine(
                degrees=(-45, 45),
                translate=(0.15, 0.15),
                scale=(0.85, 1.15),
                shear=(-15, 15)),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.2,
                hue=0.2),
            transforms.GaussianBlur(kernel_size=5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.2),
            transforms.RandomCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 静态预处理（带缓存）
        self._static_process = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return self._shot_num * self._expand

    def __getitem__(self, idx):
        base_idx = idx // self._expand
        aug_idx = idx % self._expand

        # 直接从内存读取图像
        image = self._sampled_images[base_idx]

        # 应用预处理
        if aug_idx == 0:
            processed_img = self._static_process(image)
        else:
            processed_img = self._dynamic_aug(image)

        return processed_img, torch.zeros(1, self._imagesize, self._imagesize)


class MyTestDataSet(Dataset):
    def __init__(self,
                 type_name,
                 set_name,
                 resize=256,
                 imagesize=224
                 ):
        self._type_name = type_name
        self._set_name = set_name
        self._resize = resize
        self._imagesize = imagesize

        # 路径配置
        self._key = {'MVTec-AD': 'mvtec', 'MPDD': 'mpdd', 'BTAD': 'btad', 'VisA': 'visa'}
        root_path = os.path.join('./data', set_name, self._key[set_name], type_name) # ./data/MVTec-AD/mvtec/bottle
        self._image_paths = sorted(glob.glob(os.path.join(root_path, 'test', '*', '*.jpg'))) # ./data/MVTec-AD/mvtec/bottle/test/*/*.jpg

        # 预加载所有测试数据到内存
        self._static_process = transforms.Compose([
            transforms.Resize(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self._gt_process = transforms.Compose([
            transforms.Resize(imagesize),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

        # 预加载所有数据和标签
        self.cache = []
        zeros_gt = torch.zeros(1, imagesize, imagesize)
        for path in self._image_paths:
            # 加载图像
            img = Image.open(path).convert('RGB')
            img_tensor = self._static_process(img)

            # 加载标签
            if 'good' in path:
                gt = zeros_gt
            else:
                gt_path = path.replace('test', 'ground_truth').replace('.jpg', '_mask.jpg')
                gt = Image.open(gt_path)
                gt = self._gt_process(gt)

            self.cache.append((
                img_tensor,
                gt,
                'good' in path
            ))

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, item):
        return self.cache[item]