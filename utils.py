import enum
import glob
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict
from typing import List

import cv2
import imgaug.augmenters as iaa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from PIL import Image
from kornia.filters import gaussian_blur2d
from scipy.ndimage import distance_transform_edt, label, center_of_mass
from scipy.ndimage import gaussian_filter, map_coordinates
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

def create_path_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path

def configure_optimizer(model, is_phase1=True, phase1_lr=None, phase2_lr=5e-5, lr_scheduler_type='step',
                        step_size=5, gamma=0.1, T_max=10, eta_min=1e-6):
    """
    根据不同阶段配置优化器和学习率调度器
    :param model: 主模型实例
    :param is_phase1: 是否处于第一阶段，默认为 True
    :param phase1_lr: 第一阶段第一个子模型的学习率，默认为 None
    :param phase2_lr: 第二阶段第一个子模型的学习率，默认为 1e-3
    :param lr_scheduler_type: 学习率调度器类型，可选 'step' 或 'cosine'，默认为 'step'
    :param step_size: StepLR 调度器的步长，默认为 5
    :param gamma: StepLR 调度器的衰减系数，默认为 0.1
    :param T_max: CosineAnnealingLR 调度器的总周期数，默认为 10
    :param eta_min: CosineAnnealingLR 调度器的最小学习率，默认为 1e-6
    :return: 优化器和学习率调度器
    """
    if is_phase1:
        if phase1_lr is None:
            optimizer = optim.Adam([
                {'params': model.Extractor.parameters(), 'lr': 0.},
                {'params': model.Adapter.parameters(), 'lr': 1e-3},
                {'params': model.DecModel.parameters(), 'lr': 3e-4},
                {'params': model.Discriminator.parameters(), 'lr': 5e-4}
            ])
        else:
            optimizer = optim.Adam([
                {'params': model.Extractor.parameters(), 'lr': phase1_lr},
                {'params': model.Adapter.parameters(), 'lr': 1e-3},
                {'params': model.DecModeldel.parameters(), 'lr': 3e-4},
                {'params': model.Discriminator.parameters(), 'lr': 5e-4}
            ])
    else:
        optimizer = optim.Adam([
            {'params': model.Extractor.parameters(), 'lr': phase2_lr},
            {'params': model.Adapter.parameters(), 'lr': 2e-4},
            {'params': model.DecModel.parameters(), 'lr': 1e-4},
            {'params': model.Discriminator.parameters(), 'lr': 1e-4}
        ])

    if lr_scheduler_type == 'step':
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif lr_scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    else:
        raise ValueError("lr_scheduler_type must be 'step' or 'cosine'")

    return optimizer, scheduler

def log_metrics_to_tensorboard(log_dir, kind, epoch, batch_idx, num_batches, loss=None, accuracy=None):
    writer = SummaryWriter(log_dir=f"{log_dir}/{kind}")
    global_step = epoch * num_batches + batch_idx

    if loss is not None:
        writer.add_scalar('Loss/train', loss, global_step)

    if accuracy is not None:
        writer.add_scalar('Accuracy/train', accuracy, global_step)

    writer.close()

def set_random_seed(seed:int =666):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def util_load_yaml_config(path: str) -> Dict:
    """
    Load a YAML configuration file.

    Args:
        path (str): The path to the YAML file.

    Returns:
        Dict: A dictionary containing the parsed YAML configuration.

    Raises:
        TypeError: If the provided path is not a string.
        FileNotFoundError: If the specified YAML file does not exist.
        yaml.YAMLError: If there is an issue parsing the YAML file.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Check if the path is a string
    if not isinstance(path, str):
        error_msg = f"The 'path' argument must be a string, but got {type(path).__name__}."
        logging.error(error_msg)
        raise TypeError(error_msg)

    try:
        # Try to open and load the YAML file
        with open(path, 'r') as file:
            config = yaml.safe_load(file)
        logging.info(f"Successfully loaded YAML configuration from {path}.")
        return config
    except FileNotFoundError:
        error_msg = f"The file at path {path} was not found."
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)
    except yaml.YAMLError as e:
        error_msg = f"Error parsing the YAML file at {path}: {e}"
        logging.error(error_msg)
        raise yaml.YAMLError(error_msg)

def get_config_paras(config_path: str, model_path: str):
    """
    Load configuration parameters from YAML files and create a Config dataclass.

    Args:
        config_path (str): Path to the training configuration YAML file.
        model_path (str): Path to the model configuration YAML file. Currently not used.

    Returns:
        Config: A dataclass instance containing the configuration parameters.
    """
    try:
        # 加载训练配置文件
        train_config = util_load_yaml_config(config_path)
        # 加载模型配置文件
        model_config = util_load_yaml_config(model_path)

        @dataclass(frozen=True)
        class Config:
            random_seed: int = 666
            optim: str = train_config.get('optim')
            batch_size: int = train_config.get('batch_size')
            momentum: float = train_config.get('momentum')
            learning_rate: float = train_config.get('learning_rate')
            epochs: int = train_config.get('epochs')
            model_input_ch: int = model_config.get('model_input_ch')
            patch_size: int = train_config.get('patch_size')
            patch_stride: int = train_config.get('patch_stride')
            model_input_max_height: int = model_config.get('model_input_max_height')
            model_input_max_weight: int = model_config.get('model_input_max_weight')
            mvtec: List = train_config.get('MvTec')
            mpdd: List = train_config.get('MPDD')
            input_resolution: int = model_config.get('input_resolution')
            shot_num: int = model_config.get('shot_num')
            num: int = train_config.get('num')

        # 创建 Config 类的实例
        config = Config()
        return config
    except FileNotFoundError:
        print(f"Error: The file at {config_path} was not found.")
    except yaml.YAMLError:
        print(f"Error: Failed to parse the YAML file at {config_path}.")
    except KeyError as e:
        print(f"Error: The key {e} was not found in the configuration file.")
    return None

def set_not_grad(module: nn.Module) -> None:
    """
    Disable gradient computation for all parameters in the given PyTorch module.

    Args:
        module (nn.Module): A PyTorch module whose parameters' gradients are to be disabled.
    """
    if not isinstance(module, nn.Module):
        raise TypeError(f"Expected nn.Module, but got {type(module).__name__}.")
    for param in module.parameters():
        param.requires_grad = False

def init_weight(module: nn.Module) -> None:
    """
    Initialize the weights of linear and 2D convolutional layers in the given PyTorch module
    using Xavier normal initialization.

    Args:
        module (nn.Module): A PyTorch module whose linear and conv2d layers' weights are to be initialized.
    """
    if not isinstance(module, nn.Module):
        raise TypeError(f"Expected nn.Module, but got {type(module).__name__}.")
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def save_pytorch_model(adapter, dis, class_name, i_auc, p_auc, aupro, epoch,
                       data_set_name = 'MVTec-AD', save_base_dir='experiments_result/best_model', save_mode='state_dict'):

    base_dir = os.path.join('./', save_base_dir, data_set_name) # './experiments_result/best_model/MvTec'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)

    class_dir = os.path.join(base_dir, class_name) # './experiments_result/best_model/MvTec/bottle'
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    rate_str1 = "{:.4f}".format(i_auc)
    rate_str2 = "{:.4f}".format(p_auc)
    rate_str3 = "{:.4f}".format(aupro)
    file_name = f"{data_set_name}_{class_name}_{epoch}_best_i-auroc_{rate_str1}_p-auroc_{rate_str2}_aupro_{rate_str3}.pth"
    # './experiments_result/best_model/MvTec/bottle/MvTec_bottle_100_best_0.95_stage1.pth'
    save_path = os.path.join(class_dir, file_name)

    if save_mode == 'state_dict':
        torch.save({
            'Adapter': adapter.state_dict(),
            'Discriminator': dis.state_dict()
        }, save_path)
    elif save_mode == 'whole_model':
        torch.save({
            'Adapter': adapter,
            'Discriminator': dis
        }, save_path)
    else:
        raise ValueError(f"Unsupported save mode {save_mode}. Please choose 'state_dict' or 'whole_model'.")

def patch_aggregate(feature_list: List[torch.Tensor]) -> List[torch.Tensor]:
    output_list = []
    # 创建一个 3x3 的平均卷积核
    kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32) / 9
    for tensor in feature_list:
        # 对输入张量进行填充，边缘补零
        padded_tensor = F.pad(tensor, (1, 1, 1, 1), mode='constant', value=0)
        B, C, _, _ = tensor.shape
        # 初始化输出张量
        output = torch.zeros_like(tensor)
        for c in range(C):
            # 对每个通道进行卷积操作
            output[:, c:c+1, :, :] = F.conv2d(padded_tensor[:, c:c+1, :, :], kernel, stride=1)
        output_list.append(output)
    return output_list

def concat_features(feature_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Upsample all features in the feature list to the maximum size among them and concatenate all features.
        It is assumed that the width and height of the input features are equal.

        Args:
            feature_list (List[torch.Tensor]): A list containing multiple feature tensors.

        Returns:
            torch.Tensor: The concatenated feature tensor.
        """
        # Check if the input list is empty
        if not feature_list:
            raise ValueError("The input feature list is empty.")

        # Find the maximum size
        max_size = max(feature.shape[-1] for feature in feature_list)
        max_size_index = next((i for i, feature in enumerate(feature_list) if feature.shape[-1] == max_size), None)

        # Upsample features
        upsampled_features = []
        for i, feature in enumerate(feature_list):
            if i != max_size_index:
                upsampled_feature = F.interpolate(feature, size=max_size, mode='bilinear', align_corners=False)
                upsampled_features.append(upsampled_feature)
            else:
                upsampled_features.append(feature)

        # Concatenate features
        try:
            concatenated_feature = torch.cat(upsampled_features, dim=-3)
        except RuntimeError as e:
            raise ValueError(f"Feature concatenation failed: {e}")

        return concatenated_feature

def lerp_np(x,y,w):
    fin_out = (y-x)*w + x
    return fin_out

def rand_perlin_2d_np(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    tt = np.repeat(np.repeat(gradients,d[0],axis=0),d[1],axis=1)

    tile_grads = lambda slice1, slice2: np.repeat(np.repeat(gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]],d[0],axis=0),d[1],axis=1)
    dot = lambda grad, shift: (
                np.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                            axis=-1) * grad[:shape[0], :shape[1]]).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * lerp_np(lerp_np(n00, n10, t[..., 0]), lerp_np(n01, n11, t[..., 0]), t[..., 1])

class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )

        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x

class Status(enum.Enum):
    NORMAL = enum.auto()
    PSEUDO_ABNORMAL = enum.auto()
    FEATURE_ABNORMAL = enum.auto()

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss, model, epoch, class_name, data_set_name):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, class_name, data_set_name)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, class_name, data_set_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch, class_name, data_set_name):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model,
                   os.path.join('./experiments_result/early_stop', data_set_name, class_name, f'/{epoch}_{val_loss}.pt'))
        self.val_loss_min = val_loss

def rand_augment_abnormal() -> iaa.Sequential:
        augmenters = [
            iaa.GammaContrast((0.5, 2.0), per_channel=True),
            iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
            iaa.pillike.EnhanceSharpness(),
            iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
            iaa.Solarize(0.5, threshold=(32, 128)),
            iaa.Posterize(),
            iaa.Invert(),
            iaa.pillike.Autocontrast(),
            iaa.pillike.Equalize(),
            iaa.Affine(rotate=(-45, 45))
        ]
        aug_idx = np.random.choice(np.arange(len(augmenters)), 3, replace=False)
        aug = iaa.Sequential([
            augmenters[aug_idx[0]],
            augmenters[aug_idx[1]],
            augmenters[aug_idx[2]]
        ])
        return aug

def read_and_convert_image(image_path: str, resize: int):
    """
    Read an image and convert it from BGR to RGB.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: Converted image array, or None if reading fails.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            logging.warning(f"Failed to read image: {image_path}")
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (resize, resize))
        return image
    except Exception as e:
        logging.error(f"Error reading image {image_path}: {e}")
        return None

def get_augmented_image(image: np.ndarray, aug_normal: List[iaa.Augmenter]) -> np.ndarray:
    """
    Apply random data augmentation to an image.

    Args:
        image (np.ndarray): Input image.
        aug_normal (List[iaa.Augmenter]): List of data augmenters.

    Returns:
        np.ndarray: Augmented image.
    """
    aug_num = random.randint(1, len(aug_normal))
    selected_transforms = random.sample(aug_normal, aug_num)
    aug_seq = iaa.Sequential(selected_transforms, random_order=True)
    return aug_seq.augment_image(image)

def dataloader_builder(
    one_class_name,
    shot_num,
    expand_ratio,
    is_train,
    set_class,
    batch_size=8,
    dataset_name='MVTec-AD',
    resize=224,
    each_ratio=None,
    anomaly_type='sdas',
    work_num = 4
):
    root_dir = os.path.join('./data', dataset_name)
    DATASET_SUB_DIRS = {
        'MVTec-AD': 'mvtec',
        'MPDD': 'mpdd',
        'VisA': 'visa',
        'BTAD': 'btad'
    }
    if is_train:
        dataset = set_class[dataset_name]['train'](
            one_class_name=one_class_name,
            shot_num=shot_num,
            expand_ratio=expand_ratio,
            root_dir=root_dir,
            each_ratio=each_ratio,
            resize=resize,
            anomaly_type=anomaly_type
        )
    else:
        sub_dir = DATASET_SUB_DIRS.get(dataset_name)
        if sub_dir:
            root_dir = os.path.join(root_dir, sub_dir)
        dataset = set_class[dataset_name]['test'](
            class_name=one_class_name,
            resize=resize,
            root_dir=root_dir,
        )
    return DataLoader(dataset, batch_size=batch_size, num_workers=work_num), dataset

def generate_timestamp():
    return datetime.now().strftime("%Y%m%d%H%M%S%f")

def restore_and_save_image(tensor_image, tensor_mask, one_class_name, data_set_name):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    inv_normalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(MEAN, STD)],
        std=[1 / s for s in STD]
    )
    to_pil = transforms.ToPILImage()
    save_path = os.path.join('./experiments_result/pse_images', data_set_name, one_class_name)
    save_path_images = os.path.join(save_path, 'images')
    save_path_masks = os.path.join(save_path, 'masks')
    if not os.path.exists(save_path_images):
        os.makedirs(save_path_images, exist_ok=True)
    if not os.path.exists(save_path_masks):
        os.makedirs(save_path_masks, exist_ok=True)

        # 处理单张图像+掩码
        if len(tensor_image.shape) == 3:
            timestamp = generate_timestamp()
            # 保存图像
            img_pil = to_pil(inv_normalize(tensor_image))
            img_path = f"{save_path_images}/image_{timestamp}.jpg"
            img_pil.save(img_path)
            # 保存掩码（单通道去维度）
            mask_pil = to_pil(tensor_mask.squeeze(0))  # 移除通道维度
            mask_path = f"{save_path_masks}/mask_{timestamp}.jpg"
            mask_pil.save(mask_path)

        # 处理多批次图像+掩码
        elif len(tensor_image.shape) == 4:
            assert tensor_image.shape[0] == tensor_mask.shape[0], "图像和掩码批次数量不一致"
            for i, (img, msk) in enumerate(zip(tensor_image, tensor_mask)):
                mm = random.randint(0, 100000000)
                timestamp = generate_timestamp()
                # 保存图像
                img_pil = to_pil(inv_normalize(img))
                img_path = f"{save_path_images}/image_{timestamp}_{mm}.jpg"
                img_pil.save(img_path)
                # 保存掩码
                mask_pil = to_pil(msk.squeeze(0))  # 移除通道维度
                mask_path = f"{save_path_masks}/mask_{timestamp}_{mm}.jpg"
                mask_pil.save(mask_path)

        else:
            raise ValueError("输入张量形状错误，需为3D (C,H,W) 或4D (B,C,H,W)")

def tensor_gaussian_filter(input_tensor, sigma=4):
    """
    对输入的torch.Tensor进行高斯滤波，并返回滤波后的torch.Tensor
    :param input_tensor: 输入的torch.Tensor，形状可以是 (C, H, W) 或 (B, C, H, W)
    :param sigma: 高斯滤波的标准差，默认为4
    :return: 滤波后的torch.Tensor
    """
    # 检查输入张量的设备
    device = input_tensor.device
    # 将torch.Tensor转换为numpy.ndarray
    input_numpy = input_tensor.cpu().numpy()

    if len(input_numpy.shape) == 3:  # 单张图像 (C, H, W)
        output_numpy = np.zeros_like(input_numpy)
        for c in range(input_numpy.shape[0]):
            output_numpy[c] = gaussian_filter(input_numpy[c], sigma=sigma)
    elif len(input_numpy.shape) == 4:  # 批量图像 (B, C, H, W)
        output_numpy = np.zeros_like(input_numpy)
        for b in range(input_numpy.shape[0]):
            for c in range(input_numpy.shape[1]):
                output_numpy[b, c] = gaussian_filter(input_numpy[b, c], sigma=sigma)
    else:
        raise ValueError("输入张量的形状必须是 (C, H, W) 或 (B, C, H, W)")

def handle_feature_abnormal(ori_img, ori_mask, ):
    image, mask = None, None
    return image, mask

def add_noise_to_image(image_tensor, mask_tensor, std=0.015, mix_noise=5):
    batch_size = image_tensor.shape[0]
    noisy_image = image_tensor.clone()

    # 生成 mix_noise 组不同的噪声
    all_noises = torch.stack([
        torch.normal(0, std * 1.1 ** k, image_tensor.shape)
        for k in range(mix_noise)], dim=1)

    for i in range(batch_size):
        current_mask = mask_tensor[i]
        # 为当前批次随机选择一组噪声
        noise_index = torch.randint(0, mix_noise, (1,)).item()
        selected_noise = all_noises[i, noise_index]

        # 找出非零元素的位置
        non_zero_indices = current_mask.nonzero(as_tuple=True)

        # 创建一个全零的噪声张量
        noise_tensor = torch.zeros_like(current_mask, dtype=torch.float32)
        # 将噪声值放到非零位置
        noise_tensor[non_zero_indices] = selected_noise[non_zero_indices]

        # 将噪声张量加到原始图像的对应批次上
        noisy_image[i] += noise_tensor

    return noisy_image

def process_mask(input_path, one_class, one_dataset):
    out_path = os.path.join('./experiments_result/feature_mask', one_dataset, one_class)
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)

    path = []

    for i in range(len(input_path)):
        # 读取原始mask
        img = cv2.imread(input_path[i], cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # 查找所有连通组件
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

        # 创建新mask
        new_mask = np.zeros_like(binary)

        # 存储已放置的矩形信息 (x, y, w, h)
        placed_rects = []

        # 跳过背景（标签0）
        for label in range(1, num_labels):
            # 提取当前组件
            component = np.where(labels == label, 255, 0).astype(np.uint8)

            # 获取轮廓和最小矩形区域
            contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            x, y, w, h = cv2.boundingRect(contours[0])

            # 提取异常块
            patch = component[y:y + h, x:x + w]

            # 必须进行旋转
            angle = random.uniform(-45, 45)  # 随机旋转角度
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            patch = cv2.warpAffine(patch, M, (w, h))

            # 从其他变换方式中随机选择一种或多种
            transform_options = [
                "stretch", "squeeze", "warp", "flip",
                "scale", "perspective", "affine", "elastic"
            ]
            selected_transforms = random.sample(transform_options, k=random.randint(1, len(transform_options)))

            for transform_type in selected_transforms:
                if transform_type == "stretch":  # 拉伸
                    new_h = int(h * random.uniform(1.2, 2.0))
                    new_w = int(w * random.uniform(1.2, 2.0))
                    patch = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                elif transform_type == "squeeze":  # 挤压
                    pass
                    # new_h = int(h * random.uniform(0.5, 0.8))
                    # new_w = int(w * random.uniform(0.5, 0.8))
                    # patch = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                elif transform_type == "warp":  # 扭曲
                    rows, cols = patch.shape
                    src_points = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])
                    dst_points = np.float32([
                        [random.uniform(-0.1, 0.1) * cols, random.uniform(-0.1, 0.1) * rows],
                        [cols - 1 + random.uniform(-0.1, 0.1) * cols, random.uniform(-0.1, 0.1) * rows],
                        [random.uniform(-0.1, 0.1) * cols, rows - 1 + random.uniform(-0.1, 0.1) * rows],
                        [cols - 1 + random.uniform(-0.1, 0.1) * cols, rows - 1 + random.uniform(-0.1, 0.1) * rows]
                    ])
                    M = cv2.getPerspectiveTransform(src_points, dst_points)
                    patch = cv2.warpPerspective(patch, M, (cols, rows))
                elif transform_type == "flip":  # 翻转
                    flip_code = random.choice([-1, 0, 1])  # -1: 双向, 0: 垂直, 1: 水平
                    patch = cv2.flip(patch, flip_code)
                elif transform_type == "scale":  # 缩放
                    scale = random.uniform(0.8, 1.2)
                    patch = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                elif transform_type == "perspective":  # 透视变换
                    rows, cols = patch.shape
                    src_points = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])
                    dst_points = np.float32([
                        [random.uniform(0, cols * 0.2), random.uniform(0, rows * 0.2)],
                        [random.uniform(cols * 0.8, cols), random.uniform(0, rows * 0.2)],
                        [random.uniform(0, cols * 0.2), random.uniform(rows * 0.8, rows)],
                        [random.uniform(cols * 0.8, cols), random.uniform(rows * 0.8, rows)]
                    ])
                    M = cv2.getPerspectiveTransform(src_points, dst_points)
                    patch = cv2.warpPerspective(patch, M, (cols, rows))
                elif transform_type == "affine":  # 仿射变换
                    rows, cols = patch.shape
                    src_points = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
                    dst_points = np.float32([
                        [random.uniform(0, cols * 0.2), random.uniform(0, rows * 0.2)],
                        [random.uniform(cols * 0.8, cols), random.uniform(0, rows * 0.2)],
                        [random.uniform(0, cols * 0.2), random.uniform(rows * 0.8, rows)]
                    ])
                    M = cv2.getAffineTransform(src_points, dst_points)
                    patch = cv2.warpAffine(patch, M, (cols, rows))
                elif transform_type == "elastic":  # 弹性形变
                    alpha = random.uniform(10, 20)  # 形变强度
                    sigma = random.uniform(4, 6)  # 平滑系数
                    patch = elastic_deformation(patch, alpha, sigma)

            # 二值化处理
            _, patch = cv2.threshold(patch, 127, 255, cv2.THRESH_BINARY)

            # 获取变换后的有效区域
            trans_points = np.argwhere(patch > 0)
            if len(trans_points) == 0:
                continue

            min_y, min_x = np.min(trans_points, axis=0)
            max_y, max_x = np.max(trans_points, axis=0)
            valid_region = patch[min_y:max_y + 1, min_x:max_x + 1]
            new_h, new_w = valid_region.shape

            # 计算新位置（确保不紧贴边界）
            margin = 10  # 边界留白
            attempt = 0
            while True:
                # 调整 new_w 和 new_h 以适应图像大小
                if margin > binary.shape[1] - new_w - margin:
                    ratio = (binary.shape[1] - 2 * margin) / new_w
                    new_w = int(new_w * ratio)
                    new_h = int(new_h * ratio)
                    valid_region = cv2.resize(valid_region, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                if margin > binary.shape[0] - new_h - margin:
                    ratio = (binary.shape[0] - 2 * margin) / new_h
                    new_w = int(new_w * ratio)
                    new_h = int(new_h * ratio)
                    valid_region = cv2.resize(valid_region, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

                # 随机新位置
                new_x = random.randint(margin, binary.shape[1] - new_w - margin)
                new_y = random.randint(margin, binary.shape[0] - new_h - margin)

                # 检查重叠
                overlap = False
                for (px, py, pw, ph) in placed_rects:
                    if (new_x < px + pw and
                            new_x + new_w > px and
                            new_y < py + ph and
                            new_y + new_h > py):
                        overlap = True
                        break

                if not overlap or attempt > 100:
                    break
                attempt += 1

            # 放置到新mask
            if not overlap:
                roi = new_mask[new_y:new_y + new_h, new_x:new_x + new_w]
                roi[valid_region > 0] = 255
                placed_rects.append((new_x, new_y, new_w, new_h))

        # 保存结果
        path_ = os.path.join(out_path, f'{generate_timestamp()}_new_mask_{i}.png')
        path.append(path_)
        cv2.imwrite(path_, new_mask)

    return path

def elastic_deformation(image, alpha, sigma):
    """弹性形变"""
    random_state = np.random.RandomState(None)
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    distorted_image = map_coordinates(image, indices, order=1, mode="reflect").reshape(shape)
    return distorted_image.astype(np.uint8)

def give_me_mask(image, one_class, one_dataset):
    root_dir = './data/'
    map = {
        'MVTec-AD': 'mvtec',
        'BTAD': 'btad',
        'MPDD': 'mpdd',
        'VisA': 'visa'
    }
    gt_path = sorted(glob.glob(os.path.join(root_dir, one_dataset, map[one_dataset],
                                            one_class, 'ground_truth', '*', '*.png')))
    random.shuffle(gt_path)

    if len(image.shape) == 3:
        return [random.choice(gt_path)]
    else:
        return random.sample(gt_path, k=image.shape[0])

def handle_multi_mask(image_paths):
    # 定义图像预处理
    resize_transform = transforms.Resize((224, 224))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # 初始化两个列表来存储处理后的图像
    rgb_images = []
    gray_images = []

    for image_path in image_paths:
        # 读取图像
        image = Image.open(image_path)

        # Resize图像
        image = resize_transform(image)

        # 灰度读取（mask）
        gray_image = image.convert('L')  # 转换为灰度图像
        gray_image = to_tensor(gray_image)  # 转换为Tensor，值范围 [0, 1]
        gray_images.append(gray_image)

        # RGB读取
        rgb_image = image.convert('RGB')  # 确保图像是RGB格式
        rgb_image = to_tensor(rgb_image)  # 转换为Tensor
        rgb_image = normalize(rgb_image)  # 归一化
        rgb_images.append(rgb_image)

    # 将列表中的图像堆叠成张量
    rgb_tensor = torch.stack(rgb_images)  # 形状为 [B, 3, 224, 224]
    gray_tensor = torch.stack(gray_images)  # 形状为 [B, 1, 224, 224]

    return rgb_tensor, gray_tensor

def gaussian_filter_tensor(input_tensor, sigma=4):
        """
        对输入的torch.Tensor进行高斯滤波，并返回滤波后的torch.Tensor
        :param input_tensor: 输入的torch.Tensor，形状可以是 (C, H, W) 或 (B, C, H, W)
        :param sigma: 高斯滤波的标准差，默认为4
        :return: 滤波后的torch.Tensor
        """
        input_numpy = input_tensor.numpy()

        if len(input_numpy.shape) == 3:  # 单张图像 (C, H, W)
            output_numpy = np.zeros_like(input_numpy)
            for c in range(input_numpy.shape[0]):
                output_numpy[c] = gaussian_filter(input_numpy[c], sigma=sigma)
        elif len(input_numpy.shape) == 4:  # 批量图像 (B, C, H, W)
            output_numpy = np.zeros_like(input_numpy)
            for b in range(input_numpy.shape[0]):
                for c in range(input_numpy.shape[1]):
                    output_numpy[b, c] = gaussian_filter(input_numpy[b, c], sigma=sigma)
        else:
            raise ValueError("输入张量的形状必须是 (C, H, W) 或 (B, C, H, W)")

def load_custom_mask_tensor(mask_tensor, threshold=0.5):
    """
    处理已归一化的PyTorch张量输入，生成二值化mask并保持BCHW格式（C=1）
    Args:
        mask_tensor: 输入张量，形状可以是 [B,1,H,W], [1,H,W], [H,W]
                     （通道数必须为1或不存在，值范围 [0, 1]）
        threshold: 二值化阈值（默认0.5对应原128/255）
    Returns:
        mask_binary: 二值化浮点张量，形状为 [B,1,H,W]
    """
    # 检查输入合法性
    if not isinstance(mask_tensor, torch.Tensor):
        raise TypeError(f"Input must be a torch.Tensor, got {type(mask_tensor)}")
    if mask_tensor.min() < 0 or mask_tensor.max() > 1:
        raise ValueError("Input tensor values must be in [0, 1]")

    # 统一转换为四维张量 [B,1,H,W]
    if mask_tensor.ndim == 2:  # [H,W] -> [1,1,H,W]
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
    elif mask_tensor.ndim == 3:  # [1,H,W] -> [1,1,H,W]
        mask_tensor = mask_tensor.unsqueeze(1)
    elif mask_tensor.ndim == 4 and mask_tensor.shape[1] != 1:
        raise ValueError(f"Input must have 1 channel, got {mask_tensor.shape[1]} channels.")

    # 二值化处理
    mask_binary = torch.where(mask_tensor >= threshold,
                              torch.ones_like(mask_tensor),
                              torch.zeros_like(mask_tensor))

    return mask_binary.to(torch.float32)


def generate_smooth_gaussian(masks, neg_score=-0.1, sigma_scale=0.25, smooth_sigma=2.0, transition_width=10):
    """
    处理多批次输入并生成高斯分数图（支持批量输入）
    Args:
        masks: 输入mask张量，形状为 [B, 1, H, W] 或 [1, H, W]，值域需为 {0, 1}
        transition_width: 过渡区域宽度（像素）
    Returns:
        target: 高斯分数图，形状为 [B, 1, H, W]
    """
    # --- 输入合法性检查 ---
    assert not np.isnan(masks).any(), "输入 masks 包含 NaN"
    #assert np.all(np.logical_or(masks == 0, masks == 1)), "输入 masks 必须为二值 (0 或 1)"

    # --- 输入维度处理 ---
    if masks.ndim == 4 and masks.shape[1] == 1:  # [B,1,H,W] -> [B,H,W]
        masks = masks.squeeze(1)
    elif masks.ndim == 3 and masks.shape[0] == 1:  # [1,H,W] -> [1,H,W]
        pass
    else:
        raise ValueError(f"输入形状需为 [B,1,H,W] 或 [1,H,W]，当前形状: {masks.shape}")

    B, H, W = masks.shape
    targets = np.full((B, 1, H, W), neg_score, dtype=np.float32)

    for batch_idx in range(B):
        mask = masks[batch_idx]  # [H,W]
        labeled, num = label(mask)
        target = np.full((H, W), neg_score, dtype=np.float32)

        for obj_id in range(1, num + 1):
            obj_mask = (labeled == obj_id).astype(np.uint8)

            # --- 关键修复 1: 跳过空物体区域 ---
            if np.sum(obj_mask) == 0:
                continue  # 忽略无效物体

            # 计算质心（确保非空）
            cy, cx = center_of_mass(obj_mask)
            y, x = np.indices((H, W))
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

            # --- 关键修复 2: 计算标准差并防止为0 ---
            rows = np.any(obj_mask, axis=1)
            cols = np.any(obj_mask, axis=0)
            try:
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]
            except IndexError:
                continue  # 忽略无效边界

            h = max(ymax - ymin, 1)  # 防止 h=0
            w = max(xmax - xmin, 1)  # 防止 w=0
            sigma = max(h, w) * sigma_scale
            sigma = max(sigma, 1e-3)  # 防止 sigma=0

            # 生成高斯核
            gauss = np.exp(-dist ** 2 / (2 * sigma ** 2)) * obj_mask

            # 边界平滑权重
            dist_to_boundary = distance_transform_edt(obj_mask)
            boundary_weight = np.clip(dist_to_boundary / (sigma * 2), 0, 1)
            gauss_smoothed = gauss * boundary_weight

            # --- 关键修复 3: 安全归一化 ---
            if gauss_smoothed.max() > 0:
                gauss_min = gauss_smoothed.min()
                gauss_max = gauss_smoothed.max()
                if gauss_max - gauss_min < 1e-6:
                    gauss_smoothed = np.zeros_like(gauss_smoothed) + 0.3  # 避免除以零
                else:
                    gauss_smoothed = (gauss_smoothed - gauss_min) / (gauss_max - gauss_min) * 0.7 + 0.3
                target = np.maximum(target, gauss_smoothed)

        # 边界过渡优化
        dist_map = distance_transform_edt(mask == 0)
        transition_mask = (dist_map <= transition_width) & (dist_map > 0)
        transition_weight = 1 - (dist_map / transition_width)
        transition_weight = np.clip(transition_weight, 0, 1)

        target = np.where(
            transition_mask,
            target * transition_weight + neg_score * (1 - transition_weight),
            np.where(mask > 0, target, neg_score)
        )

        # 高斯模糊保护
        if not np.isnan(target).any():
            target = cv2.GaussianBlur(target, (0, 0), sigmaX=1.0)
            target = np.clip(target, neg_score, 1.0)
        else:
            target = np.full_like(target, neg_score)

        targets[batch_idx, 0] = target

    return targets

def upsample_and_smooth(pred_raw, target_size, method='bilinear', sigma=4, min_sigma=2, max_sigma=8):
    """
    上采样 + 高斯平滑
    Args:
        pred_raw: 网络原始输出 [B,C,H,W]
        target_size: 目标尺寸 (H, W)
        method: 上采样方法 ('bilinear' 或 'bicubic')
        sigma: 高斯核标准差（固定高斯）
        min_sigma/max_sigma: σ的动态范围（自适应高斯）
    Returns:
        pred_smooth: 上采样并平滑后的图像 [B,C,H,W]
    """
    # 上采样
    if method == 'bilinear':
        pred_up = F.interpolate(pred_raw, size=target_size, mode='bilinear', align_corners=False)
    elif method == 'bicubic':
        pred_up = F.interpolate(pred_raw, size=target_size, mode='bicubic', align_corners=False)
    else:
        raise ValueError(f"Unsupported upsampling method: {method}")

    # 高斯平滑
    if method == 'bilinear':
        # 固定高斯平滑
        kernel_size = int(6 * sigma) + 1
        x = torch.arange(kernel_size).float() - kernel_size // 2
        x = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel = torch.outer(x, x)
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, kernel_size).to(pred_raw.device)
        pred_smooth = F.conv2d(pred_up, kernel, padding=kernel_size // 2)
    else:
        # 自适应高斯平滑
        kernel_ = torch.tensor([[[[-1, 0, 1]] * 3]], dtype=torch.float32).to(pred_raw.device)
        grad_x = F.conv2d(pred_up, kernel_)
        grad_magnitude = torch.mean(torch.abs(grad_x))
        sigma = sigma * (1 - grad_magnitude.item())  # 梯度越大σ越小（保留细节）
        sigma = max(min_sigma, min(sigma, max_sigma))

        # 确保kernel_size为奇数
        kernel_size_val = int(6 * sigma) + 1
        if kernel_size_val % 2 == 0:
            kernel_size_val += 1
        kernel_size_val = max(1, kernel_size_val)  # 防止过小

        pred_smooth = gaussian_blur2d(
            pred_up,
            kernel_size=(kernel_size_val, kernel_size_val),
            sigma=(sigma, sigma)
        )

    return pred_smooth


def downsample_to_28x28(input_tensor: torch.Tensor, mode: str = 'bilinear') -> torch.Tensor:
    """
    将输入张量下采样到28x28分辨率

    参数：
    input_tensor: 输入张量，形状为[B, 1, H, W]
    mode: 下采样模式，可选 'avgpool'(默认)/'maxpool'/'bilinear'

    返回：
    下采样后的张量，形状为[B, 1, 28, 28]
    """
    assert len(input_tensor.shape) == 4, "输入必须是4D张量 [B, C, H, W]"
    assert input_tensor.shape[1] == 1, "输入通道数必须为1"

    # 选择下采样方式
    if mode == 'avgpool':
        pool = nn.AdaptiveAvgPool2d((28, 28))
        return pool(input_tensor)
    elif mode == 'maxpool':
        pool = nn.AdaptiveMaxPool2d((28, 28))
        return pool(input_tensor)
    elif mode == 'bilinear':
        return torch.nn.functional.interpolate(
            input_tensor,
            size=(28, 28),
            mode='bilinear',
            align_corners=False
        )
    else:
        raise ValueError(f"不支持的采样模式: {mode}，可选 ['avgpool', 'maxpool', 'bilinear']")

def anomaly_detection(score_map, thre):
    max_score = torch.amax(score_map, dim=(1, 2, 3))
    is_normal = max_score <= thre
    return is_normal, max_score

def multi_scale_anomaly_detection(score_map, factors=[1.0, 0.75, 0.5, 0.25], weights = [0.4, 0.3, 0.2, 0.1], threshold=0.5):
    """
    多尺度异常判定
    Args:
        score_map: 预测的异常分数图 [B,1,H,W]
        factors: 降采样因子列表
        threshold: 异常判定阈值
    Returns:
        is_anomaly: 是否异常的布尔值 [B]
        anomaly_score: 综合异常分数 [B]
    """
    batch_size = score_map.shape[0]
    anomaly_scores = torch.zeros(batch_size).to(score_map.device)  # 存储每个样本的综合异常分数, [0,0,0,,,,]

    for i, factor in enumerate(factors):
        # 降采样
        target_size = (int(score_map.shape[-2] * factor), int(score_map.shape[-1] * factor))
        resized_score = F.interpolate(score_map, size=target_size, mode='bilinear', align_corners=False)

        # 提取最大值
        max_score = torch.amax(resized_score, dim=(1, 2, 3))  # [B]
        anomaly_scores += max_score * weights[i]

    # 计算平均值
    anomaly_scores /= len(factors)

    # 判断异常
    is_anomaly = anomaly_scores > threshold

    return is_anomaly, anomaly_scores

def select_channels(indices: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    """
    根据索引选择特征通道

    参数：
    - indices: 通道索引张量 [num_selected_channels]
    - features: 输入特征张量 [B, 1536, H, W]

    返回：
    - selected_features: 筛选后的特征张量 [B, num_selected_channels, H, W]
    """
    # 确保索引是LongTensor类型
    if not indices.dtype == torch.int64:
        indices = indices.long()

    # 沿通道维度进行索引选择
    selected_features = features.index_select(dim=1, index=indices)

    return selected_features


def extract_anomaly(pse_image, pse_mask):
    """用二值mask提取图像异常区域，非异常区域置零

    Args:
        pse_image: 输入图像 [B, 3, 224, 224]
        pse_mask: 二值掩码 [B, 1, 224, 224]

    Returns:
        masked_image: 异常区域图像 [B, 3, 224, 224]
    """
    # 确保mask为float类型以保证乘法正确性
    return pse_image * pse_mask.type_as(pse_image)

def get_channel_idx(dataloader, fm, pm, device, num: int, idx_type: str = 'type1'):
    # 三种获取通道索引的办法
    # 初始化特征存储列表
    pse_features = []
    normal_features = []
    block_features = []
    with torch.no_grad():
        for i, (data_dict) in enumerate(dataloader):
            # 获取数据并移动到对应设备
            pse_img = data_dict['pse_image'].to(device)
            normal_img = data_dict['normal_image'].to(device)
            pse_mask = data_dict['pse_mask'].to(device)
            batch_size = normal_img.size(0)

            # 生成异常块
            block_img = extract_anomaly(pse_img, pse_mask)

            # 提取特征
            pse_feat, _ = _embed(pse_img, fm, pm) # [B x 28 x 28, 1536]
            normal_feat, _ = _embed(normal_img, fm ,pm) # [B x 28 x 28, 1536]
            block_feat, _ = _embed(block_img, fm, pm) # [B x 28 x 28, 1536]

            normal_features_ = normal_feat.view(batch_size, 28,28,1536)  # [B,1536,28,28]
            normal_features_ = normal_features_.permute(0, 3, 1, 2)
            pse_features_ = pse_feat.view(batch_size, 28, 28, 1536)  # [B,1536,28,28]
            pse_features_ = pse_features_.permute(0, 3, 1, 2)
            block_features_ = block_feat.view(batch_size, 28, 28, 1536)  # [B,1536,28,28]
            block_features_ = block_features_.permute(0, 3, 1, 2)

            # 存储特征
            pse_features.append(pse_features_)
            normal_features.append(normal_features_)
            block_features.append(block_features_)

        # 拼接所有batch的特征
        pse_features = torch.cat(pse_features, dim=0)
        normal_features = torch.cat(normal_features, dim=0)
        block_features = torch.cat(block_features, dim=0)

        # 根据索引类型选择处理路径
        if idx_type == "type1":
            # 计算特征差异
            diff = pse_features - normal_features - block_features

            # 计算每个通道的L2范数平方 [N, 1536]
            norms = torch.sum(diff.pow(2), dim=(2, 3))

            # 计算通道平均范数 [1536]
            channel_avg = torch.mean(norms, dim=0)

            # 排序并获取前num个索引
            sorted_values, sorted_indices = torch.sort(channel_avg)
            selected_indices = sorted_indices[:num]

            # 调整索引顺序（按原始通道顺序排序）
            final_indices, _ = torch.sort(selected_indices)

            return final_indices

        elif idx_type == "type2":
            pass

        elif idx_type == "type3":
            pass

        else:
            raise ValueError(f"Invalid index type: {idx_type}")

def save_channel_indices(idx, class_name, dataset_name):
    # 创建保存路径
    save_path = os.path.join('./experiments_result/chidx', dataset_name, class_name)
    os.makedirs(save_path, exist_ok=True)

    # 转换 idx 为 NumPy 数组
    idx_np = idx.numpy()

    # 构建文件名
    base_filename = 'ch_idx.npy'
    full_path = os.path.join(save_path, base_filename)

    # 检查文件是否已存在
    if os.path.exists(full_path):
        # 如果存在，添加时间戳
        timestamp = str(int(time.time()))
        new_filename = f'ch_idx_{timestamp}.npy'
        full_path = os.path.join(save_path, new_filename)

    # 保存 NumPy 数组到文件
    np.save(full_path, idx_np)
    #print(f"Channel indices saved to {full_path}")

def _embed(images, forward_modules, patch_maker):
    with torch.no_grad():
        features = forward_modules["feature_aggregator"](images)

    features_org = [features[layer] for layer in ['layer2', 'layer3']]

    features = [
        patch_maker.patchify(x, return_spatial_info=True) for x in features_org
    ]
    patch_shapes = [x[1] for x in features]
    features = [x[0] for x in features]
    ref_num_patches = patch_shapes[0]

    for i in range(1, len(features)):
        _features = features[i]
        patch_dims = patch_shapes[i]

        # TODO(pgehler): Add comments
        _features = _features.reshape(
            _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
        )
        _features = _features.permute(0, -3, -2, -1, 1, 2)
        perm_base_shape = _features.shape
        _features = _features.reshape(-1, *_features.shape[-2:])
        _features = F.interpolate(
            _features.unsqueeze(1),
            size=(ref_num_patches[0], ref_num_patches[1]),
            mode="bilinear",
            align_corners=False,
        )
        _features = _features.squeeze(1)
        _features = _features.reshape(
            *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
        )
        _features = _features.permute(0, -2, -1, 1, 2, 3)
        _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
        features[i] = _features
    features = [x.reshape(-1, *x.shape[-3:]) for x in features]

    features = forward_modules["preprocessing"](features)
    features = forward_modules["preadapt_aggregator"](features)
    return features, patch_shapes

def cal_map(scores, input_shape, patch_shape):
    scores = scores.reshape(1, patch_shape[0][0], patch_shape[0][1]).unsqueeze(1)
    scores = F.interpolate(scores, size=input_shape[1], mode='bilinear', align_corners=False)
    anomaly_map = scores.reshape(input_shape[1], input_shape[2]).cpu().numpy()
    anomaly_map = gaussian_filter(anomaly_map, sigma=4)
    return anomaly_map

def normalize(pr_list_sp, pr_px):
    pr_list_sp = np.array(pr_list_sp)
    min_sp = pr_list_sp.min(axis=-1).reshape(-1, 1)
    max_sp = pr_list_sp.max(axis=-1).reshape(-1, 1)
    pr_list_sp = (pr_list_sp - min_sp) / (max_sp - min_sp)
    pr_list_sp = np.mean(pr_list_sp, axis=0)

    pr_px = np.array(pr_px)
    min_scores = pr_px.reshape(len(pr_px), -1).min(axis=-1).reshape(-1, 1, 1, 1)
    max_scores = pr_px.reshape(len(pr_px), -1).max(axis=-1).reshape(-1, 1, 1, 1)
    pr_px = (pr_px - min_scores) / (max_scores - min_scores)
    pr_px = np.mean(pr_px, axis=0)

    return pr_list_sp, pr_px


def save_augmented_data(normal_feature, normal_mask, pse_feature, pse_mask, feature_mask, type_, dataset):
    # 生成统一时间戳（Unix时间戳）
    timestamp = str(int(time.time()))

    # 反标准化参数（保持在CPU）
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def denormalize(gpu_tensor):
        """反标准化并转换为HWC格式的numpy数组"""
        # 移动到CPU并处理
        tensor = gpu_tensor.clone().detach().squeeze(0).cpu()  # 移除batch维度并转移至CPU
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        return tensor.permute(1, 2, 0).mul(255).byte().numpy()  # HWC, uint8

    # 保存normal图像
    normal_img = denormalize(normal_feature)
    normal_img_path = f'./augimg/{dataset}/{type_}/normal/imgs/normal_{timestamp}.png'
    os.makedirs(os.path.dirname(normal_img_path), exist_ok=True)
    Image.fromarray(normal_img, 'RGB').save(normal_img_path)

    # 保存pse图像
    pse_img = denormalize(pse_feature)
    pse_img_path = f'./augimg/{dataset}/{type_}/pse/imgs/pse_{timestamp}.png'
    os.makedirs(os.path.dirname(pse_img_path), exist_ok=True)
    Image.fromarray(pse_img, 'RGB').save(pse_img_path)

    def process_mask(gpu_mask):
        """处理GPU上的mask为二值图"""
        # 移动到CPU处理
        mask = gpu_mask.squeeze().mul(255).byte().cpu().numpy()
        return np.where(mask > 127, 255, 0).astype(np.uint8)  # 二值化

    # 保存normal mask
    mask_normal = process_mask(normal_mask)
    mask_normal_path = f'./augimg/{dataset}/{type_}/normal/masks/mask_normal_{timestamp}.png'
    os.makedirs(os.path.dirname(mask_normal_path), exist_ok=True)
    Image.fromarray(mask_normal, 'L').save(mask_normal_path)

    # 保存pse mask
    mask_pse = process_mask(pse_mask)
    mask_pse_path = f'./augimg/{dataset}/{type_}/pse/masks/mask_pse_{timestamp}.png'
    os.makedirs(os.path.dirname(mask_pse_path), exist_ok=True)
    Image.fromarray(mask_pse, 'L').save(mask_pse_path)

    # 保存feature mask
    mask_feature = process_mask(feature_mask)
    mask_feature_path = f'./augimg/{dataset}/{type_}/feature/masks/mask_feature_{timestamp}.png'
    os.makedirs(os.path.dirname(mask_feature_path), exist_ok=True)
    Image.fromarray(mask_feature, 'L').save(mask_feature_path)