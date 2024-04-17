import torch
import torchvision
from torch import nn
from d2l import torch as d2l
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import albumentations as A
import os
from torchvision import transforms
import matplotlib.pyplot as plt
import random
def random_horizontal_flip(img, depth, p=0.5):
    """
    Randomly flip the image horizontally.

    Args:
        img (PIL.Image or numpy.ndarray): Input image.
        depth (PIL.Image or numpy.ndarray): Depth image.
        p (float): Probability of applying horizontal flip.

    Returns:
        PIL.Image or numpy.ndarray: Flipped image if condition is met, else original image.
    """
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if isinstance(depth, np.ndarray):
            depth = np.fliplr(depth)
        else:
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
    return img, depth


def random_vertical_flip(img, deep, p=0.5):
    """
    Randomly flip the image vertically.

    Args:
        img (PIL.Image): Input image.
        deep (PIL.Image or np.ndarray): Input depth image.
        p (float): Probability of applying vertical flip (default: 0.5).

    Returns:
        PIL.Image: Flipped image if condition is met, else original image.
        PIL.Image or np.ndarray: Flipped depth image if condition is met, else original depth image.
    """
    if random.random() < p:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        if isinstance(deep, np.ndarray):
            deep = np.flipud(deep)
        else:
            deep = deep.transpose(Image.FLIP_TOP_BOTTOM)
    return img, deep

def random_resized_crop(img, deep, scale=(0.1, 1), ratio=(0.5, 2), p=0.5):
    """
    Randomly resize and crop the image.

    Args:
        img (PIL.Image): Input image.
        deep (PIL.Image or np.ndarray): Input depth image.
        scale (tuple): Range for the relative size of the cropped area (default: (0.1, 1)).
        ratio (tuple): Range for the aspect ratio of the cropped area (default: (0.5, 2)).
        p (float): Probability of applying the crop operation (default: 0.5).

    Returns:
        PIL.Image: Cropped image.
        PIL.Image or np.ndarray: Cropped depth image.
    """
    if random.random() < p:
        width, height = img.size
        target_area = random.uniform(*scale) * width * height
        aspect_ratio = random.uniform(*ratio)
        w = int(round((target_area * aspect_ratio) ** 0.5))
        h = int(round((target_area / aspect_ratio) ** 0.5))
        sigw = min(width, w)
        sigh = min(height, h)
        balw = width - sigw
        balh = height - sigh
        i = random.randint(0, balw)
        j = random.randint(0, balh)
        img_cropped = img.crop((i, j + sigh, i + sigw, j))
        if isinstance(deep, np.ndarray):
            # 计算裁剪后的数组的起始行和列索引
            i_deep = i
            j_deep = j
            # 裁剪数组
            deep_cropped = deep[j_deep:j_deep + sigh, i_deep:i_deep + sigw]
        else:
            # 如果深度图像是 PIL 图像，使用相同的裁剪区域来裁剪它
            deep_cropped = deep.crop((i, j, i + sigw, j + sigh))
        return img_cropped, deep_cropped
    else:
        # Return original images if crop operation is not applied
        return img, deep




# def random_resized_crop(img, deep, scale=(0.1, 1), ratio=(0.5, 2)):
#     """
#     Randomly resize and crop the image.
#
#     Args:
#         img (PIL.Image): Input image.
#         deep (PIL.Image or np.ndarray): Input depth image.
#         scale (tuple): Range for the relative size of the cropped area (default: (0.1, 1)).
#         ratio (tuple): Range for the aspect ratio of the cropped area (default: (0.5, 2)).
#
#     Returns:
#         PIL.Image: Cropped image.
#         PIL.Image or np.ndarray: Cropped depth image.
#     """
#     width, height = img.size
#     target_area = random.uniform(*scale) * width * height
#     aspect_ratio = random.uniform(*ratio)
#     w = int(round((target_area * aspect_ratio) ** 0.5))
#     h = int(round((target_area / aspect_ratio) ** 0.5))
#     sigw = min(width,w)
#     sigh = min(height,h)
#     balw = width-sigw
#     balh = height-sigh
#     i = random.randint(0, balw)
#     j = random.randint(0, balh)
#     img_cropped = img.crop((i, j + sigh, i + sigw, j ))
#     if isinstance(deep, np.ndarray):
#         # 计算裁剪后的数组的起始行和列索引
#         i_deep = i
#         j_deep = j
#         # 裁剪数组
#         deep_cropped = deep[j_deep:j_deep + sigh, i_deep:i_deep + sigw]
#     else:
#         # 如果深度图像是 PIL 图像，使用相同的裁剪区域来裁剪它
#         deep_cropped = deep.crop((i, j, i + sigw, j + sigh))
#     return img_cropped,deep_cropped


#官方的：
# augmentation = transforms.Compose([
#     transforms.RandomApply([
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomVerticalFlip(),
#         transforms.RandomResizedCrop(size=(320, 320), scale=(0.1, 1), ratio=(0.5, 2)),
#     ], p=1.0)  # 使用RandomApply确保每次都应用数据增强
# ])
# def process_images_and_depths(image_folder, depth_folder, output_folderimg, output_folderdeep, num_images):
#     # 确保输出文件夹存在
#     os.makedirs(output_folderimg, exist_ok=True)
#     os.makedirs(output_folderdeep, exist_ok=True)
#
#     # 加载图像和深度图像文件名并排序
#     image_filenames = sorted(os.listdir(image_folder))
#     depth_filenames = sorted(os.listdir(depth_folder))
#
#     # 确保图像文件和深度文件数量一致
#     assert len(image_filenames) == len(depth_filenames)
#
#     for i in range(num_images):
#         # 读取图像和深度图像
#         image_filename = os.path.join(image_folder, image_filenames[i])
#         depth_filename = os.path.join(depth_folder, depth_filenames[i])
#         image = Image.open(image_filename)
#         depth = np.load(depth_filename)
#
#         # 应用数据增强
#         augmented_image = augmentation(image)
#         augmented_depth = augmentation(Image.fromarray(depth.astype(np.uint8), mode='L'))
#
#         # 存储增强后的图像和深度图像
#         output_image_filename = os.path.join(output_folderimg, f"{i:04d}.png")
#         output_depth_filename = os.path.join(output_folderdeep, f"{i:08d}_depth.npy")
#         augmented_image.save(output_image_filename)
#         np.save(output_depth_filename, np.array(augmented_depth))
#
# # 使用示例
# process_images_and_depths("data/blender/imageo", "data/blender/deptho", "data/blender/augmented_data", "data/blender/augmented_depths", num_images=4500)