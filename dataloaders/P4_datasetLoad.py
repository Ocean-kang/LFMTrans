import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

#数据读取工具(通用)
import numpy as np
from PIL import Image
import warnings
import cv2
warnings.filterwarnings("ignore", category=UserWarning)

#实验记录
# from sacred import Experiment
# import logging
# from easydict import EasyDict as edict

# DatasetLoad库
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

# ADE20K数据集工具
from PrefetchLoader import PrefetchLoader
from ADE20K_info import *

# ADE_150信息
import ade150_obj

#clip
import clip

class ADE20K(Dataset):
    def __init__(
                 self,
                 split='all',
                 dataset_root_dir=None,
                 transform=None,
                 nfiles=None,
                 num_samples=None,
                 num_works=0,
                 ):
        assert split in ['train', 'val', 'all']
        if split == 'train':
            self.split = 'training'
        elif split == 'valid':
            self.split = 'validation'
        else:
            self.split = 'all'

        self.dataset_root_dir = dataset_root_dir
        self.transform = transform
        self.nfiles = nfiles
        self.num_samples = num_samples
        self.num_works = num_works

        # ADE20K数据信息加载
        self.index_ade20k = index_ade20k
        self.filesname = self.index_ade20k['filename'] # 所有文件的集合
        nfiles = len(self.filesname)# 获取长度(数据集大小计算)

        samples_list_train = []
        samples_list_valitation = []
        sample_list_all = []
        sample_list = []

        #加载数据库
        if self.split == 'all':
            sample_list_all = [i for i in range(len(self.filesname))]
            sample_list = sample_list_all
        elif self.split == 'training':
            samples_list_train = [k for k in range(0,25257)]
            sample_list = samples_list_train
        else:
            samples_list_valitation = [k for k in range(25258,27258)]
            sample_list = samples_list_valitation
        if self.num_samples is not None:
            self.samples_list = sample_list[:self.num_samples]
        self.samples_list = sample_list

    def __len__(self):
        return len(self.samples_list)
    
    def __getitem__(self, idx):
        img_ID = self.samples_list[idx]
        img_meta = load_image(img_ID)

        # image
        image = Image.open(img_meta['img_name'])

        sample_ = dict()
        sample_['img'] = image

        if self.transform is not None:
            tmp_img = self.transform(sample_['img'])
            sample_['img'] = tmp_img
        else:
            tmp_img = sample_['img'].convert('RGB')
            sample_['img'] = tmp_img

        sample = dict()
        sample['img'] = sample_['img']
        return sample

class ADE_150(Dataset):
    def __init__(self,
                 dataset_root_dir=None,
                 transform=None,
                 num_samples=None,
                 num_images=None,                 
                 ):
        self.dataset_root_dir = dataset_root_dir
        self.transform = transform
        self.num_samples = num_samples# 记载多少数据
        self.num_images = num_images# 记载多少obj特征图像进入
        self.index_ade20k = index_ade20k# ADE20K数据信息加载
        self.filesname = self.index_ade20k['filename'] # 所有文件的集合
        self.nfiles = len(self.filesname)# 获取长度(ADE20K数据集大小计算)
        self.ade150_objs = ade150_obj.load_txt(ade150_obj.A150_PATH) # 含有ADE_150所有的obj的list
        self.ade150_index = obj_index(self.ade150_objs)# 记录ADE_150里面obj出现的位置

        # 加载多少ADE-150的obj进入batch
        samples_list = self.ade150_index
        if self.num_samples is not None:
            samples_list = samples_list[:self.num_samples]
        self.samples_list = samples_list
            
    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, idx):
        id = self.samples_list[idx]# obj所在位置
        
        # 裁剪处obj后回复原图大小
        def crop_img(mask,raw_img:str):
            raw_size = (raw_img.shape[1],raw_img.shape[0])# 原图的长与宽
            x, y, w, h = cv2.boundingRect(mask)# 寻找边界[x,y:左上顶点.w,h:BoundingBox长宽]
            cropped_img = raw_img[y:y+h, x:x+w]
            resized_crop = cv2.resize(cropped_img,raw_size)
            return resized_crop
        
        # 找寻ADE150的obj被含有的图像的id
        def collect_img(input_index:int,num_img=None)->list:
            output_list = list()
            count_obj = self.index_ade20k['objectPresence'][input_index,:]# obj_i所在的数组
            tmp = np.where(count_obj != 0)[0]
            tmp = list(tmp)
            if num_img is not None:
                output_list = tmp[:num_img]
            else:
                output_list = tmp[:]
            return output_list
        
        #加载该id下的ADE20K的obj所有图像
        def load_obj_images(input_index:int, num_imgs=None) -> list:
            output_list = list()
            if num_imgs is not None:
                tmp = collect_img(input_index,num_img=num_imgs)
            else:
                tmp = collect_img(input_index)
            for _ in tmp:
                img_info = load_image(_)
                img = cv2.imread(img_info['img_name'])[:,:,::-1]# 读取原图
                #提取mask
                img_mask = img.copy()
                img_mask = img_mask[:,:,::-1]
                img_mask[img_info['class_mask'] != id+1] *= 0
                img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
                if np.any(img_mask != 0):
                    output_list.append(crop_img(img_mask,img))
                else:
                    output_list.append(img)
            return output_list
        
        #__getitem__运行逻辑
        f_imgs = load_obj_images(input_index=id, num_imgs=self.num_images)
        sample_ = dict()
        sample_['imgs'] = f_imgs
        sample_['obj_name'] = self.ade150_objs[idx]
        #transformer操作
        if self.transform is not None:
            sample_['imgs'] = [self.transform(_) for _ in sample_['imgs']]
        #返回值
        sample = dict()
        sample['imgs'] = sample_['imgs']
        sample['obj_name'] = sample_['obj_name']
        return sample

#将图片转换成为RGB形式
def convert_image_to_rgb(image):
    return image.convert("RGB")

# 加载目标train数据集与目标数据
def load_train_ADE20K(data_dir,n_px, num_samples=None):
    train_transform = Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])# 224为clip的CNN网络提供
    return ADE20K(dataset_root_dir=data_dir, transform=train_transform, split='train', num_samples=num_samples)

# 加载目标val数据集与目标数据
def load_val_ADE20K(data_dir, n_px, num_samples=None):
    val_transform = Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])# 224为clip的CNN网络提供
    return ADE20K(dataset_root_dir=data_dir, transform=val_transform, split='val', num_samples=num_samples)

# 加载ADE20K所有数据
def load_ADE20K(data_dir, num_samples=None, num_imgs=None, transforms=None):
    val_transform = Compose([
        ToPILImage(),
        transforms,
    ])
    return ADE20K(dataset_root_dir=data_dir, transform=val_transform, split='all', num_samples=num_samples, num_imgs=num_imgs)

# 加载ADE150的img与img_mask
def load_ADE150(data_dir, num_samples=None, num_imgs=None, transforms=None):
    transform_ade150 = Compose([
        ToPILImage(),
        transforms,
    ])
    return ADE_150(dataset_root_dir=data_dir, num_samples=num_samples, num_images=num_imgs, transform=transform_ade150)

DATASET_PATH = r'/20230031/code/OYMK_OVSS/ADE20K_Load/dataset'
SAVEPATH = r'/20230031/code/OYMK_OVSS/Data/Tensor_data/ADE20K_2017/ADE-150/Clip_V_F(10)'

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

#建立loader
def bulid_loader():
    dataset_ade150 = load_ADE150(data_dir=DATASET_PATH, transforms=preprocess, num_samples=150, num_imgs=10)
    loader_ade150 = DataLoader(dataset_ade150, batch_size=150, shuffle=False, pin_memory=False)
    if torch.cuda.is_available():
        loader_ade150 = PrefetchLoader(loader_ade150)
    return loader_ade150



