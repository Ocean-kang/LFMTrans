import os
from PIL import Image
import pickle as pkl

import pandas as pd
import numpy as np
from pycocotools.coco import COCO
from torchvision import transforms as pth_transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset

from dataloaders.coco_id_idx_map import coco_stuff_id_idx_map, thingstuff171
from dataloaders import transforms

def load_dataset_coco(cfg, split='train', num_samples=None, _log=None):

    train_transform = pth_transforms.Compose([
        transforms.ToTensor(toCUDA=False),
        transforms.ResizeTensor(size=(cfg.dataset.resize, cfg.dataset.resize), img_only=False),
    ])

    dataset = MSCOCO17(transform=train_transform,
                       split=split,
                       dataset_root_dir=cfg.dataset.root_dir_mscoco,
                       num_samples=num_samples,
                       orientation=0,
                       )

    return dataset

class MSCOCO17(Dataset):
    def __init__(self,
                 split=None,
                 dataset_root_dir=None,
                 num_things=80,
                 num_stuff=91,
                 transform=None,
                 num_samples=None,
                 orientation=0,
                 ):
        assert split in ['train', 'val']
        self.split = 'train2017' if split == 'train' else 'val2017'
        self.dataset_root_dir = dataset_root_dir
        self.transform = transform
        self.num_samples = num_samples
        self.num_things = num_things
        self.num_stuff = num_stuff

        self.JPEGPath = f"{self.dataset_root_dir}/{self.split}"
        self.PNGPath = f"{self.dataset_root_dir}/annotations/{self.split}"
        self.annFile = f"{self.dataset_root_dir}/annotations/instances_{self.split}.json"
        self.coco = COCO(self.annFile)
        all_ids = self.coco.imgToAnns.keys()

        from dataloaders.coco_id_idx_map import COCO_CATEGORIES
        self.name_list = thingstuff171

        samples_list_1 = []
        samples_list_2 = []
        for id in all_ids:

            img_meta = self.coco.loadImgs(id)
            assert len(img_meta) == 1
            H, W = img_meta[0]['height'], img_meta[0]['width']
            if H < W:
                samples_list_1.append(id)
            else:
                samples_list_2.append(id)

        if orientation == 0:
            samples_list = samples_list_1 + samples_list_2
        elif orientation == 1:
            samples_list = samples_list_1
        elif orientation == 2:
            samples_list = samples_list_2
        else:
            raise NotImplementedError

        if self.num_samples is not None:
            samples_list = samples_list[:self.num_samples]
        self.samples_list = samples_list


    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, idx):

        id = self.samples_list[idx]
        img_meta = self.coco.loadImgs(id)
        assert len(img_meta) == 1
        img_meta = img_meta[0]

        # image
        image = np.array(Image.open(f"{self.JPEGPath}/{img_meta['file_name']}").convert('RGB'))
        label_cat = np.array(Image.open(f"{self.PNGPath}/{img_meta['file_name'].replace('jpg', 'png')}"))
        assert self.num_stuff + self.num_things == 171
        _coco_id_idx_map = np.vectorize(lambda x: coco_stuff_id_idx_map[x])
        label_cat = _coco_id_idx_map(label_cat)

        meta = {
            'category': [self.name_list[i] for i in np.unique(label_cat) if i != 459 and i != 255],
            'original_size': tuple(label_cat.shape),
        }

        sample_ = dict()
        sample_['img'] = image
        sample_['label_cat'] = label_cat
        sample_['meta'] = meta


        if self.transform is not None:
            sample_ = self.transform(sample_)

        sample = dict()
        sample['images'] = sample_['img']
        sample['label_cat'] = sample_['label_cat']

        return sample

def load_dataset_ade20k(cfg, set='A150', split='train', num_samples=None, _log=None):

    train_transform = pth_transforms.Compose([
        transforms.ToTensor(toCUDA=False),
        transforms.ResizeTensor(size=(cfg.dataset.resize, cfg.dataset.resize), img_only=False),
    ])
    if set == 'A150':
        dataset = ADE20K_150(
            transform=train_transform,
            split=split,
            dataset_root_dir=cfg.dataset.root_dir_ade20k,
            num_samples=num_samples)
    else:
        dataset = ADE20K(
            transform=train_transform,
            split=split,
            dataset_root_dir=cfg.dataset.root_dir_ade20k,
            num_samples=num_samples)

    return dataset

class ADE20K_150(Dataset):
    def __init__(self,
                 split=None,
                 dataset_root_dir=None,
                 transform=None,
                 num_samples=None,
                 ):
        assert split in ['train', 'val']
        self.split = 'training' if split == 'train' else 'validation'
        self.dataset_root_dir = os.path.join(dataset_root_dir, 'ADEChallengeData2016')
        self.transform = transform
        self.num_samples = num_samples

        self.JPEGPath = f"{self.dataset_root_dir}/images/{self.split}"
        self.annPath = f"{self.dataset_root_dir}/annotations/{self.split}"
        self.objectInfo150 = pd.read_csv(f"{self.dataset_root_dir}/objectInfo150.csv")
        self.idx_list = list(self.objectInfo150.iloc[:, 3])
        self.name_list = list(self.objectInfo150.iloc[:, 4])

        self.samples_list = [_.split('.')[0] for _ in os.listdir(self.JPEGPath)]


        if self.num_samples is not None:
            self.samples_list = self.samples_list[:self.num_samples]


    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, idx):

        sample = self.samples_list[idx]

        # image
        image = np.array(Image.open(f"{self.JPEGPath}/{sample}.jpg").convert('RGB'))
        label_cat = np.array(Image.open(f"{self.annPath}/{sample}.png"))
        assert not label_cat.max() > 150
        assert label_cat.dtype == np.uint8
        label_cat = label_cat - 1  # 0 (ignore) becomes 255. others are shifted by 1
        meta = {
            'category': [self.name_list[i] for i in np.unique(label_cat) if i != 255],
            'original_size': tuple(label_cat.shape),
        }

        sample_ = dict()
        sample_['img'] = image
        sample_['label_cat'] = label_cat
        sample_['meta'] = meta

        if self.transform is not None:
            sample_ = self.transform(sample_)

        sample = dict()

        sample['images'] = sample_['img']
        sample['label_cat'] = sample_['label_cat']

        return sample

class ADE20K(Dataset):
    def __init__(self,
                 split=None,
                 dataset_root_dir=None,
                 transform=None,
                 num_samples=None,
                 ):
        assert split in ['train', 'val']
        self.split = 'training' if split == 'train' else 'validation'
        self.dataset_root_dir = dataset_root_dir
        self.transform = transform
        self.num_samples = num_samples

        index_ade20k = pkl.load(open(f"{self.dataset_root_dir}/ADE20K_2021_17_01/index_ade20k.pkl", "rb"))

        from dataloaders.ADE20K import ADE20K_SEM_SEG_FULL_CATEGORIES
        self.id_map = {}
        for cat in ADE20K_SEM_SEG_FULL_CATEGORIES:
            self.id_map[cat["id"]] = cat["trainId"]

        self.name_list = [cat['name'] for cat in ADE20K_SEM_SEG_FULL_CATEGORIES]

        self.IMG_list = [os.path.join(folder, filename)
                         for (folder, filename) in zip(index_ade20k['folder'], index_ade20k['filename'])
                         if self.split in folder]

        self.ANN_list = [os.path.join(folder, filename.split('.')[0]+'_seg.png')
                             for (folder, filename) in zip(index_ade20k['folder'], index_ade20k['filename'])
                          if self.split in folder]

        assert len(self.IMG_list) == len(self.ANN_list)

    def __len__(self):
        return len(self.IMG_list)


    def __getitem__(self, idx):

        img_path = self.IMG_list[idx]
        ann_path = self.ANN_list[idx]

        # image
        image = np.array(Image.open(os.path.join(self.dataset_root_dir, img_path)).convert('RGB'))
        label_cat_ = np.array(Image.open(os.path.join(self.dataset_root_dir, ann_path)))
        R = label_cat_[:, :, 0]
        G = label_cat_[:, :, 1]
        label_cat_ = (R / 10).astype(np.int32) * 256 + (G.astype(np.int32))

        assert image.dtype == np.uint8
        assert label_cat_.dtype == np.int32
        label_cat = np.zeros_like(label_cat_, dtype=np.uint16) + 65535
        for obj_id in np.unique(label_cat_):
            if obj_id in self.id_map:
                label_cat[label_cat_ == obj_id] = self.id_map[obj_id]

        meta = {
            'category': [self.name_list[i] for i in np.unique(label_cat) if i != 65535],
            'original_size': tuple(label_cat.shape),
        }

        sample_ = dict()
        sample_['img'] = image
        sample_['label_cat'] = label_cat
        sample_['meta'] = meta

        if self.transform is not None:
            sample_ = self.transform(sample_)

        sample = dict()

        sample['images'] = sample_['img']
        sample['label_cat'] = sample_['label_cat']

        return sample

def load_train_dataset_pc(cfg, set='pc59', split='train', num_samples=None, _log=None):

    train_transform = pth_transforms.Compose([
        transforms.ToTensor(toCUDA=False),
        transforms.ResizeTensor(size=(cfg.dataset.resize, cfg.dataset.resize), img_only=False),
    ])

    dataset = PascalContext(
        transform=train_transform,
        set=set,
        split=split,
        dataset_root_dir=cfg.dataset.root_dir_pc,
        num_samples=num_samples)

    return dataset

class PascalContext(Dataset):
    def __init__(self,
                 split=None,
                 set='pc59',
                 dataset_root_dir=None,
                 transform=None,
                 num_samples=None,
                 ):
        assert split in ['train', 'val']
        assert set in ['pc59', 'pc459']
        self.split = 'training' if split == 'train' else 'validation'
        self.set = set
        self.dataset_root_dir = dataset_root_dir + '/VOCdevkit/VOC2010'
        self.transform = transform
        self.num_samples = num_samples
        assert split == 'val' if set == 'pc459' else True

        if set == 'pc59':
            from dataloaders.PascalContext import pc59 as name_list
        else:
            from dataloaders.PascalContext import pc459 as name_list

        self.name_list = name_list

        ann_path = os.path.join(self.dataset_root_dir + '/annotations_detectron2',
                                f'{set}_{split}')
        sample_list = [s.split('.')[0] for s in os.listdir(ann_path)]

        self.IMG_list = [os.path.join(self.dataset_root_dir + '/JPEGImages', s + '.jpg') for s in sample_list]

        if set == 'pc59':
            self.ANN_list = [self.dataset_root_dir + '/annotations_detectron2' + f'/{set}_{split}/' + s + '.png' for s in sample_list]
        else:
            self.ANN_list = [self.dataset_root_dir + '/annotations_detectron2' + f'/{set}_{split}/' + s + '.tif' for s in sample_list]

        assert len(self.IMG_list) == len(self.ANN_list)

    def __len__(self):
        return len(self.IMG_list)


    def __getitem__(self, idx):

        img_path = self.IMG_list[idx]
        ann_path = self.ANN_list[idx]

        # image
        image = np.array(Image.open(img_path).convert('RGB'))
        label_cat = np.array(Image.open(ann_path)).astype(np.int32)


        meta = {
            'category': [self.name_list[i] for i in np.unique(label_cat) if i != 459 and i != 255],
            'original_size': tuple(label_cat.shape),
        }

        sample_ = dict()
        sample_['img'] = image
        sample_['label_cat'] = label_cat
        sample_['meta'] = meta

        if self.transform is not None:
            sample_ = self.transform(sample_)

        sample = dict()

        sample['images'] = sample_['img']
        sample['label_cat'] = sample_['label_cat']

        return sample

def load_train_dataset_voc(cfg, set='voc20', split='train', num_samples=None, _log=None):

    train_transform = pth_transforms.Compose([
        transforms.ToTensor(toCUDA=False),
        transforms.ResizeTensor(size=(cfg.dataset.resize, cfg.dataset.resize), img_only=False),
    ])

    dataset = PascalVOC(
        transform=train_transform,
        set=set,
        split=split,
        dataset_root_dir=cfg.dataset.root_dir_voc,
        num_samples=num_samples)

    return dataset

class PascalVOC(Dataset):
    def __init__(self,
                 split=None,
                 set='voc20',
                 dataset_root_dir=None,
                 transform=None,
                 num_samples=None,
                 ):
        assert split in ['train', 'val']
        assert set in ['voc20', 'voc20b']
        self.split = 'training' if split == 'train' else 'validation'
        self.set = set
        self.dataset_root_dir = dataset_root_dir + '/VOCdevkit/VOC2012'
        self.transform = transform
        self.num_samples = num_samples


        if set == 'voc20':
            from dataloaders.PascalContext import pc20 as name_list
            ann_path = os.path.join(self.dataset_root_dir + '/annotations_detectron2',
                                    f'{split}')
        else:
            from dataloaders.PascalContext import pc20b as name_list
            ann_path = os.path.join(self.dataset_root_dir + '/annotations_detectron2_bg',
                                    f'{split}')
        self.name_list = name_list

        sample_list = [s.split('.')[0] for s in os.listdir(ann_path)]

        self.IMG_list = [os.path.join(self.dataset_root_dir + '/JPEGImages', s + '.jpg') for s in sample_list]

        self.ANN_list = [f'{ann_path}/' + s + '.png' for s in sample_list]

        # assert len(self.IMG_list) == len(self.ANN_list)

    def __len__(self):
        return len(self.ANN_list)


    def __getitem__(self, idx):

        img_path = self.IMG_list[idx]
        ann_path = self.ANN_list[idx]

        # image
        image = np.array(Image.open(img_path).convert('RGB'))
        label_cat = np.array(Image.open(ann_path)).astype(np.int32)
        idx = np.unique(label_cat)
        # for id in idx:
        #     if id in [0, 6, 13, 23, 37, 46, 51, 53, 72, 73, 78, 80, 82, 88, 90, 91, 92, 93, 94, 96, 98, 110, 111, 113, 115, 116, 117, 119, 124, 126, 131, 132, 134, 136, 142, 145, 146, 155, 160, 162, 163, 165, 167, 170, 172, 176, 177, 178, 182, 187, 197, 205, 208, 209, 211, 214, 216, 221, 226, 228, 233, 235, 236, 238, 240, 242, 248, 252, 279, 284, 287, 303, 309, 311, 316, 320, 321, 324, 326, 330, 339, 340, 342, 350, 351, 357, 361, 363, 364, 368, 371, 374, 381, 385, 390, 391, 393, 403, 407, 408, 410, 416, 420, 425, 427, 428, 432, 438, 441, 447, 448, 449, 450, 454]:
        #         print(f'{idx} available')

        meta = {
            'category': [self.name_list[i] for i in np.unique(label_cat) if i != 459 and i != 255],
            'original_size': tuple(label_cat.shape),
        }

        sample_ = dict()
        sample_['img'] = image
        sample_['label_cat'] = label_cat
        sample_['meta'] = meta

        if self.transform is not None:
            sample_ = self.transform(sample_)

        sample = dict()

        sample['images'] = sample_['img']
        sample['label_cat'] = sample_['label_cat']

        return sample