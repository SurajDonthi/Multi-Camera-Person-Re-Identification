# from argparse import ArgumentParser
# import os
from glob import glob
from typing import Optional

import joblib
import pytorch_lightning as pl
from pathlib2 import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from metrics import smooth_st_distribution
from utils import get_ids


class ReIDDataset(Dataset):

    def __init__(self, data_dir: str, transform=None, target_transform=None):
        super(ReIDDataset, self).__init__()
        self.data_dir = str(data_dir)
        self.transform = transform
        self.target_transform = target_transform
        self._init_data()

    def _init_data(self):
        if 'market' in self.data_dir.lower():
            self.dataset = 'market'
        elif 'duke' in self.data_dir.lower():
            self.dataset = 'duke'

        self.imgs = glob(self.data_dir + '/*.jpg')
        self.num_samples = len(self.imgs)
        self.cam_ids, self.labels, self.frames = get_ids(
            self.imgs, self.dataset)
        self.num_cams = len(set(self.cam_ids))
        self.classes = tuple(set(self.labels))
        # Convert labels to continuous idxs
        self.class_to_idx = {label: i for i, label in enumerate(self.classes)}
        self.targets = [self.class_to_idx[label] for label in self.labels]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        sample = Image.open(self.imgs[index]).convert('RGB')
        # ToDo:
        target = self.targets[index]
        cam_id = self.cam_ids[index]
        frame = self.frames[index]

        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            target = self.target_transform(target)

        return sample, target, cam_id, frame


class ReIDDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str, st_distribution: Optional[str] = None,
                 train_subdir: str = 'bounding_box_train',
                 test_subdir: str = 'bounding_box_test',
                 query_subdir: str = 'query', train_batchsize: int = 16,
                 val_batchsize: int = 16, test_batchsize: int = 16,
                 num_workers: int = 4,
                 color_jitter: bool = False, random_erasing: float = 0.0,
                 save_distribution: Optional[str] = None):

        super().__init__()

        self.data_dir = data_dir
        self.train_dir = Path(data_dir) / train_subdir
        self.test_dir = Path(data_dir) / test_subdir
        self.query_dir = Path(data_dir) / query_subdir

        self.train_batchsize = train_batchsize
        self.test_batchsize = test_batchsize
        self.val_batchsize = val_batchsize
        self.num_workers = num_workers

        self.color_jitter = color_jitter
        self.random_erasing = random_erasing

        self.st_distribution = st_distribution
        if save_distribution == "":
            self.save_distribution = self.data_dir + '/st_distribution.pkl'
        else:
            self.save_distribution = save_distribution

        self.prepare_data()

    def prepare_data(self):
        transforms_list = [transforms.Resize((384, 192), interpolation=3),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [
                               0.229, 0.224, 0.225])
                           ]

        if self.random_erasing > 0:
            transforms_list.append(
                transforms.RandomErasing(self.random_erasing))
        if self.color_jitter:
            transforms_list.append(transforms.ColorJitter())

        transform = transforms.Compose(transforms_list)
        self.train_data = ReIDDataset(self.train_dir, transform)
        self.num_classes = len(self.train_data.classes)
        train_size = int(0.8 * self.train_data.num_samples)
        val_size = self.train_data.num_samples - train_size
        self.train_data, self.val_data = random_split(
            self.train_data, [train_size, val_size])
        self.query = ReIDDataset(self.query_dir, transform)
        self.gallery = ReIDDataset(self.test_dir, transform)

        self._load_st_distribution()
        if self.st_distribution is None:
            self._save_st_distribution()

    def _load_st_distribution(self):

        if self.st_distribution is None:
            print('\n\nGenerating Spatial-Temporal Distribution.\n\n')
            num_cams = self.query.num_cams
            max_hist = 5000 if self.query.dataset == 'market' else 3000

            cam_ids = self.query.cam_ids + self.gallery.cam_ids
            targets = self.query.targets + self.gallery.targets
            frames = self.query.frames + self.gallery.frames

            self.st_distribution = smooth_st_distribution(cam_ids, targets,
                                                          frames,
                                                          num_cams, max_hist)
        elif isinstance(self.st_distribution, str):
            if '.pkl' not in self.st_distribution:
                raise ValueError('File must be of type .pkl')

            print(
                f'\nLoading Spatial-Temporal Distribution from {self.st_distribution}.\n\n')
            self.st_distribution = joblib.load(self.st_distribution)

    def _save_st_distribution(self):
        if self.save_distribution:
            print(f'\nSaving distribution at {self.save_distribution}')
            joblib.dump(self.st_distribution, self.save_distribution)

    def train_dataloader(self):

        return DataLoader(self.train_data, batch_size=self.train_batchsize,
                          shuffle=True, num_workers=self.num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        transform = transforms.Compose([
            # Image.BICUBIC
            transforms.Resize(size=(384, 192), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.val_data.dataset.transform = transform

        return DataLoader(self.val_data, batch_size=self.val_batchsize,
                          shuffle=True, num_workers=self.num_workers,
                          pin_memory=True)

    def test_dataloader(self):
        transform = transforms.Compose([
            # Image.BICUBIC
            transforms.Resize(size=(384, 192), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.query.transform = transform
        self.gallery.transform = transform

        query_loader = DataLoader(self.query, batch_size=self.test_batchsize,
                                  shuffle=False, num_workers=self.num_workers,
                                  pin_memory=True)
        gall_loader = DataLoader(self.gallery, batch_size=self.test_batchsize,
                                 shuffle=False, num_workers=self.num_workers,
                                 pin_memory=True)

        return [query_loader, gall_loader]
