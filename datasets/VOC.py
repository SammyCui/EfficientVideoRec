import os
from typing import Optional, Iterable, Callable, Any, List, Tuple, Dict, cast
from torchvision.datasets.folder import default_loader, has_file_allowed_extension
import torch
import torchvision
from PIL import Image, ImageDraw
from datasets.data_utils import get_anno_stats
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.common_types import _size_2_t
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

"""
This is the dataset module corresponds to torchvision==0.6.0, as on erdos cluster
"""


class T(torchvision.datasets.VisionDataset):
    def __init__(self,
                 root: str,
                 anno_root: str,
                 cls_to_use: Optional[Iterable[str]] = None,
                 num_classes: Optional[int] = None,
                 transform: Optional[Callable] = None,
                 per_size: _size_2_t = None,
                 target_transform: Optional[Callable] = None,
                 loader: Callable[[str], Any] = default_loader,
                 is_valid_file: Optional[Callable[[str], bool]] = None,):
        super().__init__(root, transform=transform,
                         target_transform=target_transform)
        self.cls_to_use = cls_to_use
        self.loader = loader
        self.is_valid_file = is_valid_file
        self.num_classes = num_classes
        self.anno_root = anno_root
        classes, class_to_idx = self._find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, IMG_EXTENSIONS if is_valid_file is None else None,
                                    is_valid_file)
        # if len(samples) == 0:
        #     raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
        #                                                                          "Supported extensions are: " + ",".join(
        #         IMG_EXTENSIONS)))
        # self.classes = classes
        # self.class_to_idx = class_to_idx
        # self.samples = samples
        # self.targets = [s[1] for s in samples]
        self.per_size = per_size
        self.x=1

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, bndbox, target = self.samples[index]
        if self.loader:
            sample = self.loader(path)
        else:
            sample = Image.open(path)
        w, h = sample.size
        if self.transform is not None:
            sample = self.transform(sample)
            if self.per_size:
                if isinstance(sample, Iterable):
                    transformed_h, transformed_w = sample.shape[1], sample.shape[2]
                else:
                    transformed_w, transformed_h = sample.size # PIL image size: (w, h)
                if transformed_w != w or transformed_h != h: # if the image is resized, the bndbox loc has to be resized too
                    x_scalar, y_scalar = transformed_w / w, transformed_h / h
                    x_min, y_min, x_max, y_max = bndbox
                    new_x_min, new_x_max = x_min * x_scalar, x_max * x_scalar
                    new_y_min, new_y_max = y_min * y_scalar, y_max * y_scalar
                    bndbox = round(new_x_min), round(new_y_min), round(new_x_max), round(new_y_max)
        if bndbox:
            bndbox = sample[..., bndbox[1]:bndbox[3], bndbox[0]:bndbox[2]]
            # if the bounding box is too small: < 9, pad it to >=9
            if bndbox.shape[1] < 9:
                padding = int(np.ceil(9 - bndbox.shape[1]) / 2)
                bndbox = F.pad(bndbox, (0, 0, padding, padding))
            if bndbox.shape[2] < 9:
                padding = int(np.ceil(9 - bndbox.shape[2]) / 2)
                bndbox = F.pad(bndbox, (padding, padding, 0, 0))
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.per_size:
            sample = F.interpolate(sample, size=self.per_size, align_corners=False)
        return sample, bndbox, target

    def _find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Modified version of torchvision.datasets.folder.find_classes
        to select a subset of image classes
        """
        if self.cls_to_use is not None:
            classes = sorted(
                entry.name for entry in os.scandir(directory) if entry.is_dir() and entry.name in self.cls_to_use)
        else:
            classes = sorted(
                entry.name for entry in os.scandir(directory) if entry.is_dir())
            if self.num_classes:
                classes = classes[:self.num_classes]
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def make_dataset(
            self,
            directory: str,
            class_to_idx: Dict[str, int],
            extensions: Optional[Tuple[str, ...]] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, Tuple[int, int, int, int], int]]:
        directory = os.path.expanduser(directory)

        if class_to_idx is None:
            _, class_to_idx = self._find_classes(directory)
        elif not class_to_idx:
            raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:
            def is_valid_file(x: str) -> bool:
                return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        instances = []
        available_classes = set()
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            count = 0
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        img_name = fname.split('.')[0] # without .jpg
                        if self.per_size:
                            x_min, y_min, x_max, y_max, img_w, img_h = get_anno_stats(os.path.join(self.anno_root,
                                                                                                         target_class,
                                                                                                         img_name+'.xml'))
                            item = path, (x_min, y_min, x_max, y_max), class_index
                        else:
                            item = path, None, class_index
                        instances.append(item)
                        count += 1

                        if target_class not in available_classes:
                            available_classes.add(target_class)

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
            raise FileNotFoundError(msg)

        return instances


class VOCDataset(torchvision.datasets.VisionDataset):

    def __init__(self,
                 root: str,
                 anno_root: str,
                 cls_to_use: Optional[Iterable[str]] = None,
                 num_classes: Optional[int] = None,
                 transform: Optional[Callable] = None,
                 per_size: _size_2_t = None,
                 target_transform: Optional[Callable] = None,
                 loader: Callable[[str], Any] = default_loader,
                 is_valid_file: Optional[Callable[[str], bool]] = None,
                 ):
        super().__init__(root, transform=transform,
                         target_transform=target_transform)
        self.cls_to_use = cls_to_use
        self.loader = loader
        self.is_valid_file = is_valid_file
        self.num_classes = num_classes
        self.anno_root = anno_root
        self.per_size = per_size
        classes, class_to_idx = self._find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, IMG_EXTENSIONS if is_valid_file is None else None,
                                    is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                                                                 "Supported extensions are: " + ",".join(
                IMG_EXTENSIONS)))
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, bndbox, target = self.samples[index]

        if self.loader:
            sample = self.loader(path)
        else:
            sample = Image.open(path)
        if self.per_size:
            bndbox = sample.crop(bndbox)
        w, h = sample.size
        if self.transform is not None:
            if self.per_size:
                bndbox = self.transform(bndbox)
            sample = self.transform(sample)
        #     if self.per_size:
        #         if isinstance(sample, Iterable):
        #             transformed_h, transformed_w = sample.shape[1], sample.shape[2]
        #         else:
        #             transformed_w, transformed_h = sample.size # PIL image size: (w, h)
        #         if transformed_w != w or transformed_h != h: # if the image is resized, the bndbox loc has to be resized too
        #             x_scalar, y_scalar = transformed_w / w, transformed_h / h
        #             x_min, y_min, x_max, y_max = bndbox
        #             new_x_min, new_x_max = x_min * x_scalar, x_max * x_scalar
        #             new_y_min, new_y_max = y_min * y_scalar, y_max * y_scalar
        #             bndbox = round(new_x_min), round(new_y_min), round(new_x_max), round(new_y_max)
        # if bndbox:
        #     bndbox = sample[..., bndbox[1]:bndbox[3], bndbox[0]:bndbox[2]]
        #     # if the bounding box is too small: < 9, pad it to >=9
        #     if bndbox.shape[1] < 9:
        #         padding = int(np.ceil(9 - bndbox.shape[1]) / 2)
        #         bndbox = F.pad(bndbox, (0, 0, padding, padding))
        #     if bndbox.shape[2] < 9:
        #         padding = int(np.ceil(9 - bndbox.shape[2]) / 2)
        #         bndbox = F.pad(bndbox, (padding, padding, 0, 0))
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.per_size:
            sample = F.interpolate(sample.unsqueeze(0), size=self.per_size, mode='bilinear').squeeze(0)
        if self.per_size:
            return sample, bndbox, target
        else:
            return sample, target

    def _find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Modified version of torchvision.datasets.folder.find_classes
        to select a subset of image classes
        """
        if self.cls_to_use is not None:
            classes = sorted(
                entry.name for entry in os.scandir(directory) if entry.is_dir() and entry.name in self.cls_to_use)
        else:
            classes = sorted(
                entry.name for entry in os.scandir(directory) if entry.is_dir())
            if self.num_classes:
                classes = classes[:self.num_classes]
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def make_dataset(
            self,
            directory: str,
            class_to_idx: Dict[str, int],
            extensions: Optional[Tuple[str, ...]] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, Tuple[int, int, int, int], int]]:
        directory = os.path.expanduser(directory)

        if class_to_idx is None:
            _, class_to_idx = self._find_classes(directory)
        elif not class_to_idx:
            raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:
            def is_valid_file(x: str) -> bool:
                return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        instances = []
        available_classes = set()
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            count = 0
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        img_name = fname.split('.')[0] # without .jpg
                        if self.per_size:
                            x_min, y_min, x_max, y_max, img_w, img_h = get_anno_stats(os.path.join(self.anno_root,
                                                                                                         target_class,
                                                                                                         img_name+'.xml'))
                            item = path, (x_min, y_min, x_max, y_max), class_index
                        else:
                            item = path, torch.tensor(0), class_index
                        instances.append(item)
                        count += 1

                        if target_class not in available_classes:
                            available_classes.add(target_class)

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
            raise FileNotFoundError(msg)

        return instances



if __name__ == '__main__':



    dir = '/Users/xuanmingcui/Documents/projects/cnslab/cnslab/SequentialTraining/datasets/VOC2012_filtered/train'

    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((256,256)), torchvision.transforms.ToTensor()])

    dataset = VOCDataset(root=os.path.join(dir,'root'), anno_root=os.path.join(dir, 'annotations'), transform=transform, per_size=None)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)
    for img, bb, target in dataloader:
        t = torchvision.transforms.ToPILImage()
        t(img[0]).show()
        print(target)
