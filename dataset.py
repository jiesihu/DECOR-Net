# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import collections.abc
import math
import pickle
import shutil
import sys
import tempfile
import threading
import time
import warnings
from copy import copy, deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.serialization import DEFAULT_PROTOCOL
from torch.utils.data import Dataset as _TorchDataset
from torch.utils.data import Subset

from monai.data.utils import SUPPORTED_PICKLE_MOD, convert_tables_to_dicts, pickle_hashing
from monai.transforms import Compose, Randomizable, ThreadUnsafe, Transform, apply_transform, convert_to_contiguous
from monai.utils import MAX_SEED, deprecated_arg, get_seed, look_up_option, min_version, optional_import
from monai.utils.misc import first

if TYPE_CHECKING:
    from tqdm import tqdm

    has_tqdm = True
else:
    tqdm, has_tqdm = optional_import("tqdm", "4.47.0", min_version, "tqdm")

lmdb, _ = optional_import("lmdb")
pd, _ = optional_import("pandas")


class Dataset(_TorchDataset):
    """
    A generic dataset with a length property and an optional callable data transform
    when fetching a data sample.
    If passing slicing indices, will return a PyTorch Subset, for example: `data: Subset = dataset[1:4]`,
    for more details, please check: https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset

    For example, typical input data can be a list of dictionaries::

        [{                            {                            {
             'img': 'image1.nii.gz',      'img': 'image2.nii.gz',      'img': 'image3.nii.gz',
             'seg': 'label1.nii.gz',      'seg': 'label2.nii.gz',      'seg': 'label3.nii.gz',
             'extra': 123                 'extra': 456                 'extra': 789
         },                           },                           }]
    """

    def __init__(self, data: Sequence, transform: Optional[Callable] = None) -> None:
        """
        Args:
            data: input data to load and transform to generate dataset for model.
            transform: a callable data transform on input data.

        """
        self.data = data
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def _transform(self, index: int):
        """
        Fetch single data item from `self.data`.
        """
        data_i = self.data[index]
        return apply_transform(self.transform, data_i) if self.transform is not None else data_i

    def __getitem__(self, index: Union[int, slice, Sequence[int]]):
        """
        Returns a `Subset` if `index` is a slice or Sequence, a data item otherwise.
        """
        if isinstance(index, slice):
            # dataset[:42]
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)
        if isinstance(index, collections.abc.Sequence):
            # dataset[[1, 3, 4]]
            return Subset(dataset=self, indices=index)
        return self._transform(index)


class DatasetFunc(Dataset):
    """
    Execute function on the input dataset and leverage the output to act as a new Dataset.
    It can be used to load / fetch the basic dataset items, like the list of `image, label` paths.
    Or chain together to execute more complicated logic, like `partition_dataset`, `resample_datalist`, etc.
    The `data` arg of `Dataset` will be applied to the first arg of callable `func`.
    Usage example::

        data_list = DatasetFunc(
            data="path to file",
            func=monai.data.load_decathlon_datalist,
            data_list_key="validation",
            base_dir="path to base dir",
        )
        # partition dataset for every rank
        data_partition = DatasetFunc(
            data=data_list,
            func=lambda **kwargs: monai.data.partition_dataset(**kwargs)[torch.distributed.get_rank()],
            num_partitions=torch.distributed.get_world_size(),
        )
        dataset = Dataset(data=data_partition, transform=transforms)

    Args:
        data: input data for the func to process, will apply to `func` as the first arg.
        func: callable function to generate dataset items.
        kwargs: other arguments for the `func` except for the first arg.

    """

    def __init__(self, data: Any, func: Callable, **kwargs) -> None:
        super().__init__(data=None, transform=None)  # type:ignore
        self.src = data
        self.func = func
        self.kwargs = kwargs
        self.reset()

    def reset(self, data: Optional[Any] = None, func: Optional[Callable] = None, **kwargs):
        """
        Reset the dataset items with specified `func`.

        Args:
            data: if not None, execute `func` on it, default to `self.src`.
            func: if not None, execute the `func` with specified `kwargs`, default to `self.func`.
            kwargs: other arguments for the `func` except for the first arg.

        """
        src = self.src if data is None else data
        self.data = self.func(src, **self.kwargs) if func is None else func(src, **kwargs)



class ZipDataset(Dataset):
    """
    Zip several PyTorch datasets and output data(with the same index) together in a tuple.
    If the output of single dataset is already a tuple, flatten it and extend to the result.
    For example: if datasetA returns (img, imgmeta), datasetB returns (seg, segmeta),
    finally return (img, imgmeta, seg, segmeta).
    And if the datasets don't have same length, use the minimum length of them as the length
    of ZipDataset.
    If passing slicing indices, will return a PyTorch Subset, for example: `data: Subset = dataset[1:4]`,
    for more details, please check: https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset

    Examples::

        >>> zip_data = ZipDataset([[1, 2, 3], [4, 5]])
        >>> print(len(zip_data))
        2
        >>> for item in zip_data:
        >>>    print(item)
        [1, 4]
        [2, 5]

    """

    def __init__(self, datasets: Sequence, transform: Optional[Callable] = None) -> None:
        """
        Args:
            datasets: list of datasets to zip together.
            transform: a callable data transform operates on the zipped item from `datasets`.
        """
        super().__init__(list(datasets), transform=transform)

    def __len__(self) -> int:
        return min(len(dataset) for dataset in self.data)

    def _transform(self, index: int):
        def to_list(x):
            return list(x) if isinstance(x, (tuple, list)) else [x]

        data = []
        for dataset in self.data:
            data.extend(to_list(dataset[index]))
        if self.transform is not None:
            data = apply_transform(self.transform, data, map_items=False)  # transform the list data
        # use tuple instead of list as the default collate_fn callback of MONAI DataLoader flattens nested lists
        return tuple(data)


class ArrayDataset(Randomizable, _TorchDataset):
    """
    Dataset for segmentation and classification tasks based on array format input data and transforms.
    It ensures the same random seeds in the randomized transforms defined for image, segmentation and label.
    The `transform` can be :py:class:`monai.transforms.Compose` or any other callable object.
    For example:
    If train based on Nifti format images without metadata, all transforms can be composed::

        img_transform = Compose(
            [
                LoadImage(image_only=True),
                AddChannel(),
                RandAdjustContrast()
            ]
        )
        ArrayDataset(img_file_list, img_transform=img_transform)

    If training based on images and the metadata, the array transforms can not be composed
    because several transforms receives multiple parameters or return multiple values. Then Users need
    to define their own callable method to parse metadata from `LoadImage` or set `affine` matrix
    to `Spacing` transform::

        class TestCompose(Compose):
            def __call__(self, input_):
                img, metadata = self.transforms[0](input_)
                img = self.transforms[1](img)
                img, _, _ = self.transforms[2](img, metadata["affine"])
                return self.transforms[3](img), metadata
        img_transform = TestCompose(
            [
                LoadImage(image_only=False),
                AddChannel(),
                Spacing(pixdim=(1.5, 1.5, 3.0)),
                RandAdjustContrast()
            ]
        )
        ArrayDataset(img_file_list, img_transform=img_transform)

    Examples::

        >>> ds = ArrayDataset([1, 2, 3, 4], lambda x: x + 0.1)
        >>> print(ds[0])
        1.1

        >>> ds = ArrayDataset(img=[1, 2, 3, 4], seg=[5, 6, 7, 8])
        >>> print(ds[0])
        [1, 5]

    """

    def __init__(
        self,
        img: Sequence,
        img_transform: Optional[Callable] = None,
        seg: Optional[Sequence] = None,
        seg_transform: Optional[Callable] = None,
        labels: Optional[Sequence] = None,
        label_transform: Optional[Callable] = None,
    ) -> None:
        """
        Initializes the dataset with the filename lists. The transform `img_transform` is applied
        to the images and `seg_transform` to the segmentations.

        Args:
            img: sequence of images.
            img_transform: transform to apply to each element in `img`.
            seg: sequence of segmentations.
            seg_transform: transform to apply to each element in `seg`.
            labels: sequence of labels.
            label_transform: transform to apply to each element in `labels`.

        """
        items = [(img, img_transform), (seg, seg_transform), (labels, label_transform)]
        self.set_random_state(seed=get_seed())
        self.img = img
        datasets = [Dataset(x[0], x[1]) for x in items if x[0] is not None]
        self.dataset = datasets[0] if len(datasets) == 1 else ZipDataset(datasets)

        self._seed = 0  # transform synchronization seed

    def __len__(self) -> int:
        return len(self.dataset)

    def randomize(self, data: Optional[Any] = None) -> None:
        self._seed = self.R.randint(MAX_SEED, dtype="uint32")

    def __getitem__(self, index: int):
        self.randomize()
        if isinstance(self.dataset, ZipDataset):
            # set transforms of each zip component
            for dataset in self.dataset.data:
                transform = getattr(dataset, "transform", None)
                if isinstance(transform, Randomizable):
                    transform.set_random_state(seed=self._seed)
        transform = getattr(self.dataset, "transform", None)
        if isinstance(transform, Randomizable):
            transform.set_random_state(seed=self._seed)
        return self.dataset[index]


class ArrayDataset_name(Randomizable, _TorchDataset):
    """
    Jiesi: Let's also output the name!
    
    Dataset for segmentation and classification tasks based on array format input data and transforms.
    It ensures the same random seeds in the randomized transforms defined for image, segmentation and label.
    The `transform` can be :py:class:`monai.transforms.Compose` or any other callable object.
    For example:
    If train based on Nifti format images without metadata, all transforms can be composed::

        img_transform = Compose(
            [
                LoadImage(image_only=True),
                AddChannel(),
                RandAdjustContrast()
            ]
        )
        ArrayDataset(img_file_list, img_transform=img_transform)

    If training based on images and the metadata, the array transforms can not be composed
    because several transforms receives multiple parameters or return multiple values. Then Users need
    to define their own callable method to parse metadata from `LoadImage` or set `affine` matrix
    to `Spacing` transform::

        class TestCompose(Compose):
            def __call__(self, input_):
                img, metadata = self.transforms[0](input_)
                img = self.transforms[1](img)
                img, _, _ = self.transforms[2](img, metadata["affine"])
                return self.transforms[3](img), metadata
        img_transform = TestCompose(
            [
                LoadImage(image_only=False),
                AddChannel(),
                Spacing(pixdim=(1.5, 1.5, 3.0)),
                RandAdjustContrast()
            ]
        )
        ArrayDataset(img_file_list, img_transform=img_transform)

    Examples::

        >>> ds = ArrayDataset([1, 2, 3, 4], lambda x: x + 0.1)
        >>> print(ds[0])
        1.1

        >>> ds = ArrayDataset(img=[1, 2, 3, 4], seg=[5, 6, 7, 8])
        >>> print(ds[0])
        [1, 5]

    """

    def __init__(
        self,
        img: Sequence,
        img_transform: Optional[Callable] = None,
        seg: Optional[Sequence] = None,
        seg_transform: Optional[Callable] = None,
        labels: Optional[Sequence] = None,
        label_transform: Optional[Callable] = None,
    ) -> None:
        """
        Initializes the dataset with the filename lists. The transform `img_transform` is applied
        to the images and `seg_transform` to the segmentations.

        Args:
            img: sequence of images.
            img_transform: transform to apply to each element in `img`.
            seg: sequence of segmentations.
            seg_transform: transform to apply to each element in `seg`.
            labels: sequence of labels.
            label_transform: transform to apply to each element in `labels`.

        """
        items = [(img, img_transform), (seg, seg_transform), (labels, label_transform)]
        self.set_random_state(seed=get_seed())
        self.img = img
        datasets = [Dataset(x[0], x[1]) for x in items if x[0] is not None]
        self.dataset = datasets[0] if len(datasets) == 1 else ZipDataset(datasets)

        self._seed = 0  # transform synchronization seed

    def __len__(self) -> int:
        return len(self.dataset)

    def randomize(self, data: Optional[Any] = None) -> None:
        self._seed = self.R.randint(MAX_SEED, dtype="uint32")

    def __getitem__(self, index: int):
        self.randomize()
        if isinstance(self.dataset, ZipDataset):
            # set transforms of each zip component
            for dataset in self.dataset.data:
                transform = getattr(dataset, "transform", None)
                if isinstance(transform, Randomizable):
                    transform.set_random_state(seed=self._seed)
        transform = getattr(self.dataset, "transform", None)
        if isinstance(transform, Randomizable):
            transform.set_random_state(seed=self._seed)
        return list(self.dataset[index])+[self.img[index]]


class ArrayDataset_name_Y(Randomizable, _TorchDataset):
    """
    Jiesi: 
    Let's also output the name!
    Output Y for loss function(not implemented)

    """

    def __init__(
        self,
        img: Sequence,
        img_transform: Optional[Callable] = None,
        seg: Optional[Sequence] = None,
        seg_transform: Optional[Callable] = None,
        labels: Optional[Sequence] = None,
        label_transform: Optional[Callable] = None,
        Y_dict: Optional[Sequence] = None,
    ) -> None:
        """
        Initializes the dataset with the filename lists. The transform `img_transform` is applied
        to the images and `seg_transform` to the segmentations.

        Args:
            img: sequence of images.
            img_transform: transform to apply to each element in `img`.
            seg: sequence of segmentations.
            seg_transform: transform to apply to each element in `seg`.
            labels: sequence of labels.
            label_transform: transform to apply to each element in `labels`.

        """
        items = [(img, img_transform), (seg, seg_transform), (labels, label_transform)]
        self.set_random_state(seed=get_seed())
        self.img = img
        datasets = [Dataset(x[0], x[1]) for x in items if x[0] is not None]
        self.dataset = datasets[0] if len(datasets) == 1 else ZipDataset(datasets)

        self._seed = 0  # transform synchronization seed
        self.Y_dict = Y_dict

    def __len__(self) -> int:
        return len(self.dataset)

    def randomize(self, data: Optional[Any] = None) -> None:
        self._seed = self.R.randint(MAX_SEED, dtype="uint32")

    def __getitem__(self, index: int):
        self.randomize()
        if isinstance(self.dataset, ZipDataset):
            # set transforms of each zip component
            for dataset in self.dataset.data:
                transform = getattr(dataset, "transform", None)
                if isinstance(transform, Randomizable):
                    transform.set_random_state(seed=self._seed)
        transform = getattr(self.dataset, "transform", None)
        if isinstance(transform, Randomizable):
            transform.set_random_state(seed=self._seed)
        slice_name = self.img[index].split('/')[-1]
        return list(self.dataset[index])+[self.img[index]]+[self.Y_dict[slice_name]]

class ArrayDataset_GLCM(Randomizable, _TorchDataset):
    def __init__(
        self,
        img: Sequence,
        img_transform: Optional[Callable] = None,
        seg: Optional[Sequence] = None,
        seg_transform: Optional[Callable] = None,
        texture: Optional[Sequence] = None,
        texture_transform: Optional[Callable] = None,
    ) -> None:
        self.img__ = img
        items = [(img, img_transform), (seg, seg_transform), (texture, texture_transform)]
        self.set_random_state(seed=get_seed())
        datasets = [Dataset(x[0], x[1]) for x in items if x[0] is not None]
        self.dataset = datasets[0] if len(datasets) == 1 else ZipDataset(datasets)

        self._seed = 0  # transform synchronization seed

    def __len__(self) -> int:
        return len(self.dataset)

    def randomize(self, data: Optional[Any] = None) -> None:
        self._seed = self.R.randint(MAX_SEED, dtype="uint32")

    def __getitem__(self, index: int):
        self.randomize()
        if isinstance(self.dataset, ZipDataset):
            # set transforms of each zip component
            for dataset in self.dataset.data:
                transform = getattr(dataset, "transform", None)
                if isinstance(transform, Randomizable):
                    transform.set_random_state(seed=self._seed)
        transform = getattr(self.dataset, "transform", None)
        if isinstance(transform, Randomizable):
            transform.set_random_state(seed=self._seed)
        return list(self.dataset[index])+[self.img__[index]]
