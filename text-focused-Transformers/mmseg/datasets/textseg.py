import os
import os.path as osp
from functools import reduce

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable
from torch.utils.data import Dataset

from mmseg.core import eval_metrics
from mmseg.utils import get_root_logger
from .pipelines import Compose, LoadAnnotations_textseg, LoadAnnotations_icdar13, LoadAnnotations_totaltext, \
    LoadAnnotations_BTS, LoadAnnotations_mlt
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class TextSegDataset(CustomDataset):
    """ADE20K dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    CLASSES = ('background', 'text')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super(TextSegDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_maskfg.png',
            reduce_zero_label=False,
            **kwargs)
        self.gt_seg_map_loader = LoadAnnotations_textseg()  # can be removed


@DATASETS.register_module()
class BTSDataset_Det(CustomDataset):
    """ADE20K dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    CLASSES = ('background', 'text')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super(BTSDataset_Det, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            det_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        self.gt_seg_map_loader = LoadAnnotations_BTS()

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        results['det_prefix'] = self.det_dir
        if self.custom_classes:
            results['label_map'] = self.label_map

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split, det_dir=None, det_suffix=None):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            lines = mmcv.list_from_file(
                split, file_client_args=self.file_client_args)
            for line in lines:
                img_name = line.strip()
                img_info = dict(filename=img_name + img_suffix)
                if ann_dir is not None:
                    seg_map = img_name + seg_map_suffix
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
        else:
            for img in self.file_client.list_dir_or_file(
                    dir_path=img_dir,
                    list_dir=False,
                    suffix=img_suffix,
                    recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                if det_dir is not None:
                    det = img.replace(img_suffix, det_suffix)
                    img_info['ann']['det'] = det
                img_infos.append(img_info)
            img_infos = sorted(img_infos, key=lambda x: x['filename'])

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        # {'filename': 'd00571.jpg', 'ann': {'seg_map': 'd00571_maskfg.png', 'det': 'd00571.jpg'}},
        return img_infos


@DATASETS.register_module()
class TextSegDataset_Det(CustomDataset):
    """ADE20K dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    CLASSES = ('background', 'text')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super(TextSegDataset_Det, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_maskfg.png',
            det_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        self.gt_seg_map_loader = LoadAnnotations_textseg()

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        results['det_prefix'] = self.det_dir
        if self.custom_classes:
            results['label_map'] = self.label_map

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split, det_dir=None, det_suffix=None):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            lines = mmcv.list_from_file(
                split, file_client_args=self.file_client_args)
            for line in lines:
                img_name = line.strip()
                img_info = dict(filename=img_name + img_suffix)
                if ann_dir is not None:
                    seg_map = img_name + seg_map_suffix
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
        else:
            for img in self.file_client.list_dir_or_file(
                    dir_path=img_dir,
                    list_dir=False,
                    suffix=img_suffix,
                    recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                if det_dir is not None:
                    det = img.replace(img_suffix, det_suffix)
                    img_info['ann']['det'] = det
                img_infos.append(img_info)
            img_infos = sorted(img_infos, key=lambda x: x['filename'])

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        # {'filename': 'd00571.jpg', 'ann': {'seg_map': 'd00571_maskfg.png', 'det': 'd00571.jpg'}},
        return img_infos


@DATASETS.register_module()
class TotaltextDataset(CustomDataset):
    """ADE20K dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    CLASSES = ('background', 'text')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super(TotaltextDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.jpg',
            reduce_zero_label=False,
            **kwargs)
        self.gt_seg_map_loader = LoadAnnotations_totaltext()


@DATASETS.register_module()
class ICDAR13Dataset(CustomDataset):
    """ADE20K dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    CLASSES = ('background', 'text')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super(ICDAR13Dataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_GT.bmp',
            reduce_zero_label=False,
            **kwargs)
        self.gt_seg_map_loader = LoadAnnotations_icdar13()

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            lines = mmcv.list_from_file(
                split, file_client_args=self.file_client_args)
            for line in lines:
                img_name = line.strip()
                img_info = dict(filename=img_name + img_suffix)
                if ann_dir is not None:
                    seg_map = img_name + seg_map_suffix
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
        else:
            for img in self.file_client.list_dir_or_file(
                    dir_path=img_dir,
                    list_dir=False,
                    suffix=img_suffix,
                    recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    fn = img.split('.')[0]
                    possible_gt_name = [
                        'gt_{}.png'.format(fn),
                        '{}_GT.bmp'.format(fn),
                    ]
                    seg_map = None
                    for pi in possible_gt_name:
                        pi_full = os.path.join(ann_dir, pi)
                        if self.file_client.exists(pi_full):
                            # if os.path.exists(pi_full):
                            seg_map = pi_full
                    if seg_map is None:
                        raise ValueError('seg_map not found for {}'.format(pi_full))
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
            img_infos = sorted(img_infos, key=lambda x: x['filename'])

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos


@DATASETS.register_module()
class MLTSDataset(CustomDataset):
    """ADE20K dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    CLASSES = ('background', 'text')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super(MLTSDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            lines = mmcv.list_from_file(
                split, file_client_args=self.file_client_args)
            for line in lines:
                img_name = line.strip()
                img_info = dict(filename=img_name + img_suffix)
                if ann_dir is not None:
                    seg_map = img_name + seg_map_suffix
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
        else:
            for img in self.file_client.list_dir_or_file(
                    dir_path=img_dir,
                    list_dir=False,
                    suffix=img_suffix,
                    recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                img = os.path.join(img_dir, img)
                seg_map = os.path.join(ann_dir, seg_map)
                # print(img_info, img)
                # print(self.file_client.exists(seg_map) , self.file_client.exists(img))
                if self.file_client.exists(seg_map) and self.file_client.exists(img):
                    img_infos.append(img_info)
            img_infos = sorted(img_infos, key=lambda x: x['filename'])

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos


@DATASETS.register_module()
class TotaltextDataset_Det(CustomDataset):
    """ADE20K dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    CLASSES = ('background', 'text')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super(TotaltextDataset_Det, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.jpg',
            det_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        self.gt_seg_map_loader = LoadAnnotations_totaltext()

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        results['det_prefix'] = self.det_dir
        if self.custom_classes:
            results['label_map'] = self.label_map

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split, det_dir=None, det_suffix=None):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            lines = mmcv.list_from_file(
                split, file_client_args=self.file_client_args)
            for line in lines:
                img_name = line.strip()
                img_info = dict(filename=img_name + img_suffix)
                if ann_dir is not None:
                    seg_map = img_name + seg_map_suffix
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
        else:
            for img in self.file_client.list_dir_or_file(
                    dir_path=img_dir,
                    list_dir=False,
                    suffix=img_suffix,
                    recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                if det_dir is not None:
                    det = img.replace(img_suffix, det_suffix)
                    img_info['ann']['det'] = det
                img_infos.append(img_info)
            img_infos = sorted(img_infos, key=lambda x: x['filename'])

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos


@DATASETS.register_module()
class MLTSDataset_Det(CustomDataset):
    """ADE20K dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    CLASSES = ('background', 'text')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super(MLTSDataset_Det, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            det_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        self.gt_seg_map_loader = LoadAnnotations_mlt()

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        results['det_prefix'] = self.det_dir
        if self.custom_classes:
            results['label_map'] = self.label_map

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split, det_dir=None, det_suffix=None):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            lines = mmcv.list_from_file(
                split, file_client_args=self.file_client_args)
            for line in lines:
                img_name = line.strip()
                img_info = dict(filename=img_name + img_suffix)
                if ann_dir is not None:
                    seg_map = img_name + seg_map_suffix
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
        else:
            for img in self.file_client.list_dir_or_file(
                    dir_path=img_dir,
                    list_dir=False,
                    suffix=img_suffix,
                    recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                if det_dir is not None:
                    det = img.replace(img_suffix, det_suffix)
                    img_info['ann']['det'] = det
                img = os.path.join(img_dir, img)
                seg_map = os.path.join(ann_dir, seg_map)
                # print(img_info, img)
                # print(self.file_client.exists(seg_map) , self.file_client.exists(img))
                if self.file_client.exists(seg_map) and self.file_client.exists(img):
                    img_infos.append(img_info)
            img_infos = sorted(img_infos, key=lambda x: x['filename'])

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos


@DATASETS.register_module()
class ICDAR13Dataset_Det(CustomDataset):
    """ADE20K dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    CLASSES = ('background', 'text')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super(ICDAR13Dataset_Det, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_GT.bmp',
            det_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        self.gt_seg_map_loader = LoadAnnotations_icdar13()

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        results['det_prefix'] = self.det_dir
        if self.custom_classes:
            results['label_map'] = self.label_map

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split, det_dir=None, det_suffix=None):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            lines = mmcv.list_from_file(
                split, file_client_args=self.file_client_args)
            for line in lines:
                img_name = line.strip()
                img_info = dict(filename=img_name + img_suffix)
                if ann_dir is not None:
                    seg_map = img_name + seg_map_suffix
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
        else:
            for img in self.file_client.list_dir_or_file(
                    dir_path=img_dir,
                    list_dir=False,
                    suffix=img_suffix,
                    recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    fn = img.split('.')[0]
                    possible_gt_name = [
                        'gt_{}.png'.format(fn),
                        '{}_GT.bmp'.format(fn),
                    ]
                    seg_map = None
                    for pi in possible_gt_name:
                        pi_full = os.path.join(ann_dir, pi)
                        if self.file_client.exists(pi_full):
                            # if os.path.exists(pi_full):
                            seg_map = pi_full
                    if seg_map is None:
                        raise ValueError('seg_map not found for {}'.format(pi_full))
                    img_info['ann'] = dict(seg_map=seg_map)
                if det_dir is not None:
                    det = img.replace(img_suffix, det_suffix)
                    img_info['ann']['det'] = det
                img_infos.append(img_info)
            img_infos = sorted(img_infos, key=lambda x: x['filename'])

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos