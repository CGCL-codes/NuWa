from mmdet.registry import DATASETS
from mmcv.transforms import BaseTransform 
from mmengine.registry import TRANSFORMS
from mmengine.runner import Runner

import numpy as np

@TRANSFORMS.register_module()
class MapBackLabel(BaseTransform):
    def __init__(self, sub_label_ids, task_type="detection"):
        self.sub_label_ids = sub_label_ids
        self.task_type = task_type
    def transform(self, results):
        if 'gt_bboxes_labels' in results:
            labels = results['gt_bboxes_labels']
            mapped = np.array([self.sub_label_ids[int(l)] for l in labels], dtype=np.int64)
            results['gt_bboxes_labels'] = mapped
        return results

def get_coco_dataset(root="/share/hdd/coco",sub_label=None,task_type="detection"):
    dataset_type = 'CocoDataset'
    data_root = root
    backend_args = None

    if task_type == "detection":
        train_pipeline_cfg = [
            dict(type='LoadImageFromFile', backend_args=backend_args),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(type='MapBackLabel', sub_label_ids=sub_label),
            dict(type='mmdet.PackDetInputs')
        ]
        test_pipeline_cfg = [
            dict(type='LoadImageFromFile', backend_args=backend_args),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='mmdet.PackDetInputs',meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
        ]
    elif task_type == "segmentation":
        from mmcv.transforms.loading import LoadImageFromFile
        from mmdet.datasets.transforms.formatting import PackDetInputs
        from mmdet.datasets.transforms.loading import LoadAnnotations
        from mmdet.datasets.transforms.transforms import RandomFlip, Resize
        
        train_pipeline_cfg = [
            dict(type=LoadImageFromFile, backend_args=backend_args),
            dict(type=LoadAnnotations, with_bbox=True, with_mask=True, poly2mask=False),
            dict(type=Resize, scale=(1333, 800), keep_ratio=True),
            dict(type=RandomFlip, prob=0.5),
            dict(type='MapBackLabel', sub_label_ids=sub_label),
            dict(type=PackDetInputs)
        ]
        test_pipeline_cfg = [
            dict(type=LoadImageFromFile, backend_args=backend_args),
            dict(type=Resize, scale=(1333, 800), keep_ratio=True),
            dict(type=LoadAnnotations, with_bbox=True, with_mask=True, poly2mask=False),
            dict(type=PackDetInputs,meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
        ]

    if sub_label != None:
        sub_class = [category2name[i] for i in sub_label]
    else:
        sub_class = list(category2name.values())
    train_dataset_cfg = dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='images/train2017/'),
        pipeline=train_pipeline_cfg,
        backend_args=backend_args,
        metainfo=dict(classes=sub_class),
        filter_cfg=dict(filter_empty_gt=True),
    )
    trainset = DATASETS.build(train_dataset_cfg)

    # val/test pipeline
    test_dataset_cfg = dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='images/val2017/'),
        pipeline=test_pipeline_cfg,
        backend_args=backend_args,
        metainfo=dict(classes=sub_class),
        filter_cfg=dict(filter_empty_gt=True),
    )
    testset = DATASETS.build(test_dataset_cfg)
    
    return trainset, testset

category2name = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
    80: "background"
}
