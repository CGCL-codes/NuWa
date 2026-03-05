from mmdet.apis import init_detector

def mask_rcnn_swin_t():
    config = '/root/autodl-pvt/nuwa/mmdetection/configs/swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py'
    checkpoint = '/root/autodl-pvt/nuwa/model/weights/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth'
    model = init_detector(config, checkpoint)
    return model