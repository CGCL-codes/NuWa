import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import logging
from mmengine.evaluator import Evaluator
from mmdet.evaluation import CocoMetric
# from mmengine.evaluator import CocoMetric

from dataset.coco import category2name

@torch.no_grad()
def evaluate(model, test_loader, visual_task = "recognition", sub_label = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print("\n\n=================================================================================")
    logger = logging.getLogger(__name__)
    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))

    if visual_task == "recognition":
        references = []
        predictions = []
        criterion = nn.CrossEntropyLoss()

        epoch_iterator = tqdm(
            test_loader,
            desc="Validating... (loss=X.X)",
            bar_format="{l_bar}{r_bar}",dynamic_ncols=True
        )

        for batch in epoch_iterator:
            xb, yb = batch[:2]
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits,yb)
            preds = logits.argmax(dim=-1)
            references.append(yb.cpu().numpy())
            predictions.append(preds.cpu().numpy())
            epoch_iterator.set_description(f"Validating... (loss={loss:.4f})")

        references = np.concatenate(references)
        predictions = np.concatenate(predictions)
        accuracy = (predictions == references).mean()

        logger.info("Validation Results")
        logger.info(f"Acc@1: {accuracy*100:2.3f}%")
        print("=================================================================================\n")
        return accuracy*100
    
    elif visual_task == "detection":
        evaluator = dict(
            type='CocoMetric',
            metric='bbox',
            format_only=False,
            backend_args=None
        )
        evaluator = Evaluator(evaluator)
        epoch_iterator = tqdm(
            test_loader,
            desc="Validating... (loss=X.X)",
            bar_format="{l_bar}{r_bar}",dynamic_ncols=True
        )
        evaluator.dataset_meta = {
            "classes":[category2name[id] for id in sub_label]
        }

        sub_label = torch.tensor(sub_label, device=device)
        lookup = torch.full((sub_label.max().item()+1,), -1, device=device)
        lookup[sub_label] = torch.arange(len(sub_label), device=device)
        
        for data_batch in epoch_iterator:
            outputs = model.test_step(data_batch)     # outputs = List[DetDataSample]
            for output in outputs:
                # pred
                pred = output.pred_instances
                if hasattr(pred, "masks"):
                    delattr(pred, "masks")
                labels = pred.labels
                indices = torch.nonzero(torch.isin(labels, sub_label), as_tuple=True)[0]
                output.pred_instances = pred[indices]
                output.pred_instances.labels = lookup[output.pred_instances.labels]
                # ground truth
                output.instances = []
                for i in range(output.gt_instances.labels.shape[0]):
                    output.instances.append({
                        "bbox": output.gt_instances.bboxes[i].cpu().numpy(),
                        "bbox_label": output.gt_instances.labels[i].cpu().numpy(),
                    })
                delattr(output, "gt_instances")
            evaluator.process(outputs,data_batch)
        results = evaluator.evaluate(len(test_loader.dataset))
        if not results:
            return 0.0
        return results["coco/bbox_mAP"]*100

    elif visual_task == "segmentation":
        from mmdet.evaluation.metrics.coco_metric import CocoMetric
        from pycocotools import mask as maskUtils
        
        evaluator = dict(
            type=CocoMetric,
            metric=['bbox', 'segm'],
            format_only=False,
            backend_args=None
        )
        evaluator = Evaluator(evaluator)
        epoch_iterator = tqdm(
            test_loader,
            desc="Validating... (loss=X.X)",
            bar_format="{l_bar}{r_bar}",dynamic_ncols=True
        )
        evaluator.dataset_meta = {
            "classes":[category2name[id] for id in sub_label]
        }

        sub_label = torch.tensor(sub_label, device=device)
        lookup = torch.full((sub_label.max().item()+1,), -1, device=device)
        lookup[sub_label] = torch.arange(len(sub_label), device=device)
        
        for data_batch in epoch_iterator:
            outputs = model.test_step(data_batch)     # outputs = List[DetDataSample]
            for output in outputs:
                # pred
                pred = output.pred_instances
                labels = pred.labels
                indices = torch.nonzero(torch.isin(labels, sub_label), as_tuple=True)[0]
                output.pred_instances = pred[indices]
                output.pred_instances.labels = lookup[output.pred_instances.labels]
                # ground truth
                output.instances = []
                for i in range(output.gt_instances.labels.shape[0]):
                    output.instances.append({
                        "bbox": output.gt_instances.bboxes[i].cpu().numpy(),
                        "bbox_label": output.gt_instances.labels[i].cpu().numpy(),
                        "mask": output.gt_instances.masks.masks[i],
                    })
                delattr(output, "gt_instances")
            evaluator.process(outputs,data_batch)
        results = evaluator.evaluate(len(test_loader.dataset))
        if not results:
            return {
                "bbox_mAP": 0.0,
                "segm_mAP": 0.0,
            }
        print(f"Detection: {results['coco/bbox_mAP']*100:.1f}")
        print(f"Segmentation: {results['coco/segm_mAP']*100:.1f}")
        return {
            "bbox_mAP": results['coco/bbox_mAP']*100,
            "segm_mAP": results['coco/segm_mAP']*100,
        }
