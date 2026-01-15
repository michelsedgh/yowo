"""
Smart Home Evaluator

Evaluator for Smart Home model with 42 filtered actions.
Wraps CharadesAGEvaluator with remapped action indices.
"""

import os
import time
import numpy as np
import torch
from collections import defaultdict

from dataset.smart_home import SmartHomeDataset
from dataset.transforms import BaseTransform
from utils.box_ops import rescale_bboxes


class SmartHomeEvaluator:
    """
    Evaluator for Smart Home multi-task model.
    
    Same as CharadesAGEvaluator but with:
    - 42 smart home actions instead of 157 Charades actions
    - Label remapping from 219 -> 104 total classes
    """
    
    def __init__(self,
                 d_cfg,
                 data_root,
                 img_size=224,
                 len_clip=16,
                 sampling_rate=5,
                 batch_size=8,
                 transform=None,
                 collate_fn=None,
                 conf_thresh=0.01,
                 iou_thresh=0.5,
                 save_path='./evaluator/eval_results/',
                 smart_home_config=None):
        
        self.data_root = data_root
        self.img_size = img_size
        self.len_clip = len_clip
        self.sampling_rate = sampling_rate
        self.batch_size = batch_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.save_path = save_path
        self.collate_fn = collate_fn
        self.smart_home_config = smart_home_config
        
        # Create save directory
        os.makedirs(save_path, exist_ok=True)
        
        # Build test dataset using SmartHomeDataset
        self.testset = SmartHomeDataset(
            cfg=d_cfg,
            data_root=os.path.join(data_root, 'ActionGenome'),
            is_train=False,
            img_size=img_size,
            transform=transform,
            len_clip=len_clip,
            sampling_rate=sampling_rate
        )
        
        # Class info - 42 smart home actions
        self.num_objects = 36
        self.num_actions = smart_home_config['num_actions']  # 42
        self.num_relations = 26
        self.num_classes = 104  # 36 + 42 + 26
        
        # Action names from config
        self.action_names = smart_home_config['action_names']
        
        print(f"Smart Home Evaluator initialized:")
        print(f"  Test keyframes: {len(self.testset)}")
        print(f"  Objects: {self.num_objects}, Actions: {self.num_actions}, Relations: {self.num_relations}")
        print(f"  Total classes: {self.num_classes}")
    
    def _compute_iou(self, box1, box2):
        """Compute IoU between two boxes [x1, y1, x2, y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = box1_area + box2_area - inter_area
        
        if union_area <= 0:
            return 0.0
        return inter_area / union_area
    
    def _compute_ap(self, recalls, precisions):
        """Compute AP using 11-point interpolation (Pascal VOC style)."""
        ap = 0.0
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap = ap + p / 11.0
        return ap
    
    def _evaluate_class(self, all_gt, all_det, class_idx, task_type='action'):
        """Evaluate a single class using Pascal VOC protocol."""
        # Determine which slice of labels to use
        if task_type == 'object':
            label_start = 0
            label_end = self.num_objects
        elif task_type == 'action':
            label_start = self.num_objects
            label_end = self.num_objects + self.num_actions
        else:  # relation
            label_start = self.num_objects + self.num_actions
            label_end = self.num_classes
        
        local_idx = class_idx - label_start
        
        # Collect all detections for this class
        all_dets = []
        npos = 0
        gt_matched = defaultdict(lambda: defaultdict(bool))
        
        for frame_id in all_gt:
            for gt_idx, (gt_box, gt_labels) in enumerate(all_gt[frame_id]):
                if gt_labels[class_idx] > 0.5:
                    npos += 1
        
        if npos == 0:
            return 0.0, 0
        
        for frame_id in all_det:
            for det_box, det_score, det_labels in all_det[frame_id]:
                score = det_score * det_labels[class_idx]
                all_dets.append((frame_id, det_box, score))
        
        # Sort by score
        all_dets.sort(key=lambda x: -x[2])
        
        # Compute precision/recall
        tp = np.zeros(len(all_dets))
        fp = np.zeros(len(all_dets))
        
        for det_idx, (frame_id, det_box, score) in enumerate(all_dets):
            best_iou = 0.0
            best_gt_idx = -1
            
            if frame_id in all_gt:
                for gt_idx, (gt_box, gt_labels) in enumerate(all_gt[frame_id]):
                    if gt_labels[class_idx] <= 0.5:
                        continue
                    if gt_matched[frame_id][gt_idx]:
                        continue
                    
                    iou = self._compute_iou(det_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
            
            if best_iou >= self.iou_thresh and best_gt_idx >= 0:
                tp[det_idx] = 1
                gt_matched[frame_id][best_gt_idx] = True
            else:
                fp[det_idx] = 1
        
        # Compute cumulative values
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / npos
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        ap = self._compute_ap(recalls, precisions)
        return ap, npos

    @torch.no_grad()
    def evaluate_frame_map(self, model, epoch=1, max_samples=None):
        """Run evaluation on test set."""
        model.eval()
        device = next(model.parameters()).device
        
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=4
        )
        
        all_gt = defaultdict(list)
        all_det = defaultdict(list)
        
        print(f"\nEvaluating Smart Home model (Epoch {epoch})...")
        start_time = time.time()
        
        samples_processed = 0
        for batch_idx, (frame_ids, video_clips, targets) in enumerate(dataloader):
            video_clips = video_clips.to(device)
            
            # Forward pass
            outputs = model(video_clips)
            
            batch_size = video_clips.shape[0]
            
            for i in range(batch_size):
                frame_id = f"{frame_ids[i][0]}_{frame_ids[i][1]}"
                target = targets[i]
                
                # Ground truth
                if target['boxes'].shape[0] > 0:
                    gt_boxes = target['boxes'].numpy()
                    gt_labels = target['labels'].numpy()
                    
                    for j in range(gt_boxes.shape[0]):
                        all_gt[frame_id].append((gt_boxes[j], gt_labels[j]))
                
                # Predictions - concatenate all FPN levels
                pred_conf = torch.cat([o[i] for o in outputs['pred_conf']], dim=0)  # [N, 1]
                pred_obj = torch.cat([o[i] for o in outputs['pred_obj']], dim=0)    # [N, 36]
                pred_act = torch.cat([o[i] for o in outputs['pred_act']], dim=0)    # [N, 42]
                pred_rel = torch.cat([o[i] for o in outputs['pred_rel']], dim=0)    # [N, 26]
                pred_box = torch.cat([o[i] for o in outputs['pred_box']], dim=0)    # [N, 4]
                
                # Apply activations
                pred_conf = torch.sigmoid(pred_conf).squeeze(-1)
                pred_obj = torch.softmax(pred_obj, dim=-1)
                pred_act = torch.sigmoid(pred_act)
                pred_rel = torch.sigmoid(pred_rel)
                
                # Filter by confidence
                mask = pred_conf > self.conf_thresh
                if mask.sum() == 0:
                    continue
                
                pred_conf = pred_conf[mask].cpu().numpy()
                pred_obj = pred_obj[mask].cpu().numpy()
                pred_act = pred_act[mask].cpu().numpy()
                pred_rel = pred_rel[mask].cpu().numpy()
                pred_box = pred_box[mask].cpu().numpy()
                
                # Rescale boxes
                pred_box = rescale_bboxes(pred_box, self.img_size, self.img_size)
                
                # Combine predictions
                for j in range(pred_box.shape[0]):
                    combined_labels = np.concatenate([
                        pred_obj[j],  # 36
                        pred_act[j],  # 42
                        pred_rel[j]   # 26
                    ])  # 104 total
                    
                    all_det[frame_id].append((
                        pred_box[j],
                        pred_conf[j],
                        combined_labels
                    ))
            
            samples_processed += batch_size
            if max_samples and samples_processed >= max_samples:
                break
        
        elapsed = time.time() - start_time
        print(f"  Inference done: {samples_processed} samples in {elapsed:.1f}s")
        
        # Compute mAP for each task
        # Object mAP (indices 0:36)
        object_aps = []
        object_gts = []
        for cls_idx in range(self.num_objects):
            ap, ngt = self._evaluate_class(all_gt, all_det, cls_idx, 'object')
            object_aps.append(ap)
            object_gts.append(ngt)
        object_mAP = np.mean(object_aps)
        
        # Action mAP (indices 36:78)
        action_aps = []
        action_gts = []
        for cls_idx in range(self.num_objects, self.num_objects + self.num_actions):
            ap, ngt = self._evaluate_class(all_gt, all_det, cls_idx, 'action')
            action_aps.append(ap)
            action_gts.append(ngt)
        action_mAP = np.mean(action_aps)
        
        # Relation mAP (indices 78:104)
        relation_aps = []
        relation_gts = []
        for cls_idx in range(self.num_objects + self.num_actions, self.num_classes):
            ap, ngt = self._evaluate_class(all_gt, all_det, cls_idx, 'relation')
            relation_aps.append(ap)
            relation_gts.append(ngt)
        relation_mAP = np.mean(relation_aps)
        
        # Print results
        print(f"\n{'='*60}")
        print(f"Smart Home Evaluation Results (Epoch {epoch})")
        print(f"{'='*60}")
        print(f"  Object mAP:   {object_mAP*100:.2f}%")
        print(f"  Action mAP:   {action_mAP*100:.2f}% ({self.num_actions} classes)")
        print(f"  Relation mAP: {relation_mAP*100:.2f}%")
        print(f"{'='*60}")
        
        # Save detailed results
        results = {
            'object_mAP': object_mAP,
            'action_mAP': action_mAP,
            'relation_mAP': relation_mAP,
            'object_aps': object_aps,
            'action_aps': action_aps,
            'relation_aps': relation_aps,
            'object_gts': object_gts,
            'action_gts': action_gts,
            'relation_gts': relation_gts,
        }
        
        np.save(os.path.join(self.save_path, f'smart_home_eval_epoch_{epoch}.npy'), results)
        
        return results


if __name__ == "__main__":
    print("Smart Home Evaluator module loaded successfully")
