"""
Charades-AG Multi-Task Evaluator

This evaluator computes mAP for the three tasks in Charades-AG:
1. Object Detection mAP (36 classes) - using Pascal VOC protocol
2. Action Recognition mAP (157 classes) - frame-level, person-only
3. Relationship Detection mAP (26 classes) - frame-level

Following the same patterns as ava_evaluator.py and ucf_jhmdb_evaluator.py
"""

import os
import time
import numpy as np
import torch
from collections import defaultdict

from dataset.charades_ag import CharadesAGDataset
from dataset.transforms import BaseTransform
from utils.box_ops import rescale_bboxes


class CharadesAGEvaluator:
    """
    Evaluator for Charades-AG multi-task model.
    
    Computes:
    - Object mAP: Mean AP over 36 object classes (Pascal VOC style)
    - Action mAP: Mean AP over 157 Charades action classes
    - Relation mAP: Mean AP over 26 relationship classes
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
                 save_path='./evaluator/eval_results/'):
        
        self.data_root = data_root
        self.img_size = img_size
        self.len_clip = len_clip
        self.sampling_rate = sampling_rate
        self.batch_size = batch_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.save_path = save_path
        self.collate_fn = collate_fn
        
        # Create save directory
        os.makedirs(save_path, exist_ok=True)
        
        # Build test dataset
        self.testset = CharadesAGDataset(
            cfg=d_cfg,
            data_root=os.path.join(data_root, 'ActionGenome'),
            is_train=False,
            img_size=img_size,
            transform=transform,
            len_clip=len_clip,
            sampling_rate=sampling_rate
        )
        
        # Class info
        self.num_objects = self.testset.num_objects    # 36
        self.num_actions = self.testset.num_actions    # 157
        self.num_relations = self.testset.num_relations  # 26
        self.num_classes = self.testset.num_classes    # 219
        
        # Class names for reporting
        self.object_names = self.testset.ag_objects
        self.action_names = self.testset.charades_actions
        self.relation_names = self.testset.ag_relations
        
        print(f"Charades-AG Evaluator initialized:")
        print(f"  Test keyframes: {len(self.testset)}")
        print(f"  Objects: {self.num_objects}, Actions: {self.num_actions}, Relations: {self.num_relations}")
    
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
            ap += p / 11.0
        return ap
    
    def _compute_ap_all_points(self, recalls, precisions):
        """Compute AP using all-point interpolation (COCO style)."""
        # Add sentinel values
        mrec = np.concatenate(([0.], recalls, [1.]))
        mpre = np.concatenate(([0.], precisions, [0.]))
        
        # Make precision monotonically decreasing
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        
        # Find where recall changes
        i = np.where(mrec[1:] != mrec[:-1])[0]
        
        # Sum up rectangular areas
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap
    
    def _evaluate_class(self, all_gt, all_det, class_idx, task_type='object'):
        """
        Evaluate a single class using Pascal VOC protocol.
        
        Args:
            all_gt: dict {frame_id: [(box, label_vector), ...]}
            all_det: dict {frame_id: [(box, score, label_vector), ...]}
            class_idx: index of class to evaluate
            task_type: 'object', 'action', or 'relation'
            
        Returns:
            AP for this class, number of GT instances
        """
        # Collect all detections and GT for this class
        # For objects: use argmax of object scores
        # For actions/relations: use threshold on sigmoid scores (multi-label)
        
        gt_by_frame = defaultdict(list)
        det_list = []  # [(score, frame_id, box, is_tp)]
        
        n_gt = 0
        
        # Gather ground truths
        for frame_id, gt_items in all_gt.items():
            for box, labels in gt_items:
                if task_type == 'object':
                    # Object is exclusive - check if this is the class
                    gt_class = np.argmax(labels[:self.num_objects])
                    if gt_class == class_idx:
                        gt_by_frame[frame_id].append({
                            'box': box,
                            'used': False
                        })
                        n_gt += 1
                else:
                    # Actions/relations are multi-label
                    if task_type == 'action':
                        offset = self.num_objects
                        label_val = labels[offset + class_idx]
                    else:  # relation
                        offset = self.num_objects + self.num_actions
                        label_val = labels[offset + class_idx]
                    
                    if label_val > 0.5:  # GT threshold
                        gt_by_frame[frame_id].append({
                            'box': box,
                            'used': False
                        })
                        n_gt += 1
        
        if n_gt == 0:
            return 0.0, 0
        
        # Gather detections
        for frame_id, det_items in all_det.items():
            for box, conf, labels in det_items:
                if task_type == 'object':
                    # Use object probability for the class
                    score = labels[class_idx]
                else:
                    if task_type == 'action':
                        offset = self.num_objects
                        score = labels[offset + class_idx]
                    else:  # relation
                        offset = self.num_objects + self.num_actions
                        score = labels[offset + class_idx]
                
                det_list.append({
                    'score': score,
                    'frame_id': frame_id,
                    'box': box
                })
        
        # Sort by score descending
        det_list = sorted(det_list, key=lambda x: x['score'], reverse=True)
        
        # Match detections to GT
        tp = np.zeros(len(det_list))
        fp = np.zeros(len(det_list))
        
        for d_idx, det in enumerate(det_list):
            frame_id = det['frame_id']
            det_box = det['box']
            
            gts = gt_by_frame.get(frame_id, [])
            
            best_iou = 0.0
            best_gt_idx = -1
            
            for g_idx, gt in enumerate(gts):
                if gt['used']:
                    continue
                iou = self._compute_iou(det_box, gt['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx
            
            if best_iou >= self.iou_thresh:
                gts[best_gt_idx]['used'] = True
                tp[d_idx] = 1
            else:
                fp[d_idx] = 1
        
        # Compute precision/recall
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        
        recalls = cum_tp / n_gt
        precisions = cum_tp / (cum_tp + cum_fp)
        
        # Compute AP
        ap = self._compute_ap_all_points(recalls, precisions)
        
        return ap, n_gt
    
    def evaluate_frame_map(self, model, epoch=1, max_samples=None):
        """
        Run evaluation on test set.
        
        Args:
            model: trained YOWO multi-task model
            epoch: current epoch number (for logging)
            max_samples: if set, only evaluate this many samples (for quick testing)
            
        Returns:
            dict with object_mAP, action_mAP, relation_mAP
        """
        model.eval()
        
        # Create dataloader
        testloader = torch.utils.data.DataLoader(
            dataset=self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=2,
            drop_last=False,
            pin_memory=False  # Disabled to avoid OOM on constrained devices
        )
        
        epoch_size = len(testloader)
        if max_samples:
            epoch_size = min(epoch_size, max_samples // self.batch_size + 1)
        
        print(f"\n{'='*60}")
        print(f"Charades-AG Evaluation - Epoch {epoch}")
        print(f"{'='*60}")
        print(f"Evaluating {len(self.testset)} keyframes...")
        
        # Storage for GT and detections
        # Format: {frame_id: [(box, labels), ...]}
        all_gt = defaultdict(list)
        all_det = defaultdict(list)
        
        eval_start = time.time()
        
        # Run inference
        for iter_i, (batch_frame_id, batch_video_clip, batch_target) in enumerate(testloader):
            if max_samples and iter_i >= epoch_size:
                break
            
            batch_video_clip = batch_video_clip.to(model.device)
            
            with torch.no_grad():
                # Model returns list of detection arrays
                # Each detection: [x1, y1, x2, y2, conf, interact, obj_36, act_157, rel_26]
                batch_bboxes = model(batch_video_clip)
            
            # Process each sample in batch
            for bi in range(len(batch_bboxes)):
                frame_info = batch_frame_id[bi]  # [video_id, frame_idx]
                video_id = frame_info[0]
                frame_idx = frame_info[1]
                frame_id = f"{video_id}/{frame_idx:06d}"
                
                target = batch_target[bi]
                
                # Get original image size for rescaling
                orig_size = target['orig_size']  # [h, w]
                
                # Store ground truth
                gt_boxes = target['boxes'].numpy()  # [N, 4]
                gt_labels = target['labels'].numpy()  # [N, 219]
                
                for box, labels in zip(gt_boxes, gt_labels):
                    # Rescale GT boxes from normalized to original size
                    box_scaled = box.copy()
                    box_scaled[0] *= orig_size[1]  # x1 * w
                    box_scaled[1] *= orig_size[0]  # y1 * h
                    box_scaled[2] *= orig_size[1]  # x2 * w
                    box_scaled[3] *= orig_size[0]  # y2 * h
                    all_gt[frame_id].append((box_scaled, labels))
                
                # Store detections
                detections = batch_bboxes[bi]  # [M, 6+219]
                
                for det in detections:
                    box = det[:4]  # normalized [x1, y1, x2, y2]
                    conf = det[4]
                    # Labels start at index 5: [obj_36, act_157, rel_26] = 219 dims
                    labels = det[5:]  # Fixed: was det[6:], but labels start at 5
                    
                    if conf < self.conf_thresh:
                        continue
                    
                    # Rescale detection boxes to original size
                    box_scaled = np.zeros(4)
                    box_scaled[0] = box[0] * max(orig_size)
                    box_scaled[1] = box[1] * max(orig_size)
                    box_scaled[2] = box[2] * max(orig_size)
                    box_scaled[3] = box[3] * max(orig_size)
                    
                    all_det[frame_id].append((box_scaled, conf, labels))
            
            if iter_i % 50 == 0:
                print(f"  [{iter_i}/{epoch_size}] processed...")
        
        inference_time = time.time() - eval_start
        print(f"\nInference done in {inference_time:.1f}s")
        print(f"GT frames: {len(all_gt)}, Detection frames: {len(all_det)}")
        
        # Compute mAP for each task
        print(f"\nComputing mAP...")
        
        # 1. Object mAP
        object_aps = []
        object_gts = []
        for cls_idx in range(self.num_objects):
            ap, n_gt = self._evaluate_class(all_gt, all_det, cls_idx, 'object')
            object_aps.append(ap)
            object_gts.append(n_gt)
        
        # Filter classes with GT instances
        valid_obj_aps = [ap for ap, gt in zip(object_aps, object_gts) if gt > 0]
        object_mAP = np.mean(valid_obj_aps) if valid_obj_aps else 0.0
        
        # 2. Action mAP
        action_aps = []
        action_gts = []
        for cls_idx in range(self.num_actions):
            ap, n_gt = self._evaluate_class(all_gt, all_det, cls_idx, 'action')
            action_aps.append(ap)
            action_gts.append(n_gt)
        
        valid_act_aps = [ap for ap, gt in zip(action_aps, action_gts) if gt > 0]
        action_mAP = np.mean(valid_act_aps) if valid_act_aps else 0.0
        
        # 3. Relation mAP
        relation_aps = []
        relation_gts = []
        for cls_idx in range(self.num_relations):
            ap, n_gt = self._evaluate_class(all_gt, all_det, cls_idx, 'relation')
            relation_aps.append(ap)
            relation_gts.append(n_gt)
        
        valid_rel_aps = [ap for ap, gt in zip(relation_aps, relation_gts) if gt > 0]
        relation_mAP = np.mean(valid_rel_aps) if valid_rel_aps else 0.0
        
        # Print results
        total_time = time.time() - eval_start
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS - Epoch {epoch}")
        print(f"{'='*60}")
        print(f"Object mAP @ IoU=0.5:   {object_mAP*100:.2f}% ({len(valid_obj_aps)}/{self.num_objects} classes)")
        print(f"Action mAP @ IoU=0.5:   {action_mAP*100:.2f}% ({len(valid_act_aps)}/{self.num_actions} classes)")
        print(f"Relation mAP @ IoU=0.5: {relation_mAP*100:.2f}% ({len(valid_rel_aps)}/{self.num_relations} classes)")
        print(f"{'='*60}")
        print(f"Total eval time: {total_time:.1f}s")
        
        # Per-class breakdown (top/bottom performers)
        if len(valid_obj_aps) > 0:
            obj_sorted = sorted([(ap, name) for ap, name, gt in zip(object_aps, self.object_names, object_gts) if gt > 0], reverse=True)
            print(f"\nTop 5 Objects: {[(f'{name}:{ap*100:.1f}%') for ap, name in obj_sorted[:5]]}")
            print(f"Bottom 5 Objects: {[(f'{name}:{ap*100:.1f}%') for ap, name in obj_sorted[-5:]]}")
        
        if len(valid_rel_aps) > 0:
            rel_sorted = sorted([(ap, name) for ap, name, gt in zip(relation_aps, self.relation_names, relation_gts) if gt > 0], reverse=True)
            print(f"\nTop 5 Relations: {[(f'{name}:{ap*100:.1f}%') for ap, name in rel_sorted[:5]]}")
        
        # Save results
        results = {
            'epoch': epoch,
            'object_mAP': object_mAP,
            'action_mAP': action_mAP,
            'relation_mAP': relation_mAP,
            'object_aps': object_aps,
            'action_aps': action_aps,
            'relation_aps': relation_aps,
            'object_gts': object_gts,
            'action_gts': action_gts,
            'relation_gts': relation_gts,
            'eval_time': total_time
        }
        
        save_file = os.path.join(self.save_path, f'charades_ag_eval_epoch_{epoch}.npy')
        np.save(save_file, results)
        print(f"\nResults saved to: {save_file}")
        
        return results


if __name__ == "__main__":
    # Quick test of the evaluator structure
    print("Charades-AG Evaluator module loaded successfully")
