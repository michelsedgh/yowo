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
        
        # Also compute classification-only metrics (using matched boxes)
        self._compute_classification_metrics(all_gt, all_det, epoch)
        
        return results
    
    def _compute_classification_metrics(self, all_gt, all_det, epoch):
        """
        Compute classification accuracy metrics ONLY for detected boxes that matched GT.
        
        This gives a cleaner view of "how well is the model classifying?" 
        without penalizing for localization errors.
        
        Metrics computed:
        - Action recall @ IoU=0.5: Of all GT person actions, how many were correctly predicted?
        - Object accuracy @ IoU=0.5: Of matched boxes, what % had correct object class?
        - Action multi-label F1: F1 score for multi-label action prediction
        """
        print(f"\n{'='*60}")
        print(f"CLASSIFICATION METRICS (on IoU≥0.5 matched boxes)")
        print(f"{'='*60}")
        
        # Track metrics
        obj_correct = 0
        obj_total = 0
        
        act_tp = 0  # True positives for actions
        act_fp = 0  # False positives
        act_fn = 0  # False negatives
        
        rel_tp = 0
        rel_fp = 0
        rel_fn = 0
        
        person_action_correct = 0
        person_action_total = 0
        
        # For each frame, match detections to GT and compute classification metrics
        for frame_id, gt_items in all_gt.items():
            det_items = all_det.get(frame_id, [])
            
            # Track which GTs have been matched
            gt_matched = [False] * len(gt_items)
            
            for det_box, det_conf, det_labels in det_items:
                # Find best matching GT
                best_iou = 0.0
                best_gt_idx = -1
                
                for g_idx, (gt_box, gt_labels) in enumerate(gt_items):
                    if gt_matched[g_idx]:
                        continue
                    iou = self._compute_iou(det_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx
                
                if best_iou >= self.iou_thresh and best_gt_idx >= 0:
                    gt_matched[best_gt_idx] = True
                    gt_box, gt_labels = gt_items[best_gt_idx]
                    
                    # Object classification accuracy
                    pred_obj = np.argmax(det_labels[:self.num_objects])
                    gt_obj = np.argmax(gt_labels[:self.num_objects])
                    if pred_obj == gt_obj:
                        obj_correct += 1
                    obj_total += 1
                    
                    # Action metrics (multi-label) - only for person boxes
                    if gt_obj == 0:  # Person class
                        gt_acts = gt_labels[self.num_objects:self.num_objects+self.num_actions] > 0.5
                        pred_acts = det_labels[self.num_objects:self.num_objects+self.num_actions] > 0.3  # Lower threshold for predictions
                        
                        # Count TP, FP, FN
                        act_tp += np.sum(gt_acts & pred_acts)
                        act_fp += np.sum(~gt_acts & pred_acts)
                        act_fn += np.sum(gt_acts & ~pred_acts)
                        
                        # Also count if ANY action is correct
                        if np.any(gt_acts & pred_acts):
                            person_action_correct += 1
                        person_action_total += 1
                    
                    # Relation metrics (multi-label)
                    gt_rels = gt_labels[self.num_objects+self.num_actions:] > 0.5
                    pred_rels = det_labels[self.num_objects+self.num_actions:] > 0.3
                    
                    rel_tp += np.sum(gt_rels & pred_rels)
                    rel_fp += np.sum(~gt_rels & pred_rels)
                    rel_fn += np.sum(gt_rels & ~pred_rels)
        
        # Compute final metrics
        obj_accuracy = obj_correct / max(obj_total, 1) * 100
        
        act_precision = act_tp / max(act_tp + act_fp, 1)
        act_recall = act_tp / max(act_tp + act_fn, 1)
        act_f1 = 2 * act_precision * act_recall / max(act_precision + act_recall, 1e-8) * 100
        
        rel_precision = rel_tp / max(rel_tp + rel_fp, 1)
        rel_recall = rel_tp / max(rel_tp + rel_fn, 1)
        rel_f1 = 2 * rel_precision * rel_recall / max(rel_precision + rel_recall, 1e-8) * 100
        
        person_act_rate = person_action_correct / max(person_action_total, 1) * 100
        
        print(f"  📦 Object Accuracy (matched boxes): {obj_accuracy:.1f}% ({obj_correct}/{obj_total})")
        print(f"  🎬 Action F1 (person boxes only):   {act_f1:.1f}% (P={act_precision*100:.1f}%, R={act_recall*100:.1f}%)")
        print(f"  👤 Person boxes with ≥1 correct action: {person_act_rate:.1f}% ({person_action_correct}/{person_action_total})")
        print(f"  🔗 Relation F1 (all boxes):         {rel_f1:.1f}% (P={rel_precision*100:.1f}%, R={rel_recall*100:.1f}%)")
        print(f"{'='*60}")
        
        return {
            'obj_accuracy': obj_accuracy,
            'act_f1': act_f1,
            'act_precision': act_precision * 100,
            'act_recall': act_recall * 100,
            'rel_f1': rel_f1,
            'person_any_action_rate': person_act_rate
        }


if __name__ == "__main__":
    # Quick test of the evaluator structure
    print("Charades-AG Evaluator module loaded successfully")
