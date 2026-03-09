"""
Readable Evaluator for Charades-AG Multi-Task Model

This evaluator provides HUMAN-READABLE metrics instead of mAP:
- Per-class accuracy/recall/precision percentages
- Confusion matrix summaries
- Top predictions analysis
- Easy-to-understand performance breakdown

Author: Created for smart home action detection
"""

import os
import time
import numpy as np
import torch
from collections import defaultdict, Counter

from dataset.charades_ag import CharadesAGDataset
from dataset.transforms import BaseTransform


class ReadableEvaluator:
    """
    Human-readable evaluator that shows:
    - What % of each class is detected correctly
    - Which classes are confused with which
    - Clear pass/fail metrics per class
    """
    
    def __init__(self,
                 d_cfg,
                 data_root,
                 img_size=224,
                 len_clip=16,
                 sampling_rate=1,
                 batch_size=8,
                 transform=None,
                 collate_fn=None,
                 conf_thresh=0.3,
                 iou_thresh=0.5):
        
        self.img_size = img_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        
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
        self.num_objects = self.testset.num_objects
        self.num_actions = self.testset.num_actions
        self.num_relations = self.testset.num_relations
        
        self.object_names = self.testset.ag_objects
        self.action_names = self.testset.charades_actions
        self.relation_names = self.testset.ag_relations
        
        print(f"✅ Readable Evaluator initialized")
        print(f"   Test samples: {len(self.testset)}")
    
    def _compute_iou(self, box1, box2):
        """Compute IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / max(union, 1e-8)
    
    def evaluate(self, model, epoch=1, max_samples=None, verbose=True):
        """
        Run evaluation and produce READABLE report.
        """
        model.eval()
        device = next(model.parameters()).device
        
        num_workers = 8
        testloader = torch.utils.data.DataLoader(
            dataset=self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None
        )
        
        # Metrics storage
        # Per-class: TP, FP, FN, Total GT
        obj_stats = {i: {'tp': 0, 'fp': 0, 'fn': 0, 'gt': 0} for i in range(self.num_objects)}
        act_stats = {i: {'tp': 0, 'fp': 0, 'fn': 0, 'gt': 0} for i in range(self.num_actions)}
        rel_stats = {i: {'tp': 0, 'fp': 0, 'fn': 0, 'gt': 0} for i in range(self.num_relations)}
        
        # Confusion tracking
        obj_confusion = defaultdict(Counter)  # obj_confusion[gt_class][pred_class] = count
        
        # Overall
        total_boxes_gt = 0
        total_boxes_detected = 0
        total_boxes_matched = 0
        
        print(f"\n{'='*70}")
        print(f"📊 READABLE EVALUATION - Epoch {epoch}")
        print(f"{'='*70}")
        
        if max_samples:
            print(f"⚠️ Quick mode: evaluating {max_samples} samples only")
        
        eval_size = len(testloader)
        if max_samples:
            eval_size = min(eval_size, max_samples // self.batch_size + 1)
        
        start_time = time.time()
        
        for iter_i, (batch_frame_id, batch_video_clip, batch_target) in enumerate(testloader):
            if max_samples and iter_i >= eval_size:
                break
            
            batch_video_clip = batch_video_clip.to(device, non_blocking=True)
            
            with torch.no_grad():
                batch_outputs = model(batch_video_clip)
            
            # Process each sample
            for bi in range(len(batch_outputs)):
                target = batch_target[bi]
                detections = batch_outputs[bi]
                
                gt_boxes = target['boxes'].numpy()
                gt_labels = target['labels'].numpy()
                orig_size = target['orig_size']
                
                total_boxes_gt += len(gt_boxes)
                
                # Match detections to GT
                gt_matched = [False] * len(gt_boxes)
                
                for det in detections:
                    if det[4] < self.conf_thresh:
                        continue
                    
                    total_boxes_detected += 1
                    
                    det_box = det[:4].copy()
                    det_box[0] *= orig_size[1]
                    det_box[1] *= orig_size[0]
                    det_box[2] *= orig_size[1]
                    det_box[3] *= orig_size[0]
                    
                    det_labels = det[5:]
                    
                    # Find best matching GT
                    best_iou = 0
                    best_gt_idx = -1
                    
                    for g_idx, gt_box in enumerate(gt_boxes):
                        if gt_matched[g_idx]:
                            continue
                        gt_box_scaled = gt_box.copy()
                        gt_box_scaled[0] *= orig_size[1]
                        gt_box_scaled[1] *= orig_size[0]
                        gt_box_scaled[2] *= orig_size[1]
                        gt_box_scaled[3] *= orig_size[0]
                        
                        iou = self._compute_iou(det_box, gt_box_scaled)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = g_idx
                    
                    if best_iou >= self.iou_thresh and best_gt_idx >= 0:
                        gt_matched[best_gt_idx] = True
                        total_boxes_matched += 1
                        
                        gt_label = gt_labels[best_gt_idx]
                        
                        # === OBJECT EVALUATION ===
                        gt_obj = np.argmax(gt_label[:self.num_objects])
                        pred_obj = np.argmax(det_labels[:self.num_objects])
                        
                        obj_stats[gt_obj]['gt'] += 1
                        if pred_obj == gt_obj:
                            obj_stats[gt_obj]['tp'] += 1
                        else:
                            obj_stats[gt_obj]['fn'] += 1
                            obj_stats[pred_obj]['fp'] += 1
                            obj_confusion[self.object_names[gt_obj]][self.object_names[pred_obj]] += 1
                        
                        # === ACTION EVALUATION (multi-label) ===
                        gt_acts = gt_label[self.num_objects:self.num_objects+self.num_actions] > 0.5
                        pred_acts = det_labels[self.num_objects:self.num_objects+self.num_actions] > 0.3
                        
                        for a_idx in range(self.num_actions):
                            if gt_acts[a_idx]:
                                act_stats[a_idx]['gt'] += 1
                                if pred_acts[a_idx]:
                                    act_stats[a_idx]['tp'] += 1
                                else:
                                    act_stats[a_idx]['fn'] += 1
                            else:
                                if pred_acts[a_idx]:
                                    act_stats[a_idx]['fp'] += 1
                        
                        # === RELATION EVALUATION (multi-label) ===
                        gt_rels = gt_label[self.num_objects+self.num_actions:] > 0.5
                        pred_rels = det_labels[self.num_objects+self.num_actions:] > 0.3
                        
                        for r_idx in range(self.num_relations):
                            if gt_rels[r_idx]:
                                rel_stats[r_idx]['gt'] += 1
                                if pred_rels[r_idx]:
                                    rel_stats[r_idx]['tp'] += 1
                                else:
                                    rel_stats[r_idx]['fn'] += 1
                            else:
                                if pred_rels[r_idx]:
                                    rel_stats[r_idx]['fp'] += 1
            
            if iter_i % 20 == 0:
                print(f"  [{iter_i+1}/{eval_size}] processed...")
        
        eval_time = time.time() - start_time
        
        # === PRINT READABLE REPORT ===
        self._print_report(obj_stats, act_stats, rel_stats, obj_confusion,
                          total_boxes_gt, total_boxes_detected, total_boxes_matched,
                          eval_time, epoch, verbose)
        
        return {
            'obj_stats': obj_stats,
            'act_stats': act_stats,
            'rel_stats': rel_stats,
            'detection_rate': total_boxes_matched / max(total_boxes_gt, 1)
        }
    
    def _print_report(self, obj_stats, act_stats, rel_stats, obj_confusion,
                     total_gt, total_det, total_matched, eval_time, epoch, verbose):
        """Print human-readable evaluation report."""
        
        print(f"\n{'='*70}")
        print(f"📊 EVALUATION REPORT - Epoch {epoch}")
        print(f"{'='*70}")
        
        # Detection overview
        det_rate = total_matched / max(total_gt, 1) * 100
        print(f"\n🎯 DETECTION OVERVIEW:")
        print(f"   Ground Truth boxes: {total_gt}")
        print(f"   Detected boxes:     {total_det}")
        print(f"   Matched (IoU≥0.5):  {total_matched}")
        print(f"   Detection Rate:     {det_rate:.1f}%")
        
        # === OBJECT CLASSIFICATION ===
        print(f"\n{'='*70}")
        print(f"📦 OBJECT CLASSIFICATION ({self.num_objects} classes)")
        print(f"{'='*70}")
        
        obj_results = []
        for i in range(self.num_objects):
            s = obj_stats[i]
            if s['gt'] > 0:
                recall = s['tp'] / s['gt'] * 100
                precision = s['tp'] / max(s['tp'] + s['fp'], 1) * 100
                obj_results.append((self.object_names[i], s['gt'], s['tp'], recall, precision))
        
        obj_results.sort(key=lambda x: x[3], reverse=True)  # Sort by recall
        
        print(f"\n{'Class':<25} {'GT':>6} {'Correct':>8} {'Recall':>8} {'Precision':>10}")
        print("-" * 60)
        
        for name, gt, tp, recall, precision in obj_results[:15]:
            bar = "█" * int(recall / 10) + "░" * (10 - int(recall / 10))
            print(f"{name:<25} {gt:>6} {tp:>8} {recall:>7.1f}% {precision:>9.1f}%  {bar}")
        
        if len(obj_results) > 15:
            print(f"... and {len(obj_results) - 15} more classes")
        
        # Overall object accuracy
        total_obj_tp = sum(s['tp'] for s in obj_stats.values())
        total_obj_gt = sum(s['gt'] for s in obj_stats.values())
        obj_accuracy = total_obj_tp / max(total_obj_gt, 1) * 100
        print(f"\n✅ Overall Object Accuracy: {obj_accuracy:.1f}% ({total_obj_tp}/{total_obj_gt})")
        
        # Object confusion
        if verbose and obj_confusion:
            print(f"\n⚠️ COMMON OBJECT CONFUSIONS:")
            confusions = []
            for gt_cls, pred_counts in obj_confusion.items():
                for pred_cls, count in pred_counts.items():
                    confusions.append((gt_cls, pred_cls, count))
            confusions.sort(key=lambda x: x[2], reverse=True)
            for gt_cls, pred_cls, count in confusions[:5]:
                print(f"   {gt_cls} → {pred_cls}: {count} times")
        
        # === ACTION CLASSIFICATION ===
        print(f"\n{'='*70}")
        print(f"🎬 ACTION CLASSIFICATION ({self.num_actions} classes)")
        print(f"{'='*70}")
        
        act_results = []
        for i in range(self.num_actions):
            s = act_stats[i]
            if s['gt'] > 0:
                recall = s['tp'] / s['gt'] * 100
                precision = s['tp'] / max(s['tp'] + s['fp'], 1) * 100
                f1 = 2 * recall * precision / max(recall + precision, 1)
                act_results.append((self.action_names[i], s['gt'], s['tp'], recall, precision, f1))
        
        act_results.sort(key=lambda x: x[3], reverse=True)
        
        print(f"\n{'Class':<50} {'GT':>6} {'TP':>5} {'Recall':>7} {'Prec':>6}")
        print("-" * 80)
        
        for name, gt, tp, recall, precision, f1 in act_results[:20]:
            name_short = name[:48] + ".." if len(name) > 50 else name
            bar = "█" * int(recall / 10) + "░" * (10 - int(recall / 10))
            print(f"{name_short:<50} {gt:>6} {tp:>5} {recall:>6.1f}% {precision:>5.1f}%  {bar}")
        
        if len(act_results) > 20:
            print(f"... and {len(act_results) - 20} more classes")
        
        # Worst actions
        print(f"\n⚠️ HARDEST ACTIONS (lowest recall with GT > 10):")
        hard_acts = [r for r in act_results if r[1] > 10]  # GT > 10
        hard_acts.sort(key=lambda x: x[3])  # Sort by recall ascending
        for name, gt, tp, recall, precision, f1 in hard_acts[:5]:
            print(f"   {name[:40]}: {recall:.1f}% recall ({tp}/{gt})")
        
        # Overall action metrics
        total_act_tp = sum(s['tp'] for s in act_stats.values())
        total_act_gt = sum(s['gt'] for s in act_stats.values())
        total_act_fp = sum(s['fp'] for s in act_stats.values())
        act_recall = total_act_tp / max(total_act_gt, 1) * 100
        act_precision = total_act_tp / max(total_act_tp + total_act_fp, 1) * 100
        print(f"\n✅ Overall Action Recall: {act_recall:.1f}%")
        print(f"✅ Overall Action Precision: {act_precision:.1f}%")
        
        # === RELATION CLASSIFICATION ===
        print(f"\n{'='*70}")
        print(f"🔗 RELATION CLASSIFICATION ({self.num_relations} classes)")
        print(f"{'='*70}")
        
        print(f"\n{'Class':<30} {'GT':>6} {'TP':>5} {'Recall':>7} {'Prec':>6}")
        print("-" * 60)
        
        for i in range(self.num_relations):
            s = rel_stats[i]
            if s['gt'] > 0:
                recall = s['tp'] / s['gt'] * 100
                precision = s['tp'] / max(s['tp'] + s['fp'], 1) * 100
                bar = "█" * int(recall / 10) + "░" * (10 - int(recall / 10))
                print(f"{self.relation_names[i]:<30} {s['gt']:>6} {s['tp']:>5} {recall:>6.1f}% {precision:>5.1f}%  {bar}")
        
        # Summary
        total_rel_tp = sum(s['tp'] for s in rel_stats.values())
        total_rel_gt = sum(s['gt'] for s in rel_stats.values())
        rel_recall = total_rel_tp / max(total_rel_gt, 1) * 100
        
        print(f"\n✅ Overall Relation Recall: {rel_recall:.1f}%")
        
        # === FINAL SUMMARY ===
        print(f"\n{'='*70}")
        print(f"📋 FINAL SUMMARY - Epoch {epoch}")
        print(f"{'='*70}")
        print(f"   🎯 Detection Rate:    {det_rate:.1f}%")
        print(f"   📦 Object Accuracy:   {obj_accuracy:.1f}%")
        print(f"   🎬 Action Recall:     {act_recall:.1f}%")
        print(f"   🔗 Relation Recall:   {rel_recall:.1f}%")
        print(f"   ⏱️ Eval Time:         {eval_time:.1f}s")
        print(f"{'='*70}")


def build_readable_evaluator(args, d_cfg, img_size, sampling_rate, collate_fn=None):
    """Build the readable evaluator."""
    transform = BaseTransform(img_size=img_size)
    
    return ReadableEvaluator(
        d_cfg=d_cfg,
        data_root=args.root,
        img_size=img_size,
        len_clip=args.len_clip,
        sampling_rate=sampling_rate,
        batch_size=args.test_batch_size if hasattr(args, 'test_batch_size') else 8,
        transform=transform,
        collate_fn=collate_fn,
        conf_thresh=args.conf_thresh if hasattr(args, 'conf_thresh') else 0.3,
        iou_thresh=0.5
    )
