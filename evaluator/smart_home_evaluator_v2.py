"""
Smart Home Evaluator V2 - Real Metrics Edition

Shows ACTUAL numbers instead of thresholds:
- Raw probability distributions
- Automatic optimal threshold finding
- Object/Relation/Action metrics
- IoU distribution analysis
"""

import os
import gc
import time
import numpy as np
import torch
from collections import defaultdict
import json

from dataset.smart_home import SmartHomeDataset
from dataset.transforms import BaseTransform


class SmartHomeEvaluatorV2:
    """
    Smart Home evaluator with real, interpretable metrics.
    No hardcoded thresholds - shows actual probability values.
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
                 conf_thresh=0.1,
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
        
        # Load smart home config
        if smart_home_config is None:
            config_path = os.path.join(os.path.dirname(__file__), 
                                       '../config/smart_home_final.json')
            with open(config_path) as f:
                smart_home_config = json.load(f)
        self.smart_home_config = smart_home_config
        
        os.makedirs(save_path, exist_ok=True)
        
        # Build test dataset
        self.testset = SmartHomeDataset(
            cfg=d_cfg,
            data_root=os.path.join(data_root, 'ActionGenome'),
            is_train=False,
            img_size=img_size,
            transform=transform,
            len_clip=len_clip,
            sampling_rate=sampling_rate
        )
        
        # Class info - use config-driven counts
        self.num_objects = 36
        self.num_actions = smart_home_config['num_actions']
        self.num_relations = 26
        self.num_classes = self.num_objects + self.num_actions + self.num_relations
        
        self.action_names = smart_home_config['action_names']
        self.object_names = self.testset.base_dataset.ag_objects
        self.relation_names = self.testset.base_dataset.ag_relations
        
        print(f"Smart Home Evaluator V2 initialized:")
        print(f"  Test keyframes: {len(self.testset)}")
        print(f"  Objects: {self.num_objects}, Actions: {self.num_actions}, Relations: {self.num_relations}")
    
    def _compute_iou_matrix(self, boxes1, boxes2):
        """Vectorized IoU: [N,4] x [M,4] -> [N,M] matrix."""
        x1 = np.maximum(boxes1[:, 0:1], boxes2[:, 0:1].T)
        y1 = np.maximum(boxes1[:, 1:2], boxes2[:, 1:2].T)
        x2 = np.minimum(boxes1[:, 2:3], boxes2[:, 2:3].T)
        y2 = np.minimum(boxes1[:, 3:4], boxes2[:, 3:4].T)
        
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1[:, None] + area2[None, :] - inter
        
        return inter / np.maximum(union, 1e-6)

    @torch.no_grad()
    def evaluate_frame_map(self, model, epoch=1, max_samples=None):
        """Run evaluation with real, interpretable metrics."""
        model.eval()
        device = model.device
        
        num_workers = 8
        def _no_gc(wid):
            gc.disable()
        dataloader = torch.utils.data.DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=4 if num_workers > 0 else None,
            worker_init_fn=_no_gc if num_workers > 0 else None
        )
        
        print(f"\n{'='*70}")
        print(f"🏠 SMART HOME EVALUATION - Epoch {epoch}")
        print(f"{'='*70}")
        print(f"Evaluating {len(self.testset)} keyframes...")
        
        # Storage for RAW DATA
        all_action_preds = []
        all_action_gts = []
        all_object_preds = []
        all_object_gts = []
        all_relation_preds = []
        all_relation_gts = []
        all_ious = []
        
        # Detection stats
        total_gt_boxes = 0
        total_person_gt = 0
        total_det_boxes = 0
        total_person_det = 0
        matched_boxes = 0
        matched_person = 0
        
        start_time = time.time()
        
        for batch_idx, (frame_ids, video_clips, targets) in enumerate(dataloader):
            video_clips = video_clips.to(device, non_blocking=True)
            batch_bboxes = model(video_clips)
            
            for i in range(len(batch_bboxes)):
                target = targets[i]
                detections = batch_bboxes[i]
                
                gt_boxes = target['boxes'].numpy()
                gt_labels = target['labels'].numpy()
                
                if gt_boxes.shape[0] == 0:
                    continue
                
                total_gt_boxes += len(gt_boxes)
                
                # Scale GT boxes
                orig_size = target['orig_size']
                gt_boxes_scaled = gt_boxes.copy()
                gt_boxes_scaled[:, [0, 2]] *= orig_size[1]
                gt_boxes_scaled[:, [1, 3]] *= orig_size[0]
                
                person_mask = gt_labels[:, 0] > 0.5
                total_person_gt += person_mask.sum()
                
                if len(detections) == 0:
                    continue
                
                total_det_boxes += len(detections)
                
                det_boxes = detections[:, :4]
                det_confs = detections[:, 4]
                det_labels = detections[:, 5:]
                
                # FIX: Scale det boxes the SAME way as GT boxes
                # orig_size = [height, width]
                det_boxes_scaled = det_boxes.copy()
                det_boxes_scaled[:, [0, 2]] *= orig_size[1]  # x uses width
                det_boxes_scaled[:, [1, 3]] *= orig_size[0]  # y uses height
                
                det_obj_probs = det_labels[:, :self.num_objects]
                person_det_mask = det_obj_probs.argmax(axis=1) == 0
                total_person_det += person_det_mask.sum()
                
                conf_order = np.argsort(-det_confs)
                
                # Class-aware matching: person dets -> person GTs, object dets -> object GTs
                det_is_person = det_obj_probs.argmax(axis=1) == 0
                gt_is_person = gt_labels[:, 0] > 0.5
                
                # Vectorized IoU matrix [num_det, num_gt] + class mask
                iou_matrix = self._compute_iou_matrix(det_boxes_scaled, gt_boxes_scaled)
                class_mask = det_is_person[:, None] == gt_is_person[None, :]
                iou_matrix_masked = iou_matrix * class_mask
                
                gt_matched = np.zeros(len(gt_boxes), dtype=bool)
                
                for det_idx in conf_order:
                    ious = iou_matrix_masked[det_idx].copy()
                    ious[gt_matched] = 0.0
                    
                    best_gt_idx = ious.argmax()
                    best_iou = ious[best_gt_idx]
                    all_ious.append(best_iou)
                    
                    if best_iou >= self.iou_thresh:
                        gt_matched[best_gt_idx] = True
                        matched_boxes += 1
                        
                        gt_label = gt_labels[best_gt_idx]
                        det_label = det_labels[det_idx]
                        is_person = gt_label[0] > 0.5
                        
                        if is_person:
                            matched_person += 1
                        
                        all_object_preds.append(det_label[:self.num_objects])
                        all_object_gts.append(gt_label[:self.num_objects].argmax())
                        
                        all_relation_preds.append(det_label[self.num_objects + self.num_actions:])
                        all_relation_gts.append(gt_label[self.num_objects + self.num_actions:] > 0.5)
                        
                        if is_person:
                            all_action_preds.append(det_label[self.num_objects:self.num_objects + self.num_actions])
                            all_action_gts.append(gt_label[self.num_objects:self.num_objects + self.num_actions] > 0.5)
            
            if batch_idx % 50 == 0:
                print(f"  [{batch_idx}/{len(dataloader)}] processed...")
            
            if max_samples and (batch_idx + 1) * self.batch_size >= max_samples:
                break
        
        elapsed = time.time() - start_time
        
        # Convert to arrays
        all_action_preds = np.array(all_action_preds) if all_action_preds else np.zeros((0, self.num_actions))
        all_action_gts = np.array(all_action_gts) if all_action_gts else np.zeros((0, self.num_actions), dtype=bool)
        all_object_preds = np.array(all_object_preds) if all_object_preds else np.zeros((0, self.num_objects))
        all_object_gts = np.array(all_object_gts) if all_object_gts else np.array([])
        all_relation_preds = np.array(all_relation_preds) if all_relation_preds else np.zeros((0, self.num_relations))
        all_relation_gts = np.array(all_relation_gts) if all_relation_gts else np.zeros((0, self.num_relations), dtype=bool)
        all_ious = np.array(all_ious)
        
        # =====================================================================
        # PRINT RESULTS
        # =====================================================================
        
        print(f"\n{'='*70}")
        print(f"📊 RESULTS - Epoch {epoch}")
        print(f"{'='*70}")
        
        # 1. DETECTION
        print(f"\n📦 BOX DETECTION:")
        print(f"   GT boxes:       {total_gt_boxes}")
        print(f"   Detected:       {total_det_boxes}")
        print(f"   Matched (IoU≥{self.iou_thresh}): {matched_boxes} ({100*matched_boxes/max(total_gt_boxes,1):.1f}%)")
        print(f"   Person GT:      {total_person_gt}")
        print(f"   Person Matched: {matched_person} ({100*matched_person/max(total_person_gt,1):.1f}%)")
        
        if len(all_ious) > 0:
            print(f"\n   IoU Stats: mean={all_ious.mean():.3f}, median={np.median(all_ious):.3f}")
            print(f"\n   📊 MATCH RATE AT DIFFERENT IoU THRESHOLDS:")
            print(f"   (Use lower threshold for smart home - you just need rough overlap)")
            for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]:
                match_rate = 100 * (all_ious >= thresh).mean()
                marker = " ← current" if abs(thresh - self.iou_thresh) < 0.01 else ""
                print(f"      IoU≥{thresh}: {match_rate:5.1f}% boxes matched{marker}")
        
        # 2. OBJECTS - Full Per-Class Breakdown
        obj_acc = 0
        object_results = []
        print(f"\n🎯 OBJECT CLASSIFICATION ({len(all_object_gts)} matched boxes):")
        if len(all_object_gts) > 0:
            obj_preds_class = all_object_preds.argmax(axis=1)
            obj_acc = (obj_preds_class == all_object_gts).mean()
            obj_max_probs = all_object_preds.max(axis=1)
            print(f"   Overall Accuracy: {100*obj_acc:.1f}%")
            print(f"   Confidence: mean={obj_max_probs.mean():.3f}, std={obj_max_probs.std():.3f}")
            
            # Per-class object accuracy
            print(f"\n   {'='*65}")
            print(f"   📦 PER-OBJECT BREAKDOWN (all {self.num_objects} classes)")
            print(f"   {'='*65}")
            print(f"   {'Object':<20} {'GT':>5} {'Correct':>8} {'Acc':>6} {'AvgConf':>8}")
            print(f"   {'-'*65}")
            
            for cls_idx in range(self.num_objects):
                mask = all_object_gts == cls_idx
                gt_count = mask.sum()
                if gt_count > 0:
                    correct = (obj_preds_class[mask] == cls_idx).sum()
                    acc = correct / gt_count
                    avg_conf = all_object_preds[mask, cls_idx].mean()
                else:
                    correct = 0
                    acc = 0
                    avg_conf = 0
                
                obj_name = self.object_names[cls_idx] if cls_idx < len(self.object_names) else f"class_{cls_idx}"
                object_results.append({
                    'name': obj_name,
                    'gt': int(gt_count),
                    'correct': int(correct),
                    'accuracy': float(acc),
                    'avg_conf': float(avg_conf)
                })
            
            # Sort by GT count (most common first)
            object_results.sort(key=lambda x: -x['gt'])
            
            for r in object_results:
                if r['gt'] > 0:  # Only show classes with GT samples
                    name = r['name'][:19] if len(r['name']) > 19 else r['name']
                    print(f"   {name:<20} {r['gt']:>5} {r['correct']:>8} {100*r['accuracy']:>5.1f}% {r['avg_conf']:>7.3f}")
            
            print(f"   {'-'*65}")
        
        # 3. ACTIONS
        best_f1 = 0
        best_thresh = 0.3
        top1 = top5 = n_samples = 0
        action_results = []
        
        print(f"\n🎬 ACTION RECOGNITION ({len(all_action_preds)} person boxes):")
        if len(all_action_preds) > 0:
            # Raw probabilities
            print(f"\n   Raw Probabilities:")
            print(f"      Overall mean: {all_action_preds.mean():.4f}")
            
            gt_pos = all_action_gts == True
            gt_neg = all_action_gts == False
            if gt_pos.any():
                pos_preds = all_action_preds[gt_pos]
                print(f"      When GT=True:  mean={pos_preds.mean():.4f}, median={np.median(pos_preds):.4f}")
                print(f"         >0.5: {100*(pos_preds > 0.5).mean():.1f}%, >0.3: {100*(pos_preds > 0.3).mean():.1f}%, >0.1: {100*(pos_preds > 0.1).mean():.1f}%")
            if gt_neg.any():
                neg_preds = all_action_preds[gt_neg]
                print(f"      When GT=False: mean={neg_preds.mean():.4f}")
            
            # Find optimal threshold
            print(f"\n   Threshold Analysis:")
            for thresh in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
                pred_bin = all_action_preds > thresh
                tp = (pred_bin & all_action_gts).sum()
                fp = (pred_bin & ~all_action_gts).sum()
                fn = (~pred_bin & all_action_gts).sum()
                prec = tp / max(tp + fp, 1)
                rec = tp / max(tp + fn, 1)
                f1 = 2 * prec * rec / max(prec + rec, 0.001)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh
                print(f"      @{thresh:.2f}: P={100*prec:.1f}% R={100*rec:.1f}% F1={100*f1:.1f}%")
            
            print(f"\n   ✅ Best: thresh={best_thresh}, F1={100*best_f1:.1f}%")
            
            # Top-K
            for pred, gt in zip(all_action_preds, all_action_gts):
                if gt.any():
                    n_samples += 1
                    top_k = np.argsort(-pred)
                    gt_idx = np.where(gt)[0]
                    if any(i in top_k[:1] for i in gt_idx): top1 += 1
                    if any(i in top_k[:5] for i in gt_idx): top5 += 1
            
            if n_samples > 0:
                print(f"\n   Top-K Accuracy:")
                print(f"      Top-1: {100*top1/n_samples:.1f}%")
                print(f"      Top-5: {100*top5/n_samples:.1f}%")
            
            # Per-action with RAW PROBABILITIES
            print(f"\n   {'='*75}")
            print(f"   📋 PER-ACTION BREAKDOWN (all {self.num_actions} actions)")
            print(f"   {'='*75}")
            print(f"   {'Action':<30} {'GT':>4} {'PredT':>6} {'PredF':>6} {'P':>5} {'R':>5} {'F1':>5}")
            print(f"   {'-'*75}")
            
            for act_idx in range(self.num_actions):
                gt_count = all_action_gts[:, act_idx].sum()
                preds_this_action = all_action_preds[:, act_idx]
                gt_this_action = all_action_gts[:, act_idx]
                
                # Raw probability stats
                mean_when_true = preds_this_action[gt_this_action].mean() if gt_this_action.any() else 0
                mean_when_false = preds_this_action[~gt_this_action].mean() if (~gt_this_action).any() else 0
                
                # P/R/F1
                pred_bin = preds_this_action > best_thresh
                tp = (pred_bin & gt_this_action).sum()
                fp = (pred_bin & ~gt_this_action).sum()
                fn = (~pred_bin & gt_this_action).sum()
                prec = tp / max(tp + fp, 1)
                rec = tp / max(tp + fn, 1)
                f1 = 2 * prec * rec / max(prec + rec, 0.001)
                
                action_results.append({
                    'name': self.action_names[act_idx], 
                    'gt': int(gt_count),
                    'pred_when_true': float(mean_when_true),
                    'pred_when_false': float(mean_when_false),
                    'prec': prec, 'rec': rec, 'f1': f1
                })
            
            # Sort by F1 for display
            action_results.sort(key=lambda x: -x['f1'])
            
            # Print ALL actions (readable format)
            for r in action_results:
                name = r['name'][:29] if len(r['name']) > 29 else r['name']
                # Color coding: PredT should be HIGH (good if >0.3), PredF should be LOW (good if <0.1)
                pred_t = r['pred_when_true']
                pred_f = r['pred_when_false']
                print(f"   {name:<30} {r['gt']:>4} {pred_t:>5.2f}  {pred_f:>5.2f}  {100*r['prec']:>4.0f}% {100*r['rec']:>4.0f}% {100*r['f1']:>4.0f}%")
            
            print(f"   {'-'*75}")
            print(f"   Legend: GT=count, PredT=mean pred when GT=True (want HIGH)")
            print(f"           PredF=mean pred when GT=False (want LOW), P/R/F1=metrics @{best_thresh}")
        
        # 4. RELATIONS - Full Per-Class Breakdown
        best_f1_rel = 0
        best_thresh_rel = 0.3
        relation_results = []
        print(f"\n🔗 RELATIONS ({len(all_relation_preds)} boxes):")
        if len(all_relation_preds) > 0:
            rel_gt_pos = all_relation_gts == True
            rel_gt_neg = all_relation_gts == False
            if rel_gt_pos.any():
                print(f"   When GT=True:  mean={all_relation_preds[rel_gt_pos].mean():.4f}")
            if rel_gt_neg.any():
                print(f"   When GT=False: mean={all_relation_preds[rel_gt_neg].mean():.4f}")
            
            # Find best threshold
            print(f"\n   Threshold Analysis:")
            for thresh in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
                pred_bin = all_relation_preds > thresh
                tp = (pred_bin & all_relation_gts).sum()
                fp = (pred_bin & ~all_relation_gts).sum()
                fn = (~pred_bin & all_relation_gts).sum()
                prec = tp / max(tp + fp, 1)
                rec = tp / max(tp + fn, 1)
                f1 = 2 * prec * rec / max(prec + rec, 0.001)
                if f1 > best_f1_rel:
                    best_f1_rel = f1
                    best_thresh_rel = thresh
                print(f"      @{thresh:.2f}: P={100*prec:.1f}% R={100*rec:.1f}% F1={100*f1:.1f}%")
            
            print(f"\n   ✅ Best: thresh={best_thresh_rel}, F1={100*best_f1_rel:.1f}%")
            
            # Per-relation breakdown
            print(f"\n   {'='*75}")
            print(f"   📦 PER-RELATION BREAKDOWN (all {self.num_relations} relations)")
            print(f"   {'='*75}")
            print(f"   {'Relation':<25} {'GT':>4} {'PredT':>6} {'PredF':>6} {'P':>5} {'R':>5} {'F1':>5}")
            print(f"   {'-'*75}")
            
            for rel_idx in range(self.num_relations):
                gt_count = all_relation_gts[:, rel_idx].sum()
                preds_this_rel = all_relation_preds[:, rel_idx]
                gt_this_rel = all_relation_gts[:, rel_idx]
                
                # Raw probability stats
                mean_when_true = preds_this_rel[gt_this_rel].mean() if gt_this_rel.any() else 0
                mean_when_false = preds_this_rel[~gt_this_rel].mean() if (~gt_this_rel).any() else 0
                
                # P/R/F1
                pred_bin = preds_this_rel > best_thresh_rel
                tp = (pred_bin & gt_this_rel).sum()
                fp = (pred_bin & ~gt_this_rel).sum()
                fn = (~pred_bin & gt_this_rel).sum()
                prec = tp / max(tp + fp, 1)
                rec = tp / max(tp + fn, 1)
                f1 = 2 * prec * rec / max(prec + rec, 0.001)
                
                rel_name = self.relation_names[rel_idx] if rel_idx < len(self.relation_names) else f"rel_{rel_idx}"
                relation_results.append({
                    'name': rel_name,
                    'gt': int(gt_count),
                    'pred_when_true': float(mean_when_true),
                    'pred_when_false': float(mean_when_false),
                    'prec': float(prec), 'rec': float(rec), 'f1': float(f1)
                })
            
            # Sort by F1 for display
            relation_results.sort(key=lambda x: -x['f1'])
            
            for r in relation_results:
                name = r['name'][:24] if len(r['name']) > 24 else r['name']
                pred_t = r['pred_when_true']
                pred_f = r['pred_when_false']
                print(f"   {name:<25} {r['gt']:>4} {pred_t:>5.2f}  {pred_f:>5.2f}  {100*r['prec']:>4.0f}% {100*r['rec']:>4.0f}% {100*r['f1']:>4.0f}%")
            
            print(f"   {'-'*75}")
            print(f"   Legend: GT=count, PredT=mean pred when GT=True (want HIGH)")
            print(f"           PredF=mean pred when GT=False (want LOW), P/R/F1=metrics @{best_thresh_rel}")
        
        # SUMMARY
        print(f"\n{'='*70}")
        print(f"📋 SUMMARY - Epoch {epoch}")
        print(f"{'='*70}")
        print(f"   Box Match:      {100*matched_boxes/max(total_gt_boxes,1):.1f}%")
        print(f"   Person Match:   {100*matched_person/max(total_person_gt,1):.1f}%")
        print(f"   Object Acc:     {100*obj_acc:.1f}%")
        if n_samples > 0:
            print(f"   Action Top-1:   {100*top1/n_samples:.1f}%")
            print(f"   Action Top-5:   {100*top5/n_samples:.1f}%")
        print(f"   Action F1:      {100*best_f1:.1f}% @{best_thresh}")
        print(f"   Relation F1:    {100*best_f1_rel:.1f}% @{best_thresh_rel}")
        print(f"   Time:           {elapsed:.1f}s")
        print(f"{'='*70}")
        
        # PRACTICAL AUTOMATION SCORES
        print(f"\n{'='*70}")
        print(f"🏠 PRACTICAL AUTOMATION SCORES - Epoch {epoch}")
        print(f"{'='*70}")
        
        # Calculate automation-ready scores per action
        automation_ready = []
        if len(action_results) > 0:
            for r in action_results:
                # An action is "automation ready" if:
                # - High precision (low false positives - don't trigger wrong automations)
                # - Reasonable recall (catches most occurrences)
                # - Good separation (pred_when_true >> pred_when_false)
                separation = r['pred_when_true'] - r['pred_when_false']
                score = r['prec'] * 0.4 + r['rec'] * 0.3 + separation * 0.3
                
                if r['gt'] >= 10:  # Only consider actions with enough samples
                    automation_ready.append({
                        'name': r['name'],
                        'gt': r['gt'],
                        'prec': r['prec'],
                        'rec': r['rec'],
                        'separation': separation,
                        'automation_score': score
                    })
            
            if automation_ready:
                automation_ready.sort(key=lambda x: -x['automation_score'])
                
                print(f"\n   🟢 READY FOR AUTOMATION (>70% score):")
                ready_count = 0
                for a in automation_ready:
                    if a['automation_score'] > 0.70:
                        ready_count += 1
                        print(f"      ✓ {a['name'][:30]:<32} Score: {100*a['automation_score']:.0f}%")
                
                if ready_count == 0:
                    print(f"      None yet - keep training!")
                
                print(f"\n   🟡 GETTING CLOSE (50-70% score):")
                close_count = 0
                for a in automation_ready:
                    if 0.50 <= a['automation_score'] <= 0.70:
                        close_count += 1
                        print(f"      ~ {a['name'][:30]:<32} Score: {100*a['automation_score']:.0f}% (P:{100*a['prec']:.0f}% R:{100*a['rec']:.0f}%)")
                
                if close_count == 0:
                    print(f"      None in this range")
                
                print(f"\n   🔴 NEEDS MORE TRAINING (<50% score):")
                need_count = 0
                for a in automation_ready[:10]:  # Top 10 only
                    if a['automation_score'] < 0.50:
                        need_count += 1
                        print(f"      ✗ {a['name'][:30]:<32} Score: {100*a['automation_score']:.0f}%")
                
                if need_count == 0:
                    print(f"      All actions above 50%!")
                
                # Overall readiness
                pct_ready = sum(1 for a in automation_ready if a['automation_score'] > 0.70) / len(automation_ready)
                pct_close = sum(1 for a in automation_ready if a['automation_score'] > 0.50) / len(automation_ready)
                
                print(f"\n   📊 OVERALL READINESS:")
                print(f"      Actions >70% (ready):  {100*pct_ready:.0f}% ({sum(1 for a in automation_ready if a['automation_score'] > 0.70)}/{len(automation_ready)})")
                print(f"      Actions >50% (usable): {100*pct_close:.0f}% ({sum(1 for a in automation_ready if a['automation_score'] > 0.50)}/{len(automation_ready)})")
        
        print(f"{'='*70}")
        
        # Save
        results = {
            'epoch': epoch,
            'box_match_rate': float(matched_boxes / max(total_gt_boxes, 1)),
            'person_match_rate': float(matched_person / max(total_person_gt, 1)),
            'object_accuracy': float(obj_acc),
            'action_top1': float(top1 / max(n_samples, 1)),
            'action_top5': float(top5 / max(n_samples, 1)),
            'action_best_f1': float(best_f1),
            'action_best_thresh': float(best_thresh),
            'relation_best_f1': float(best_f1_rel),
            'relation_best_thresh': float(best_thresh_rel),
            'iou_mean': float(all_ious.mean()) if len(all_ious) > 0 else 0,
            'per_object': object_results,
            'per_action': action_results,
            'per_relation': relation_results,
        }
        
        save_file = os.path.join(self.save_path, f'smart_home_eval_epoch_{epoch}.json')
        with open(save_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Saved: {save_file}")
        
        return results


if __name__ == "__main__":
    print("Smart Home Evaluator V2 loaded")
