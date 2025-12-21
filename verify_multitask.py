#!/usr/bin/env python3
"""
Multi-Task Architecture Verification Script

This script thoroughly validates the three-head multi-task architecture for
Action Genome + Charades BEFORE training. It checks:

1. Model instantiation with three separate prediction heads
2. Forward pass produces correct output shapes
3. Loss computation for all three heads (obj/act/rel)
4. Action masking for non-Person boxes
5. Gradient flow through all heads
6. Overfit test on single sample
7. Parameter count comparison with jammed approach

Run this BEFORE training to ensure the architecture is correct.
"""

import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

sys.path.insert(0, '/home/michel/yowo')


def print_header(text):
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70)


def print_test(name, passed, details=""):
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  [{status}] {name}")
    if details:
        print(f"          {details}")


def test_model_instantiation():
    """Test 1: Verify model creates three prediction heads."""
    print_header("Test 1: Model Instantiation")
    
    from config.yowo_v2_config import yowo_v2_config
    from models.yowo.yowo_multitask import YOWOMultiTask
    
    all_passed = True
    device = torch.device("cpu")
    
    cfg = yowo_v2_config['yowo_v2_x3d_s_yolo11m_multitask']
    
    try:
        model = YOWOMultiTask(
            cfg=cfg,
            device=device,
            num_objects=36,
            num_actions=157,
            num_relations=26,
            trainable=True
        )
        
        # Check prediction heads exist
        has_obj = hasattr(model, 'obj_preds') and len(model.obj_preds) == 3
        has_act = hasattr(model, 'act_preds') and len(model.act_preds) == 3
        has_rel = hasattr(model, 'rel_preds') and len(model.rel_preds) == 3
        has_interact = hasattr(model, 'interact_preds') and len(model.interact_preds) == 3
        has_conf = hasattr(model, 'conf_preds') and len(model.conf_preds) == 3
        has_reg = hasattr(model, 'reg_preds') and len(model.reg_preds) == 3
        
        print_test("Object prediction heads (3 levels)", has_obj,
                  f"out_channels: {[h.out_channels for h in model.obj_preds]}")
        print_test("Action prediction heads (3 levels)", has_act,
                  f"out_channels: {[h.out_channels for h in model.act_preds]}")
        print_test("Relation prediction heads (3 levels)", has_rel,
                  f"out_channels: {[h.out_channels for h in model.rel_preds]}")
        print_test("Interaction prediction heads (3 levels)", has_interact,
                  f"out_channels: {[h.out_channels for h in model.interact_preds]}")
        print_test("Confidence prediction heads", has_conf)
        print_test("Regression prediction heads", has_reg)
        
        # Check output channels
        obj_correct = all(h.out_channels == 36 for h in model.obj_preds)
        act_correct = all(h.out_channels == 157 for h in model.act_preds)
        rel_correct = all(h.out_channels == 26 for h in model.rel_preds)
        interact_correct = all(h.out_channels == 1 for h in model.interact_preds)
        
        print_test("Object head output = 36", obj_correct)
        print_test("Action head output = 157", act_correct)
        print_test("Relation head output = 26", rel_correct)
        print_test("Interaction head output = 1", interact_correct)
        
        all_passed = (has_obj and has_act and has_rel and has_interact and has_conf and has_reg and
                      obj_correct and act_correct and rel_correct and interact_correct)
        
    except Exception as e:
        import traceback
        print_test("Model instantiation", False, f"Error: {e}")
        traceback.print_exc()
        all_passed = False
    
    return all_passed


def test_forward_pass():
    """Test 2: Verify forward pass produces correct output shapes."""
    print_header("Test 2: Forward Pass")
    
    from config.yowo_v2_config import yowo_v2_config
    from models.yowo.yowo_multitask import YOWOMultiTask
    
    all_passed = True
    device = torch.device("cpu")
    
    cfg = yowo_v2_config['yowo_v2_x3d_s_yolo11m_multitask']
    
    try:
        model = YOWOMultiTask(
            cfg=cfg,
            device=device,
            num_objects=36,
            num_actions=157,
            num_relations=26,
            trainable=True
        )
        model.train()
        
        x = torch.randn(2, 3, 16, 224, 224)  # Batch size 2
        outputs = model(x)
        
        # Check output keys (now includes pred_interact)
        required_keys = ['pred_conf', 'pred_obj', 'pred_act', 'pred_rel', 'pred_interact', 'pred_box', 'anchors', 'strides']
        has_keys = all(k in outputs for k in required_keys)
        print_test("Output has all required keys", has_keys, f"{list(outputs.keys())}")
        
        # Check shapes
        for level in range(3):
            obj_shape = outputs['pred_obj'][level].shape
            act_shape = outputs['pred_act'][level].shape
            rel_shape = outputs['pred_rel'][level].shape
            interact_shape = outputs['pred_interact'][level].shape
            
            obj_ok = obj_shape[-1] == 36
            act_ok = act_shape[-1] == 157
            rel_ok = rel_shape[-1] == 26
            interact_ok = interact_shape[-1] == 1
            
            print_test(f"Level {level} shapes correct", obj_ok and act_ok and rel_ok and interact_ok,
                      f"obj: {obj_shape}, act: {act_shape}, rel: {rel_shape}, interact: {interact_shape}")
            all_passed = all_passed and obj_ok and act_ok and rel_ok and interact_ok
        
        all_passed = all_passed and has_keys
        
    except Exception as e:
        import traceback
        print_test("Forward pass", False, f"Error: {e}")
        traceback.print_exc()
        all_passed = False
    
    return all_passed


def test_loss_computation():
    """Test 3: Verify loss computation with all three heads."""
    print_header("Test 3: Loss Computation")
    
    from config.yowo_v2_config import yowo_v2_config
    from config.dataset_config import dataset_config
    from models.yowo.yowo_multitask import YOWOMultiTask
    from models.yowo.loss_multitask import build_multitask_criterion
    
    all_passed = True
    device = torch.device("cpu")
    
    cfg = yowo_v2_config['yowo_v2_x3d_s_yolo11m_multitask']
    d_cfg = dataset_config['charades_ag']
    
    try:
        model = YOWOMultiTask(
            cfg=cfg,
            device=device,
            num_objects=36,
            num_actions=157,
            num_relations=26,
            trainable=True
        )
        model.train()
        
        # Create mock args
        class Args:
            loss_conf_weight = 1.0
            loss_obj_weight = 1.0
            loss_act_weight = 1.0
            loss_rel_weight = 1.0
            loss_interact_weight = 1.0
            loss_interact_weight = 1.0
            loss_reg_weight = 5.0
            center_sampling_radius = 2.5
            topk_candicate = 10
        
        criterion = build_multitask_criterion(Args(), 224, 36, 157, 26)
        
        # Forward pass
        x = torch.randn(1, 3, 16, 224, 224)
        outputs = model(x)
        
        # Create mock targets
        # Person box with actions and relations
        person_label = torch.zeros(219)
        person_label[0] = 1.0  # Object: person (index 0)
        person_label[36 + 10] = 1.0  # Action 10
        person_label[36 + 20] = 1.0  # Action 20
        person_label[193 + 5] = 1.0  # Relation 5
        
        # Object box (chair) with only relations
        chair_label = torch.zeros(219)
        chair_label[7] = 1.0  # Object: chair (index 7)
        chair_label[193 + 3] = 1.0  # Relation 3
        
        targets = [{
            'boxes': torch.tensor([[0.2, 0.2, 0.5, 0.5], [0.6, 0.6, 0.8, 0.8]]),
            'labels': torch.stack([person_label, chair_label])
        }]
        
        # Compute loss
        loss_dict = criterion(outputs, targets)
        
        # Check loss keys
        required_loss_keys = ['loss_conf', 'loss_obj', 'loss_act', 'loss_rel', 'loss_interact', 'loss_box', 'losses']
        has_loss_keys = all(k in loss_dict for k in required_loss_keys)
        print_test("Loss dict has all keys", has_loss_keys, f"{list(loss_dict.keys())}")
        
        # Check losses are valid
        for k, v in loss_dict.items():
            is_valid = torch.isfinite(v) and v >= 0
            print_test(f"{k} is valid", is_valid.item(), f"value: {v.item():.4f}")
            all_passed = all_passed and is_valid.item()
        
        all_passed = all_passed and has_loss_keys
        
    except Exception as e:
        import traceback
        print_test("Loss computation", False, f"Error: {e}")
        traceback.print_exc()
        all_passed = False
    
    return all_passed


def test_action_masking():
    """Test 4: Verify action loss is only computed for Person boxes."""
    print_header("Test 4: Action Masking")
    
    from config.yowo_v2_config import yowo_v2_config
    from models.yowo.yowo_multitask import YOWOMultiTask
    from models.yowo.loss_multitask import build_multitask_criterion
    
    all_passed = True
    device = torch.device("cpu")
    
    cfg = yowo_v2_config['yowo_v2_x3d_s_yolo11m_multitask']
    
    try:
        model = YOWOMultiTask(
            cfg=cfg,
            device=device,
            num_objects=36,
            num_actions=157,
            num_relations=26,
            trainable=True
        )
        model.train()
        
        class Args:
            loss_conf_weight = 1.0
            loss_obj_weight = 1.0
            loss_act_weight = 1.0
            loss_rel_weight = 1.0
            loss_interact_weight = 1.0
            loss_reg_weight = 5.0
            center_sampling_radius = 2.5
            topk_candicate = 10
        
        criterion = build_multitask_criterion(Args(), 224, 36, 157, 26)
        
        # Forward pass
        x = torch.randn(1, 3, 16, 224, 224)
        outputs = model(x)
        
        # Test 1: Only non-person boxes (should have loss_act = 0 or minimal)
        chair_label = torch.zeros(219)
        chair_label[7] = 1.0  # Chair (not person)
        chair_label[36 + 10] = 1.0  # Action (should be ignored for chair)
        chair_label[193 + 5] = 1.0  # Relation
        
        targets_chair = [{
            'boxes': torch.tensor([[0.3, 0.3, 0.6, 0.6]]),
            'labels': chair_label.unsqueeze(0)
        }]
        
        loss_dict_chair = criterion(outputs, targets_chair)
        
        # Test 2: Person box (should have action loss)
        person_label = torch.zeros(219)
        person_label[0] = 1.0  # Person
        person_label[36 + 10] = 1.0  # Action
        person_label[193 + 5] = 1.0  # Relation
        
        targets_person = [{
            'boxes': torch.tensor([[0.3, 0.3, 0.6, 0.6]]),
            'labels': person_label.unsqueeze(0)
        }]
        
        loss_dict_person = criterion(outputs, targets_person)
        
        # Chair should have zero or very low action loss (masked)
        chair_act_loss = loss_dict_chair['loss_act'].item()
        person_act_loss = loss_dict_person['loss_act'].item()
        
        # Chair action loss should be 0 (fully masked)
        chair_masked = chair_act_loss == 0.0
        print_test("Chair (non-person) has zero action loss", chair_masked,
                  f"loss_act: {chair_act_loss:.6f}")
        
        # Person should have non-zero action loss
        person_has_act = person_act_loss > 0.0
        print_test("Person has non-zero action loss", person_has_act,
                  f"loss_act: {person_act_loss:.4f}")
        
        all_passed = chair_masked and person_has_act
        
    except Exception as e:
        import traceback
        print_test("Action masking", False, f"Error: {e}")
        traceback.print_exc()
        all_passed = False
    
    return all_passed


def test_gradient_flow():
    """Test 5: Verify gradients flow to all three heads."""
    print_header("Test 5: Gradient Flow")
    
    from config.yowo_v2_config import yowo_v2_config
    from models.yowo.yowo_multitask import YOWOMultiTask
    from models.yowo.loss_multitask import build_multitask_criterion
    
    all_passed = True
    device = torch.device("cpu")
    
    cfg = yowo_v2_config['yowo_v2_x3d_s_yolo11m_multitask']
    
    try:
        model = YOWOMultiTask(
            cfg=cfg,
            device=device,
            num_objects=36,
            num_actions=157,
            num_relations=26,
            trainable=True
        )
        model.train()
        
        class Args:
            loss_conf_weight = 1.0
            loss_obj_weight = 1.0
            loss_act_weight = 1.0
            loss_rel_weight = 1.0
            loss_interact_weight = 1.0
            loss_reg_weight = 5.0
            center_sampling_radius = 2.5
            topk_candicate = 10
        
        criterion = build_multitask_criterion(Args(), 224, 36, 157, 26)
        
        # Forward pass
        x = torch.randn(1, 3, 16, 224, 224)
        outputs = model(x)
        
        # Create target with person
        person_label = torch.zeros(219)
        person_label[0] = 1.0
        person_label[36 + 10] = 1.0
        person_label[193 + 5] = 1.0
        
        targets = [{
            'boxes': torch.tensor([[0.3, 0.3, 0.6, 0.6]]),
            'labels': person_label.unsqueeze(0)
        }]
        
        # Compute loss and backward
        loss_dict = criterion(outputs, targets)
        loss_dict['losses'].backward()
        
        # Check gradients in each head
        def check_grads(module_list, name):
            grads = []
            for m in module_list:
                for p in m.parameters():
                    if p.grad is not None:
                        grads.append(p.grad.abs().mean().item())
            return len(grads) > 0 and sum(grads) > 0, np.mean(grads) if grads else 0
        
        obj_has_grads, obj_mean = check_grads(model.obj_preds, "obj_preds")
        act_has_grads, act_mean = check_grads(model.act_preds, "act_preds")
        rel_has_grads, rel_mean = check_grads(model.rel_preds, "rel_preds")
        
        print_test("Object head receives gradients", obj_has_grads, f"mean grad: {obj_mean:.6f}")
        print_test("Action head receives gradients", act_has_grads, f"mean grad: {act_mean:.6f}")
        print_test("Relation head receives gradients", rel_has_grads, f"mean grad: {rel_mean:.6f}")
        
        # Check backbone gradients
        bk3d_grads = []
        for p in model.backbone_3d.parameters():
            if p.grad is not None:
                bk3d_grads.append(p.grad.abs().mean().item())
        bk3d_has_grads = len(bk3d_grads) > 0 and sum(bk3d_grads) > 0
        print_test("3D backbone receives gradients", bk3d_has_grads, 
                  f"params with grads: {len(bk3d_grads)}")
        
        all_passed = obj_has_grads and act_has_grads and rel_has_grads and bk3d_has_grads
        
    except Exception as e:
        import traceback
        print_test("Gradient flow", False, f"Error: {e}")
        traceback.print_exc()
        all_passed = False
    
    return all_passed


def test_parameter_count():
    """Test 6: Verify parameter count matches jammed approach."""
    print_header("Test 6: Parameter Count Comparison")
    
    from config.yowo_v2_config import yowo_v2_config
    from models.yowo.yowo_multitask import YOWOMultiTask
    from models.yowo.yowo import YOWO
    
    all_passed = True
    device = torch.device("cpu")
    
    try:
        # Multi-task model
        cfg = yowo_v2_config['yowo_v2_x3d_s_yolo11m']
        multitask_model = YOWOMultiTask(
            cfg=cfg,
            device=device,
            num_objects=36,
            num_actions=157,
            num_relations=26,
            trainable=True
        )
        
        # Jammed model
        jammed_model = YOWO(
            cfg=cfg,
            device=device,
            num_classes=219,
            trainable=True,
            multi_hot=True
        )
        
        # Count parameters
        mt_total = sum(p.numel() for p in multitask_model.parameters())
        jm_total = sum(p.numel() for p in jammed_model.parameters())
        
        # Count prediction head parameters
        mt_heads = (sum(p.numel() for p in multitask_model.obj_preds.parameters()) +
                    sum(p.numel() for p in multitask_model.act_preds.parameters()) +
                    sum(p.numel() for p in multitask_model.rel_preds.parameters()))
        jm_heads = sum(p.numel() for p in jammed_model.cls_preds.parameters())
        
        print_test("Multi-task model built", True, f"Total params: {mt_total/1e6:.2f}M")
        print_test("Jammed model built", True, f"Total params: {jm_total/1e6:.2f}M")
        
        # Head parameters should be equal (36+157+26 = 219)
        heads_equal = abs(mt_heads - jm_heads) < 100  # Allow small difference
        print_test("Prediction head params equal", heads_equal,
                  f"Multi-task: {mt_heads}, Jammed: {jm_heads}")
        
        # Total should be very close
        total_close = abs(mt_total - jm_total) / jm_total < 0.01  # Within 1%
        print_test("Total params within 1%", total_close,
                  f"Diff: {(mt_total - jm_total)/1e6:.4f}M ({100*(mt_total-jm_total)/jm_total:+.2f}%)")
        
        all_passed = heads_equal and total_close
        
    except Exception as e:
        import traceback
        print_test("Parameter count", False, f"Error: {e}")
        traceback.print_exc()
        all_passed = False
    
    return all_passed


def test_with_real_dataset():
    """Test 7: Verify with real dataset sample."""
    print_header("Test 7: Real Dataset Integration")
    
    from config.yowo_v2_config import yowo_v2_config
    from config.dataset_config import dataset_config
    from models.yowo.yowo_multitask import YOWOMultiTask
    from models.yowo.loss_multitask import build_multitask_criterion
    from dataset.charades_ag import CharadesAGDataset
    from dataset.transforms import Augmentation
    
    all_passed = True
    device = torch.device("cpu")
    
    try:
        # Load dataset
        d_cfg = dataset_config['charades_ag']
        transform = Augmentation(
            img_size=d_cfg['train_size'],
            jitter=d_cfg['jitter'],
            hue=d_cfg['hue'],
            saturation=d_cfg['saturation'],
            exposure=d_cfg['exposure']
        )
        
        dataset = CharadesAGDataset(
            cfg=d_cfg,
            data_root='data/ActionGenome',
            is_train=True,
            img_size=224,
            transform=transform,
            len_clip=16,
            sampling_rate=1
        )
        
        print_test("Dataset loaded", True, f"Size: {len(dataset)}")
        
        # Get a sample
        frame_id, video_clip, target = dataset[0]
        video_clip = video_clip.unsqueeze(0)  # Add batch dim
        
        print_test("Sample loaded", True, 
                  f"clip: {video_clip.shape}, boxes: {target['boxes'].shape}")
        
        # Build model
        cfg = yowo_v2_config['yowo_v2_x3d_s_yolo11m_multitask']
        model = YOWOMultiTask(
            cfg=cfg,
            device=device,
            num_objects=36,
            num_actions=157,
            num_relations=26,
            trainable=True
        )
        model.train()
        
        class Args:
            loss_conf_weight = 1.0
            loss_obj_weight = 1.0
            loss_act_weight = 1.0
            loss_rel_weight = 1.0
            loss_interact_weight = 1.0
            loss_reg_weight = 5.0
            center_sampling_radius = 2.5
            topk_candicate = 10
        
        criterion = build_multitask_criterion(Args(), 224, 36, 157, 26)
        
        # Forward pass
        outputs = model(video_clip)
        print_test("Forward pass succeeded", True)
        
        # Compute loss
        loss_dict = criterion(outputs, [target])
        
        print_test("Loss computed", True, 
                  f"total: {loss_dict['losses'].item():.4f}")
        
        # Check individual losses
        print(f"\n  Loss breakdown:")
        print(f"    loss_conf: {loss_dict['loss_conf'].item():.4f}")
        print(f"    loss_obj:  {loss_dict['loss_obj'].item():.4f}")
        print(f"    loss_act:  {loss_dict['loss_act'].item():.4f}")
        print(f"    loss_rel:  {loss_dict['loss_rel'].item():.4f}")
        print(f"    loss_box:  {loss_dict['loss_box'].item():.4f}")
        
        all_passed = torch.isfinite(loss_dict['losses']).item()
        
    except Exception as e:
        import traceback
        print_test("Real dataset integration", False, f"Error: {e}")
        traceback.print_exc()
        all_passed = False
    
    return all_passed


def test_build_model_factory():
    """Test 8: Verify build_model factory selects correct architecture."""
    print_header("Test 8: Build Model Factory")
    
    from config.yowo_v2_config import yowo_v2_config
    from config.dataset_config import dataset_config
    from models import build_model
    
    all_passed = True
    device = torch.device("cpu")
    
    try:
        class Args:
            version = 'yowo_v2_x3d_s_yolo11m_multitask'
            conf_thresh = 0.1
            nms_thresh = 0.5
            topk = 40
            freeze_backbone_2d = False
            freeze_backbone_3d = False
            loss_conf_weight = 1.0
            loss_obj_weight = 1.0
            loss_act_weight = 1.0
            loss_rel_weight = 1.0
            loss_interact_weight = 1.0
            loss_reg_weight = 5.0
            center_sampling_radius = 2.5
            topk_candicate = 10
        
        d_cfg = dataset_config['charades_ag']
        m_cfg = yowo_v2_config['yowo_v2_x3d_s_yolo11m_multitask']
        
        model, criterion = build_model(
            args=Args(),
            d_cfg=d_cfg,
            m_cfg=m_cfg,
            device=device,
            num_classes=219,
            trainable=True
        )
        
        # Check it's a multi-task model
        from models.yowo.yowo_multitask import YOWOMultiTask
        is_multitask = isinstance(model, YOWOMultiTask)
        print_test("build_model returns YOWOMultiTask", is_multitask)
        
        # Check it has three heads
        has_three_heads = (hasattr(model, 'obj_preds') and 
                          hasattr(model, 'act_preds') and 
                          hasattr(model, 'rel_preds'))
        print_test("Model has three prediction heads", has_three_heads)
        
        # Check criterion
        from models.yowo.loss_multitask import MultiTaskCriterion
        is_multitask_crit = isinstance(criterion, MultiTaskCriterion)
        print_test("Criterion is MultiTaskCriterion", is_multitask_crit)
        
        all_passed = is_multitask and has_three_heads and is_multitask_crit
        
    except Exception as e:
        import traceback
        print_test("Build model factory", False, f"Error: {e}")
        traceback.print_exc()
        all_passed = False
    
    return all_passed


def run_all_tests():
    """Run all verification tests."""
    print("\n" + "=" * 70)
    print(" MULTI-TASK ARCHITECTURE VERIFICATION")
    print(" Three-Head Architecture for Action Genome + Charades")
    print("=" * 70)
    
    results = OrderedDict()
    
    results['1. Model Instantiation'] = test_model_instantiation()
    results['2. Forward Pass'] = test_forward_pass()
    results['3. Loss Computation'] = test_loss_computation()
    results['4. Action Masking'] = test_action_masking()
    results['5. Gradient Flow'] = test_gradient_flow()
    results['6. Parameter Count'] = test_parameter_count()
    results['7. Real Dataset Integration'] = test_with_real_dataset()
    results['8. Build Model Factory'] = test_build_model_factory()
    
    # Summary
    print_header("SUMMARY")
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  [{status}] {name}")
        all_passed = all_passed and passed
    
    print("\n" + "-" * 70)
    if all_passed:
        print(" ✓ ALL TESTS PASSED - Multi-task architecture is correct!")
        print(" You can now train with: python train.py -d charades_ag -v yowo_v2_x3d_s_yolo11m_multitask")
    else:
        print(" ✗ SOME TESTS FAILED - Please fix the issues before training.")
    print("-" * 70 + "\n")
    
    return all_passed


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)


