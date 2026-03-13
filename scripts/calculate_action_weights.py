#!/usr/bin/env python3
"""
Calculate proper per-class pos_weight for BCE loss from actual dataset statistics.

For multi-label BCE, each class needs its own pos_weight = num_negatives / num_positives
This balances the gradient so rare classes get appropriate learning signal.
"""

import json
import os

# From E5 evaluation logs (11448 person boxes total)
# Format: action_name -> GT count (number of positive samples)
TOTAL_PERSON_BOXES = 11448

ACTION_COUNTS = {
    "Closing a door": 440,
    "Opening a door": 568,
    "Sitting at a table": 1526,
    "Working at a table": 519,
    "Using a phone/camera": 1658,
    "Talking on a phone/camera": 301,
    "Reading/working on book or paper": 1689,
    "Closing a laptop": 56,
    "Opening a laptop": 106,
    "Working/Playing on a laptop": 659,
    "Putting on shoe/shoes": 119,
    "Taking off some shoes": 145,
    "Sitting in a chair": 2285,
    "Holding some food": 1497,
    "Taking food from somewhere": 626,
    "Snuggling with a blanket": 382,
    "Walking through a doorway": 869,
    "Drinking from a cup/glass/bottle": 899,
    "Holding a cup/glass/bottle": 1189,
    "Washing a dish/dishes": 121,
    "Lying down": 416,
    "Sitting on sofa/couch": 847,
    "Taking medicine": 254,
    "Watching television": 404,
    "Someone is awakening": 183,
    "Sitting in a bed": 537,
    "Holding a vacuum": 235,
    "Someone is cooking something": 258,
    "Someone is dressing": 319,
    "Someone is running somewhere": 277,
    "Someone is going from standing to sitting": 760,
    "Someone is sneezing": 408,
    "Someone is standing up from somewhere": 1157,
    "Someone is undressing": 328,
    "Someone is eating something": 1143,
}

# Action names in the order they appear in smart_home_final.json
ACTION_ORDER = [
    "Closing a door",
    "Opening a door",
    "Sitting at a table",
    "Working at a table",
    "Using a phone/camera",
    "Talking on a phone/camera",
    "Reading/working on book or paper",
    "Closing a laptop",
    "Opening a laptop",
    "Working/Playing on a laptop",
    "Putting on shoe/shoes",
    "Taking off some shoes",
    "Sitting in a chair",
    "Holding some food",
    "Taking food from somewhere",
    "Snuggling with a blanket",
    "Walking through a doorway",
    "Drinking from a cup/glass/bottle",
    "Holding a cup/glass/bottle",
    "Washing a dish/dishes",
    "Lying down",
    "Sitting on sofa/couch",
    "Taking medicine",
    "Watching television",
    "Someone is awakening",
    "Sitting in a bed",
    "Holding a vacuum",
    "Someone is cooking something",
    "Someone is dressing",
    "Someone is running somewhere",
    "Someone is going from standing to sitting",
    "Someone is sneezing",
    "Someone is standing up from somewhere",
    "Someone is undressing",
    "Someone is eating something",
]

def calculate_pos_weights():
    """Calculate per-class pos_weight = num_negatives / num_positives"""
    pos_weights = []
    
    print("=" * 80)
    print("PER-CLASS POS_WEIGHT CALCULATION")
    print("=" * 80)
    print(f"Total person boxes: {TOTAL_PERSON_BOXES}")
    print()
    print(f"{'Action':<45} {'GT':>6} {'Neg':>6} {'Ratio':>8} {'pos_weight':>10}")
    print("-" * 80)
    
    for action in ACTION_ORDER:
        gt_count = ACTION_COUNTS.get(action, 100)  # Default if not found
        neg_count = TOTAL_PERSON_BOXES - gt_count
        ratio = neg_count / max(gt_count, 1)
        
        # Cap pos_weight to prevent exploding gradients for very rare classes
        pos_weight = min(ratio, 50.0)
        pos_weights.append(round(pos_weight, 2))
        
        print(f"{action:<45} {gt_count:>6} {neg_count:>6} {ratio:>7.1f}:1 {pos_weight:>10.2f}")
    
    print("-" * 80)
    print()
    
    # Summary statistics
    print(f"Min pos_weight: {min(pos_weights):.2f} (most common action)")
    print(f"Max pos_weight: {max(pos_weights):.2f} (rarest action, capped)")
    print(f"Mean pos_weight: {sum(pos_weights)/len(pos_weights):.2f}")
    print()
    
    return pos_weights

def main():
    pos_weights = calculate_pos_weights()
    
    # Load existing config
    config_path = "/home/michel/yowo/config/smart_home_final.json"
    with open(config_path) as f:
        config = json.load(f)
    
    # Add/update pos_weights
    config["action_pos_weights"] = pos_weights
    config["action_pos_weights_formula"] = "min(num_negatives / num_positives, 50.0) from E5 eval data"
    
    # Save updated config
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Updated {config_path} with per-class pos_weights")
    print()
    print("pos_weights array for loss_multitask.py:")
    print(f"action_pos_weights = torch.tensor({pos_weights}, dtype=torch.float32)")

if __name__ == "__main__":
    main()
