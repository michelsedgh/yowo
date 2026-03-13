#!/usr/bin/env python3
"""
Calculate proper per-class pos_weight for BCE loss for RELATIONS.
"""

import json

# From E3 evaluation logs (29013 total boxes with relations)
TOTAL_BOXES = 29013

# Per-class GT counts from E3 logs
RELATION_COUNTS = {
    "lookingat": 13317,
    "notlookingat": 17282,
    "unsure": 500,  # estimated
    "above": 800,   # estimated
    "beneath": 6943,
    "infrontof": 20284,
    "behind": 6031,
    "onthesideof": 6967,
    "in": 1500,     # estimated
    "carrying": 800,  # estimated
    "coveredby": 500,  # estimated
    "drinkingfrom": 400,  # estimated
    "eating": 600,  # estimated
    "haveitontheback": 200,  # estimated
    "holding": 13539,
    "leaningon": 1000,  # estimated
    "lyingon": 800,  # estimated
    "notcontacting": 9711,
    "otherrelationship": 300,  # estimated
    "sittingon": 5470,
    "standingon": 1200,  # estimated
    "touching": 2000,  # estimated
    "twisting": 100,  # estimated
    "wearing": 1500,  # estimated
    "wiping": 200,  # estimated
    "writingon": 300,  # estimated
}

# Relation names in order (from relationship_classes.txt)
RELATION_ORDER = [
    "lookingat",
    "notlookingat", 
    "unsure",
    "above",
    "beneath",
    "infrontof",
    "behind",
    "onthesideof",
    "in",
    "carrying",
    "coveredby",
    "drinkingfrom",
    "eating",
    "haveitontheback",
    "holding",
    "leaningon",
    "lyingon",
    "notcontacting",
    "otherrelationship",
    "sittingon",
    "standingon",
    "touching",
    "twisting",
    "wearing",
    "wiping",
    "writingon",
]

def calculate_pos_weights():
    """Calculate per-class pos_weight = num_negatives / num_positives"""
    pos_weights = []
    
    print("=" * 80)
    print("PER-CLASS RELATION POS_WEIGHT CALCULATION")
    print("=" * 80)
    print(f"Total boxes: {TOTAL_BOXES}")
    print()
    print(f"{'Relation':<25} {'GT':>6} {'Neg':>6} {'Ratio':>8} {'pos_weight':>10}")
    print("-" * 60)
    
    for rel in RELATION_ORDER:
        gt_count = RELATION_COUNTS.get(rel, 500)
        neg_count = TOTAL_BOXES - gt_count
        ratio = neg_count / max(gt_count, 1)
        
        # Cap pos_weight to prevent exploding gradients
        pos_weight = min(ratio, 50.0)
        pos_weights.append(round(pos_weight, 2))
        
        print(f"{rel:<25} {gt_count:>6} {neg_count:>6} {ratio:>7.1f}:1 {pos_weight:>10.2f}")
    
    print("-" * 60)
    print()
    print(f"Min pos_weight: {min(pos_weights):.2f}")
    print(f"Max pos_weight: {max(pos_weights):.2f}")
    print(f"Mean pos_weight: {sum(pos_weights)/len(pos_weights):.2f}")
    print()
    
    return pos_weights

def main():
    pos_weights = calculate_pos_weights()
    
    # Update smart_home config
    config_path = "/home/michel/yowo/config/smart_home_final.json"
    with open(config_path) as f:
        config = json.load(f)
    
    config["relation_pos_weights"] = pos_weights
    config["relation_pos_weights_formula"] = "min(num_negatives / num_positives, 50.0)"
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Updated {config_path}")
    print()
    print("relation_pos_weights array:")
    print(f"torch.tensor({pos_weights}, dtype=torch.float32)")

if __name__ == "__main__":
    main()
