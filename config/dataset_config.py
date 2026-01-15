# Dataset configuration


dataset_config = {
    'ucf24': {
        # dataset
        'gt_folder': './evaluator/groundtruths_ucf_jhmdb/groundtruths_ucf/',
        # input size
        'train_size': 224,
        'test_size': 224,
        # transform
        'jitter': 0.2,
        'hue': 0.1,
        'saturation': 1.5,
        'exposure': 1.5,
        'sampling_rate': 1,
        # cls label
        'multi_hot': False,  # one hot
        # optimizer
        'optimizer': 'adamw',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_iter': 500,
        # class names
        'valid_num_classes': 24,
        'label_map': (
                    'Basketball',     'BasketballDunk',    'Biking',            'CliffDiving',
                    'CricketBowling', 'Diving',            'Fencing',           'FloorGymnastics', 
                    'GolfSwing',      'HorseRiding',       'IceDancing',        'LongJump',
                    'PoleVault',      'RopeClimbing',      'SalsaSpin',         'SkateBoarding',
                    'Skiing',         'Skijet',            'SoccerJuggling',    'Surfing',
                    'TennisSwing',    'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog'
                ),
    },
    
    'ava_v2.2':{
        # ... (keep existing)
    },

    'charades_ag': {
        # dataset
        'data_root': 'data/ActionGenome/',
        'train_size': 224,
        'test_size': 224,
        # transform
        'jitter': 0.2,
        'hue': 0.1,
        'saturation': 1.5,
        'exposure': 1.5,
        # Sampling rate: how many frames to skip between each sampled frame
        # Original YOWO uses 1 (every consecutive frame) for all datasets
        # Note: During training, UCF/JHMDB randomly sample 1-2 for augmentation
        # For Charades with K=16 at 24fps: covers 16/24 = 0.67 seconds
        'sampling_rate': 1,
        # cls label
        'multi_hot': True,
        'multi_task': True,  # Enable three-head architecture
        # train config
        'optimizer': 'adamw',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_iter': 500,
        # class counts (for multi-task architecture)
        'num_objects': 36,    # AG objects (person + 35 objects)
        'num_actions': 157,   # Charades action classes
        'num_relations': 26,  # AG relationship classes
        'valid_num_classes': 219,  # Total: 36 + 157 + 26 = 219
    },

    'smart_home': {
        # Smart Home dataset - filtered Charades-AG with 42 actions
        # Same as charades_ag but with reduced action classes
        'data_root': 'data/ActionGenome/',
        'train_size': 224,
        'test_size': 224,
        # transform
        'jitter': 0.2,
        'hue': 0.1,
        'saturation': 1.5,
        'exposure': 1.5,
        'sampling_rate': 1,
        # cls label
        'multi_hot': True,
        'multi_task': True,
        # train config
        'optimizer': 'adamw',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_iter': 500,
        # class counts for Smart Home (filtered)
        'num_objects': 36,    # AG objects (unchanged)
        'num_actions': 42,    # Smart home action classes (was 157)
        'num_relations': 26,  # AG relationship classes (unchanged)
        'valid_num_classes': 104,  # Total: 36 + 42 + 26 = 104
    }
}

