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
        # X3D-M was pretrained at 5fps temporal sampling (X3D-S uses 6fps)
        # Charades is 24fps, so sampling_rate=5 gives 24/5=4.8fps ≈ 5fps (matches X3D-M)
        # With len_clip=16: covers 16*5/24 = 3.33 seconds (~27% of avg Charades action)
        # This is better for long Charades actions + Action Genome relationship understanding
        'sampling_rate': 5,
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
    }
}
