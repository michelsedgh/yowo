import os
import gc

import torch
import torch.nn as nn

from dataset.ucf_jhmdb import UCF_JHMDB_Dataset
from dataset.ava import AVA_Dataset
from dataset.transforms import Augmentation, BaseTransform

from evaluator.ucf_jhmdb_evaluator import UCF_JHMDB_Evaluator
from evaluator.ava_evaluator import AVA_Evaluator


def build_dataset(d_cfg, args, is_train=False):
    """
        d_cfg: dataset config
    """
    # transform
    augmentation = Augmentation(
        img_size=d_cfg['train_size'],
        jitter=d_cfg['jitter'],
        hue=d_cfg['hue'],
        saturation=d_cfg['saturation'],
        exposure=d_cfg['exposure']
        )
    basetransform = BaseTransform(
        img_size=d_cfg['test_size'],
        )

    # dataset
    if args.dataset in ['ucf24', 'jhmdb21']:
        data_dir = os.path.join(args.root, 'ucf24')

        # dataset
        dataset = UCF_JHMDB_Dataset(
            data_root=data_dir,
            dataset=args.dataset,
            img_size=d_cfg['train_size'],
            transform=augmentation,
            is_train=is_train,
            len_clip=args.len_clip,
            sampling_rate=d_cfg['sampling_rate']
            )
        num_classes = dataset.num_classes

        # evaluator
        evaluator = UCF_JHMDB_Evaluator(
            data_root=data_dir,
            dataset=args.dataset,
            model_name=args.version,
            metric='fmap',
            img_size=d_cfg['test_size'],
            len_clip=args.len_clip,
            batch_size=args.test_batch_size,
            conf_thresh=0.01,
            iou_thresh=0.5,
            gt_folder=d_cfg['gt_folder'],
            save_path='./evaluator/eval_results/',
            transform=basetransform,
            collate_fn=CollateFunc()            
        )

    elif args.dataset == 'ava_v2.2':
        data_dir = os.path.join(args.root, 'AVA_Dataset')
        
        # dataset
        dataset = AVA_Dataset(
            cfg=d_cfg,
            data_root=data_dir,
            is_train=True,
            img_size=d_cfg['train_size'],
            transform=augmentation,
            len_clip=args.len_clip,
            sampling_rate=d_cfg['sampling_rate']
        )
        num_classes = 80

        # evaluator
        evaluator = AVA_Evaluator(
            d_cfg=d_cfg,
            data_root=data_dir,
            img_size=d_cfg['test_size'],
            len_clip=args.len_clip,
            sampling_rate=d_cfg['sampling_rate'],
            batch_size=args.test_batch_size,
            transform=basetransform,
            collate_fn=CollateFunc(),
            full_test_on_val=False,
            version='v2.2'
            )

    elif args.dataset == 'charades_ag':
        from dataset.charades_ag import CharadesAGDataset
        from evaluator.charades_ag_evaluator import CharadesAGEvaluator
        data_dir = os.path.join(args.root, 'ActionGenome')
        
        # dataset
        dataset = CharadesAGDataset(
            cfg=d_cfg,
            data_root=data_dir,
            is_train=is_train,
            img_size=d_cfg['train_size'],
            transform=augmentation,
            len_clip=args.len_clip,
            sampling_rate=d_cfg['sampling_rate']
        )
        num_classes = d_cfg['valid_num_classes']
        
        # evaluator
        evaluator = CharadesAGEvaluator(
            d_cfg=d_cfg,
            data_root=args.root,  # evaluator adds 'ActionGenome' internally
            img_size=d_cfg['test_size'],
            len_clip=args.len_clip,
            sampling_rate=d_cfg['sampling_rate'],
            batch_size=args.test_batch_size,
            transform=basetransform,
            collate_fn=CollateFunc(),
            conf_thresh=args.conf_thresh,
            iou_thresh=args.nms_thresh,
            save_path='./evaluator/eval_results/',
            num_workers=args.num_workers
        )

    elif args.dataset == 'smart_home':
        # Smart Home dataset - filtered Charades-AG
        from dataset.smart_home import SmartHomeDataset
        from evaluator.smart_home_evaluator_v2 import SmartHomeEvaluatorV2
        import json
        
        data_dir = os.path.join(args.root, 'ActionGenome')
        
        # Load smart home config for class weights
        config_path = os.path.join(os.path.dirname(__file__), '../config/smart_home_final.json')
        with open(config_path) as f:
            smart_home_config = json.load(f)
        
        # dataset
        dataset = SmartHomeDataset(
            cfg=d_cfg,
            data_root=data_dir,
            is_train=is_train,
            img_size=d_cfg['train_size'],
            transform=augmentation,
            len_clip=args.len_clip,
            sampling_rate=d_cfg['sampling_rate']
        )
        # Use config-driven class counts: 36 objects + N actions + 26 relations
        num_classes = d_cfg['num_objects'] + smart_home_config['num_actions'] + d_cfg['num_relations']
        
        # Store config in args for loss function to use
        args.smart_home_config = smart_home_config
        
        # evaluator - MEMORY FIX: Share base dataset to avoid loading ~40GB pickles twice
        evaluator = SmartHomeEvaluatorV2(
            d_cfg=d_cfg,
            data_root=args.root,
            img_size=d_cfg['test_size'],
            len_clip=args.len_clip,
            sampling_rate=d_cfg['sampling_rate'],
            batch_size=args.test_batch_size,
            transform=basetransform,
            collate_fn=CollateFunc(),
            conf_thresh=args.conf_thresh,
            iou_thresh=getattr(args, 'iou_thresh', 0.5),
            save_path='./evaluator/eval_results/',
            smart_home_config=smart_home_config,
            num_workers=args.num_workers,
            shared_base_dataset=dataset.base_dataset  # Share pickle data!
        )

    else:
        print('unknow dataset !! Only support ucf24 & jhmdb21 & ava_v2.2 & charades_ag & smart_home !!')
        exit(0)

    print('==============================')
    print('Training model on:', args.dataset)
    print('The dataset size:', len(dataset))

    if not args.eval:
        # no evaluator during training stage
        evaluator = None

    return dataset, evaluator, num_classes


def _worker_init_fn(worker_id):
    """Disable GC in forked workers to prevent copy-on-write memory explosion.
    
    Large pickle dicts (person_bboxes, object_data) are shared via CoW after fork.
    Python's GC walks all tracked objects, touching every page and triggering CoW.
    Disabling GC keeps pages shared. Reference counting still frees non-cyclic objects.
    """
    gc.disable()


def build_dataloader(args, dataset, batch_size, collate_fn=None, is_train=False):
    prefetch_factor = None
    if args.num_workers > 0:
        if getattr(args, 'prefetch_factor', None) is not None:
            prefetch_factor = args.prefetch_factor
        else:
            img_size = getattr(dataset, 'img_size', None)
            if img_size is None:
                img_size = getattr(args, 'img_size', None) or 224
            len_clip = getattr(dataset, 'len_clip', None)
            if len_clip is None:
                len_clip = getattr(args, 'len_clip', 16)

            estimated_sample_mb = 3 * len_clip * img_size * img_size / (1024 ** 2)
            estimated_inflight_gb = estimated_sample_mb * batch_size * args.num_workers * 2 / 1024
            prefetch_factor = 1 if estimated_inflight_gb > 8.0 else 2

    # Prevent copy-on-write memory explosion in forked worker processes.
    # gc.freeze() moves all currently tracked objects (especially the large pickle
    # dicts person_bboxes/object_data) to a permanent generation that GC never visits.
    # Combined with gc.disable() in workers, this keeps forked pages shared.
    if args.num_workers > 0:
        gc.collect()
        gc.freeze()

    worker_init = _worker_init_fn if args.num_workers > 0 else None
    
    # MEMORY FIX: Don't use persistent_workers if dataset has resample_negatives
    # because workers fork with a copy of filtered_indices that becomes stale.
    # Each epoch, workers must re-fork to see the updated indices.
    has_resample = hasattr(dataset, 'resample_negatives')
    use_persistent = args.num_workers > 0 and not has_resample
    
    if has_resample and args.num_workers > 0:
        print(f'Note: persistent_workers disabled (dataset has resample_negatives)')

    if is_train:
        # distributed
        if args.distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            sampler = torch.utils.data.RandomSampler(dataset)

        batch_sampler_train = torch.utils.data.BatchSampler(sampler, 
                                                            batch_size, 
                                                            drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, 
            batch_sampler=batch_sampler_train,
            collate_fn=collate_fn, 
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=use_persistent,
            prefetch_factor=prefetch_factor,
            worker_init_fn=worker_init
            )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, 
            shuffle=False,
            collate_fn=collate_fn, 
            num_workers=args.num_workers,
            drop_last=False,
            pin_memory=True,
            persistent_workers=use_persistent,
            prefetch_factor=prefetch_factor,
            worker_init_fn=worker_init
            )
    
    if args.num_workers > 0:
        print(f'DataLoader workers={args.num_workers}, prefetch_factor={prefetch_factor}, persistent={use_persistent}')

    return dataloader
    

def load_weight(model, path_to_ckpt=None):
    if path_to_ckpt is None:
        print('No trained weight ..')
        return model
        
    checkpoint = torch.load(path_to_ckpt, map_location='cpu', weights_only=False)
    # checkpoint state dict
    checkpoint_state_dict = checkpoint.pop("model")
    # model state dict
    model_state_dict = model.state_dict()
    # check
    for k in list(checkpoint_state_dict.keys()):
        if k in model_state_dict:
            shape_model = tuple(model_state_dict[k].shape)
            shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
            if shape_model != shape_checkpoint:
                checkpoint_state_dict.pop(k)
        else:
            checkpoint_state_dict.pop(k)
            print(k)

    model.load_state_dict(checkpoint_state_dict)
    print('Finished loading model!')

    return model


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


class CollateFunc(object):
    def __call__(self, batch):
        batch_frame_id = []
        batch_key_target = []
        batch_video_clips = []

        for sample in batch:
            key_frame_id = sample[0]
            video_clip = sample[1]
            key_target = sample[2]
            
            batch_frame_id.append(key_frame_id)
            batch_video_clips.append(video_clip)
            batch_key_target.append(key_target)

        # List [B, 3, T, H, W] -> [B, 3, T, H, W]
        batch_video_clips = torch.stack(batch_video_clips)
        
        return batch_frame_id, batch_video_clips, batch_key_target
