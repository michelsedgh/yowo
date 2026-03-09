import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import os
import time
import argparse
from copy import deepcopy
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import distributed_utils
from utils.com_flops_params import FLOPs_and_Params
from utils.misc import CollateFunc, build_dataset, build_dataloader
from utils.solver.optimizer import build_optimizer
from utils.solver.warmup_schedule import build_warmup



from config import build_dataset_config, build_model_config
from models import build_model


GLOBAL_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description='YOWOv2')
    # CUDA
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')

    # Visualization
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--save_folder', default='./weights/', type=str, 
                        help='path to save weight')
    parser.add_argument('--vis_data', action='store_true', default=False,
                        help='use tensorboard')

    # Evaluation
    parser.add_argument('--eval', action='store_true', default=False, 
                        help='do evaluation during training.')
    parser.add_argument('--eval_epoch', default=3, type=int, 
                        help='evaluate every N epochs (default: 3)')
    parser.add_argument('--save_dir', default='inference_results/',
                        type=str, help='save inference results.')
    parser.add_argument('--eval_first', action='store_true', default=False,
                        help='evaluate model before training.')

    # Batchsize
    parser.add_argument('-bs', '--batch_size', default=16, type=int, 
                        help='batch size on a single GPU.')
    parser.add_argument('-tbs', '--test_batch_size', default=16, type=int, 
                        help='test batch size on a single GPU.')
    parser.add_argument('-accu', '--accumulate', default=1, type=int, 
                        help='gradient accumulate.')
    parser.add_argument('-lr', '--base_lr', default=0.0001, type=float, 
                        help='base lr.')
    parser.add_argument('-ldr', '--lr_decay_ratio', default=0.5, type=float, 
                        help='base lr.')

    # Epoch
    parser.add_argument('--max_epoch', default=10, type=int, 
                        help='max epoch.')
    parser.add_argument('--lr_epoch', nargs='+', default=[2,3,4], type=int,
                        help='lr epoch to decay')

    # Model
    parser.add_argument('-v', '--version', default='yowo_v2_tiny', type=str,
                        help='build YOWOv2')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='keep training')
    parser.add_argument('-ct', '--conf_thresh', default=0.05, type=float,
                        help='confidence threshold. We suggest 0.005 for UCF24 and 0.1 for AVA.')
    parser.add_argument('-nt', '--nms_thresh', default=0.5, type=float,
                        help='NMS threshold. We suggest 0.5 for UCF24 and AVA.')
    parser.add_argument('--iou_thresh', default=0.5, type=float,
                        help='IoU threshold for evaluation box matching. 0.5 standard, 0.3 for smart home.')
    parser.add_argument('--topk', default=40, type=int,
                        help='topk prediction candidates.')
    parser.add_argument('-K', '--len_clip', default=16, type=int,
                        help='video clip length.')
    parser.add_argument('--freeze_backbone_2d', action="store_true", default=False,
                        help="freeze 2D backbone.")
    parser.add_argument('--freeze_backbone_3d', action="store_true", default=False,
                        help="freeze 3d backbone.")
    parser.add_argument('-m', '--memory', action="store_true", default=False,
                        help="memory propagate.")

    # Dataset
    parser.add_argument('-d', '--dataset', default='ucf24',
                        help='ucf24, ava_v2.2')
    parser.add_argument('--root', default='/mnt/share/ssd2/dataset/STAD/',
                        help='data root')
    parser.add_argument('--num_workers', default=8, type=int, 
                        help='Number of workers used in dataloading (8+ recommended for A100)')
    parser.add_argument('-size', '--img_size', default=None, type=int,
                        help='Override train/test image size (e.g., 320, 480, 640). If not set, uses dataset config default.')
    parser.add_argument('--prefetch_factor', default=None, type=int,
                        help='Override DataLoader prefetch factor. If not set, an adaptive value is used.')

    # Matcher
    parser.add_argument('--center_sampling_radius', default=2.5, type=float, 
                        help='conf loss weight factor.')
    parser.add_argument('--topk_candicate', default=10, type=int, 
                        help='cls loss weight factor.')

    # Loss
    parser.add_argument('--loss_conf_weight', default=1, type=float, 
                        help='conf loss weight factor.')
    parser.add_argument('--loss_cls_weight', default=1, type=float, 
                        help='cls loss weight factor.')
    parser.add_argument('--loss_reg_weight', default=5, type=float, 
                        help='reg loss weight factor.')
    parser.add_argument('-fl', '--focal_loss', action="store_true", default=False,
                        help="use focal loss for classification (OLD - kept for compatibility).")
    parser.add_argument('--no_focal_loss', action="store_true", default=False,
                        help="Disable Focal Loss for action/relation heads. By default, Focal Loss is ENABLED and handles class imbalance automatically.")
    parser.add_argument('--end2end', action="store_true", default=False,
                        help="Enable NMS-free dual-head training (O2M + O2O heads). Use for YOLO26-style NMS-free inference.")
    parser.add_argument('--label_smoothing', default=0.0, type=float,
                        help='Label smoothing factor for action/relation heads. 0.0=disabled, 0.1=10% smoothing.')
    
    # DDP train
    parser.add_argument('-dist', '--distributed', action='store_true', default=False,
                        help='distributed training')
    parser.add_argument('--dist_url', default='env://', 
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--sybn', action='store_true', default=False, 
                        help='use sybn.')
    
    # Mixed Precision Training
    parser.add_argument('--amp', action='store_true', default=False,
                        help='use Automatic Mixed Precision (AMP) for faster training and lower memory.')
    parser.add_argument('--profile_iters', default=0, type=int,
                        help='Profile the first N train iterations with detailed stage timing.')

    return parser.parse_args()


def train():
    args = parse_args()
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")

    # dist
    world_size = distributed_utils.get_world_size()
    per_gpu_batch = args.batch_size // world_size
    print('World size: {}'.format(world_size))
    if args.distributed:
        distributed_utils.init_distributed_mode(args)
        print("git:\n  {}\n".format(distributed_utils.get_sha()))

    # path to save model
    path_to_save = os.path.join(args.save_folder, args.dataset, args.version)
    os.makedirs(path_to_save, exist_ok=True)

    # cuda
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
        if torch.cuda.get_device_capability(0)[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high')
    else:
        device = torch.device("cpu")

    # config
    d_cfg = build_dataset_config(args)
    m_cfg = build_model_config(args)

    # dataset and evaluator
    dataset, evaluator, num_classes = build_dataset(d_cfg, args, is_train=True)
    
    # DEBUG: Verify class counts are correct
    print(f"🔍 CLASS COUNT VERIFICATION:")
    print(f"   Objects:  {d_cfg.get('num_objects', 'default')} (expect 36)")
    print(f"   Actions:  {d_cfg.get('num_actions', 'default')} (expect 42 for smart_home, 157 for charades)")
    print(f"   Relations: {d_cfg.get('num_relations', 'default')} (expect 26)")
    print(f"   Total:    {num_classes}")

    # dataloader
    dataloader = build_dataloader(args, dataset, per_gpu_batch, CollateFunc(), is_train=True)

    # build model
    model, criterion = build_model(
        args=args,
        d_cfg=d_cfg,
        m_cfg=m_cfg,
        device=device,
        num_classes=num_classes, 
        trainable=True,
        resume=args.resume
        )
    model = model.to(device).train()

    # DDP
    model_without_ddp = model
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # SyncBatchNorm
    if args.sybn and args.distributed:
        print('use SyncBatchNorm ...')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Compute FLOPs and Params
    if distributed_utils.is_main_process():
        model_copy = deepcopy(model_without_ddp)
        FLOPs_and_Params(
            model=model_copy,
            img_size=d_cfg['test_size'],
            len_clip=args.len_clip,
            device=device)
        del model_copy

    # optimizer
    base_lr = args.base_lr
    accumulate = args.accumulate
    optimizer, start_epoch = build_optimizer(d_cfg, model_without_ddp, base_lr, args.resume)

    # lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_epoch, args.lr_decay_ratio)
    
    # Load or advance LR scheduler if resuming from checkpoint
    if start_epoch > 0 and args.resume is not None:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        restored = False
        if 'lr_scheduler' in checkpoint:
            saved_milestones = checkpoint['lr_scheduler'].get('milestones', None)
            # Only restore saved state if milestones match exactly (same training run)
            # If milestones differ, the user changed lr_epoch, so we must NOT load the
            # old state (it would import a stale decayed LR from the previous schedule).
            if saved_milestones is not None:
                # milestones in state_dict is a Counter; convert to sorted list for comparison
                import collections
                saved_list = sorted(saved_milestones.keys() if isinstance(saved_milestones, collections.Counter) else saved_milestones)
                new_list = sorted(args.lr_epoch)
                if saved_list == new_list:
                    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                    print(f'✅ Loaded LR scheduler state (milestones match), current LR: {lr_scheduler.get_last_lr()[0]:.6f}')
                    restored = True
                else:
                    print(f'⚠️  lr_epoch changed ({saved_list} → {new_list}): ignoring saved scheduler state.')
            else:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                print(f'✅ Loaded LR scheduler state, current LR: {lr_scheduler.get_last_lr()[0]:.6f}')
                restored = True
        if not restored:
            # Advance the new scheduler by start_epoch steps so its internal
            # last_epoch counter is correct, but since new milestones haven't
            # been reached yet the LR stays at base_lr.
            for _ in range(start_epoch):
                lr_scheduler.step()
            print(f'✅ LR scheduler advanced to epoch {start_epoch}, current LR: {lr_scheduler.get_last_lr()[0]:.6f}')

    # warmup scheduler
    warmup_scheduler = build_warmup(d_cfg, base_lr=base_lr)

    # training configuration
    max_epoch = args.max_epoch
    epoch_size = len(dataloader)
    
    # Warmup should be skipped if resuming past the warmup phase
    # wp_iter is measured in global iterations (iter + epoch * epoch_size)
    total_past_iters = start_epoch * epoch_size if start_epoch > 0 else 0
    if total_past_iters >= d_cfg['wp_iter']:
        warmup = False
        print(f'Warmup already completed in previous training (past {total_past_iters} iters, wp_iter={d_cfg["wp_iter"]})')
    else:
        warmup = True
    
    # Mixed Precision Training (AMP)
    if args.amp:
        print('Using Automatic Mixed Precision (AMP) training')
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = None

    profile_enabled = args.profile_iters > 0 and device.type == 'cuda'
    if args.profile_iters > 0 and device.type != 'cuda':
        print('Detailed train profiling requires CUDA. Skipping profiling.')
    profile_stats = {
        'data': 0.0,
        'h2d': 0.0,
        'forward': 0.0,
        'loss': 0.0,
        'reduce': 0.0,
        'backward': 0.0,
        'optim': 0.0,
    }



    # eval before training
    if args.eval_first and distributed_utils.is_main_process():
        # to check whether the evaluator can work
        eval_one_epoch(args, model_without_ddp, evaluator, 0, path_to_save, optimizer, lr_scheduler)

    # start to train
    training_start_time = time.time()  # Track total training time for ETA
    t0 = time.time()
    last_iter_end = t0
    epoch_times = []  # Track epoch durations for ETA
    
    for epoch in range(start_epoch, max_epoch):
        epoch_start = time.time()
        
        if args.distributed:
            dataloader.batch_sampler.sampler.set_epoch(epoch)            

        # train one epoch
        for iter_i, (frame_ids, video_clips, targets) in enumerate(dataloader):
            ni = iter_i + epoch * epoch_size
            do_profile = profile_enabled and iter_i < args.profile_iters
            if do_profile:
                profile_stats['data'] += time.time() - last_iter_end

            # warmup
            if ni < d_cfg['wp_iter'] and warmup:
                warmup_scheduler.warmup(ni, optimizer)

            elif ni == d_cfg['wp_iter'] and warmup:
                # warmup is over
                print('Warmup is over')
                warmup = False
                warmup_scheduler.set_lr(optimizer, lr=base_lr, base_lr=base_lr)

            if do_profile:
                torch.cuda.synchronize()
                stage_t0 = time.time()

            video_clips = video_clips.to(device, non_blocking=True)
            if video_clips.dtype == torch.uint8:
                video_clips = video_clips.float().div_(255.0)
            if do_profile:
                torch.cuda.synchronize()
                profile_stats['h2d'] += time.time() - stage_t0

            # inference and loss (with optional AMP)
            if do_profile:
                stage_t0 = time.time()
            if scaler is not None:
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(video_clips)
            else:
                outputs = model(video_clips)
            if do_profile:
                torch.cuda.synchronize()
                profile_stats['forward'] += time.time() - stage_t0

            if do_profile:
                stage_t0 = time.time()
            if scaler is not None:
                with torch.amp.autocast(device_type='cuda'):
                    loss_dict = criterion(outputs, targets)
                    losses = loss_dict['losses']
            else:
                loss_dict = criterion(outputs, targets)
                losses = loss_dict['losses']
            if do_profile:
                torch.cuda.synchronize()
                profile_stats['loss'] += time.time() - stage_t0

            # reduce            
            if do_profile:
                stage_t0 = time.time()
            loss_dict_reduced = distributed_utils.reduce_dict(loss_dict)
            if do_profile:
                torch.cuda.synchronize()
                profile_stats['reduce'] += time.time() - stage_t0

            # check loss
            if torch.isnan(losses):
                print('loss is NAN !!')
                continue

            # Backward (with optional AMP scaling)
            losses = losses / accumulate
            if do_profile:
                stage_t0 = time.time()
            if scaler is not None:
                scaler.scale(losses).backward()
            else:
                losses.backward()
            if do_profile:
                torch.cuda.synchronize()
                profile_stats['backward'] += time.time() - stage_t0

            # Cross-attention monitoring disabled (noisy, not useful for training progress)

            # Optimize (step after accumulate batches, or at end of epoch)
            is_accumulate_step = (iter_i + 1) % accumulate == 0
            is_last_iter = (iter_i + 1) == epoch_size
            if is_accumulate_step or is_last_iter:
                if do_profile:
                    stage_t0 = time.time()
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    # Gradient clipping - REQUIRED to prevent NaN with AMP
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Gradient clipping - prevents training explosion
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if do_profile:
                    torch.cuda.synchronize()
                    profile_stats['optim'] += time.time() - stage_t0

            if do_profile and (iter_i + 1) == args.profile_iters and distributed_utils.is_main_process():
                total_profile = sum(profile_stats.values())
                print('\n' + '=' * 70)
                print(f'PROFILE OVER FIRST {args.profile_iters} ITERATIONS')
                for key in ['data', 'h2d', 'forward', 'loss', 'reduce', 'backward', 'optim']:
                    avg_ms = profile_stats[key] * 1000 / args.profile_iters
                    pct = 100.0 * profile_stats[key] / max(total_profile, 1e-8)
                    print(f'  {key:>8}: {avg_ms:7.1f} ms/iter ({pct:5.1f}%)')
                print('=' * 70 + '\n')
                    
            # Display
            if distributed_utils.is_main_process() and iter_i % 10 == 0:
                t1 = time.time()
                cur_lr = [param_group['lr']  for param_group in optimizer.param_groups]
                print_log(cur_lr, epoch, max_epoch, iter_i, epoch_size, loss_dict_reduced, 
                          t1-t0, accumulate, training_start_time)
            
                t0 = time.time()

            if profile_enabled:
                last_iter_end = time.time()

        lr_scheduler.step()
        
        # Calculate epoch duration and ETA
        epoch_duration = time.time() - epoch_start
        epoch_times.append(epoch_duration)
        
        # Print epoch summary
        if distributed_utils.is_main_process():
            epoch_num = epoch + 1
            remaining_epochs = max_epoch - epoch_num
            
            # Calculate ETA based on average epoch time
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            eta_seconds = remaining_epochs * avg_epoch_time
            eta_hours = eta_seconds / 3600
            
            print(f"\n{'='*70}")
            print(f"📊 EPOCH {epoch_num}/{max_epoch} COMPLETED")
            print(f"   ⏱️  Epoch time: {epoch_duration/60:.1f} min | ETA: {eta_hours:.1f} hours")
            print(f"   📈 Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # GPU memory if available
            if args.cuda and torch.cuda.is_available():
                mem_used = torch.cuda.max_memory_allocated() / 1024**3
                mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"   🖥️  GPU Memory: {mem_used:.1f}/{mem_total:.1f} GB ({100*mem_used/mem_total:.0f}%)")
                torch.cuda.reset_peak_memory_stats()  # Reset for next epoch
            
            print(f"   🎯 Remaining: {remaining_epochs} epochs")
            print(f"{'='*70}\n")
        
        
        
        # Evaluation: every eval_epoch epochs OR at the last epoch
        should_eval = (epoch + 1) % args.eval_epoch == 0 or (epoch + 1) == max_epoch
        if should_eval:
            if distributed_utils.is_main_process():
                print(f"🔍 Running evaluation at epoch {epoch + 1}...")
            eval_one_epoch(args, model_without_ddp, evaluator, epoch, path_to_save, optimizer, lr_scheduler)


def eval_one_epoch(args, model_eval, evaluator, epoch, path_to_save, optimizer, lr_scheduler):
    # check evaluator
    if distributed_utils.is_main_process():
        # SAVE MODEL FIRST (before eval, so checkpoint exists even if eval crashes)
        print(f'💾 Saving checkpoint for epoch {epoch + 1}...')
        
        weight_name = '{}_epoch_{}.pth'.format(args.version, epoch+1)
        checkpoint_path = os.path.join(path_to_save, weight_name)
        checkpoint_data = {
            'model': model_eval.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args
        }
        torch.save(checkpoint_data, checkpoint_path)
        print(f'   ✅ Saved: {checkpoint_path}')
        
        # AUTO-BACKUP TO GOOGLE DRIVE (Colab only)
        # If /content/drive/MyDrive/yooowo exists, save a backup there
        gdrive_backup_path = '/content/drive/MyDrive/yooowo'
        if os.path.isdir(gdrive_backup_path):
            try:
                gdrive_checkpoint = os.path.join(gdrive_backup_path, weight_name)
                torch.save(checkpoint_data, gdrive_checkpoint)
                print(f'   ☁️  Google Drive backup: {gdrive_checkpoint}')
            except Exception as e:
                print(f'   ⚠️  Google Drive backup failed (continuing): {e}')
        
        # THEN EVALUATE
        if evaluator is None:
            print('No evaluator ... continuing training.')
        else:
            print('🔍 Running evaluation...')
            # set eval mode
            model_eval.trainable = False
            model_eval.eval()

            # evaluate
            evaluator.evaluate_frame_map(model_eval, epoch + 1)
                
            # set train mode.
            model_eval.trainable = True
            model_eval.train()                      

    if args.distributed:
        # wait for all processes to synchronize
        dist.barrier()


def print_log(lr, epoch, max_epoch, iter_i, epoch_size, loss_dict, batch_time, accumulate, training_start_time=None):
    """Print clean, useful training log with ETA."""
    import time as time_module
    
    # Basic info
    log = f'[Epoch {epoch+1}/{max_epoch}][Iter {iter_i}/{epoch_size}]'
    
    # Learning rate
    if len(lr) > 1 and lr[0] != lr[1]:
        log += f'[lr: {lr[0]:.6f}/{lr[1]:.6f}]'
    else:
        log += f'[lr: {lr[0]:.6f}]'
    
    # Key losses only (clean format)
    loss_conf = loss_dict.get('loss_conf', 0)
    loss_act = loss_dict.get('loss_act', 0)
    loss_obj = loss_dict.get('loss_obj', 0)
    loss_rel = loss_dict.get('loss_rel', 0)
    loss_box = loss_dict.get('loss_box', 0)
    total_loss = loss_dict.get('losses', 0) * accumulate
    
    # Convert tensors to floats
    if hasattr(loss_conf, 'item'): loss_conf = loss_conf.item()
    if hasattr(loss_act, 'item'): loss_act = loss_act.item()
    if hasattr(loss_obj, 'item'): loss_obj = loss_obj.item()
    if hasattr(loss_rel, 'item'): loss_rel = loss_rel.item()
    if hasattr(loss_box, 'item'): loss_box = loss_box.item()
    if hasattr(total_loss, 'item'): total_loss = total_loss.item()
    
    log += f'[conf:{loss_conf:.2f}][act:{loss_act:.2f}][obj:{loss_obj:.2f}][rel:{loss_rel:.2f}][box:{loss_box:.2f}][total:{total_loss:.2f}]'
    
    # Time per batch
    log += f'[{batch_time:.1f}s]'
    
    # ETA calculation
    if training_start_time is not None:
        elapsed = time_module.time() - training_start_time
        total_iters = max_epoch * epoch_size
        current_iter = epoch * epoch_size + iter_i
        
        if current_iter > 0:
            time_per_iter = elapsed / current_iter
            remaining_iters = total_iters - current_iter
            eta_seconds = time_per_iter * remaining_iters
            eta_hours = eta_seconds / 3600
            
            if eta_hours >= 1:
                log += f'[ETA: {eta_hours:.1f}h]'
            else:
                log += f'[ETA: {eta_seconds/60:.0f}m]'
    
    print(log, flush=True)


if __name__ == '__main__':
    train()
