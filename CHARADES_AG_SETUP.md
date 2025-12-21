# Charades + Action Genome Setup for YOWO

## What We're Doing (Option 1 - Correct Method)

We're downloading the **original 480p Charades videos** and extracting frames at **original resolution** using Action Genome's official method. This ensures perfect alignment between frames and annotations.

## Why This is Necessary

The pre-extracted `Charades_v1_rgb.tar` frames were scaled down to 320px max dimension, but Action Genome annotations were made on original video resolution (~480p). This caused coordinate mismatches.

**Solution**: Extract frames directly from videos using Action Genome's official `dump_frames.py` script.

## Download & Extraction Steps

### 1. Download Charades 480p Videos (13 GB)
```bash
wget -c -O Charades_v1_480.zip "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip"
```
- **Status**: In progress (~3.4GB / 13GB as of last check)
- **ETA**: ~8 minutes remaining
- **Source**: Official Allen AI dataset repository

### 2. Download Action Genome Annotations
```bash
gdown --folder "https://drive.google.com/drive/folders/1LGGPK_QgGbh9gH9SDFv_9LIhBliZbZys"
```
- **Status**: ✓ Complete
- **Files**:
  - `person_bbox.pkl` (156 MB) - Human bounding boxes
  - `object_bbox_and_relationship.pkl` (136 MB) - Objects + relationships  
  - `frame_list.txt` (5.8 MB) - List of annotated frames
  - `object_classes.txt` - 38 object categories
  - `relationship_classes.txt` - 26 relationship types

### 3. Extract Frames Using Official Method
```bash
./extract_ag_frames.sh
```
This script:
1. Unzips `Charades_v1_480.zip` 
2. Places videos in `data/ActionGenome/videos/`
3. Runs `ActionGenome/tools/dump_frames.py` to extract frames
4. Saves frames to `data/ActionGenome/frames/` (~74 GB)

**Key Details**:
- Uses `ffmpeg` to extract frames at original FPS
- Frame indices match Action Genome annotations
- Each frame is saved as `VIDEO_ID/NNNNNN.png` (e.g., `001YG.mp4/000089.png`)

### 4. Verify Frame Extraction
```bash
python verify_frame_extraction.py
```
Checks:
- Frames exist for sample annotations
- Image dimensions match `bbox_size` in annotations (~480p)
- Bounding boxes will align correctly

## Dataset Structure After Setup

```
/home/michel/yowo/data/ActionGenome/
├── annotations/
│   ├── person_bbox.pkl              # Human bboxes + keyframes
│   ├── object_bbox_and_relationship.pkl  # Objects + relations
│   ├── frame_list.txt               # All annotated frames
│   ├── object_classes.txt           # 38 objects
│   ├── relationship_classes.txt     # 26 relationships
│   ├── Charades_v1_train.csv        # Training action labels
│   ├── Charades_v1_test.csv         # Test action labels
│   └── Charades/
│       ├── Charades_v1_classes.txt  # 157 actions
│       └── Charades_v1_objectclasses.txt  # 38 objects
├── videos/
│   ├── 001YG.mp4
│   ├── 002XP.mp4
│   └── ... (9,848 videos total)
└── frames/
    ├── 001YG.mp4/
    │   ├── 000001.png
    │   ├── 000002.png
    │   └── ...
    ├── 002XP.mp4/
    └── ...
```

## Label Space (222 classes total)

Our unified multi-hot label vector:

| Indices | Content | Count | Source |
|---------|---------|-------|--------|
| 0-37 | Object classes | 38 | Charades/Action Genome |
| 38 | Human flag | 1 | Indicator |
| 39-195 | Action classes | 157 | Charades |
| 196-221 | Relationships | 26 | Action Genome |

**Example Multi-Hot Label for a Human**:
- Index 38 = 1 (is human)
- Index 39+47 = 1 (action c047: "holding dish")
- Index 196+8 = 1 (relationship: "holding")
- Index 196+20 = 1 (relationship: "touching")

## Coordinate System

With 480p frames extracted using the official method:
- **Frames**: ~480p (varies by video, e.g., 480x640 or similar)
- **Annotations**: Made on original resolution, stored in `bbox_size` field
- **Scaling**: `charades_ag.py` handles any minor scaling automatically

```python
# In dataset loader
bbox_h, bbox_w = person_info['bbox_size']  # Annotation resolution
ow, oh = image.size  # Actual frame resolution

scale_x = ow / bbox_w
scale_y = oh / bbox_h

# Apply scaling
x_scaled = x * scale_x
y_scaled = y * scale_y
```

For properly extracted 480p frames, `scale_x` and `scale_y` should be ~1.0 or very close.

## Model Configuration

### YOLOv11m as 2D Backbone
- **File**: `models/backbone/backbone_2d/cnn_2d/yolo_11.py`
- **Config**: `config/yowo_v2_config.py` → `yowo_v2_medium_yolo11m`
- **Features**: P3, P4, P5 at strides 8, 16, 32

### YOWO Model
- **Num Classes**: 222 (objects + human + actions + relationships)
- **Multi-Hot**: Yes (each bbox can have multiple active labels)
- **3D Backbone**: ShuffleNetV2 (for temporal features)
- **2D Backbone**: YOLOv11m (for spatial features)

## Training Command (After Setup)

```bash
conda activate yowov2
cd /home/michel/yowo

python train.py \
    --cuda \
    -d charades_ag \
    --root /home/michel/yowo/data \
    -m yowo_v2_medium_yolo11m \
    -bs 8 \
    -lr 1e-4 \
    --max_epoch 10 \
    --lr_epoch 2 5 8 \
    --eval_epoch 2 \
    --num_workers 4
```

## Timeline

1. ✓ Cloned Action Genome repository
2. ✓ Downloaded Action Genome annotations (frame_list.txt, person_bbox.pkl, object_bbox_and_relationship.pkl)
3. ⏳ Downloading Charades 480p videos (~8 min remaining)
4. ⏳ Extract frames using official method (~30-60 min)
5. ⏳ Verify frame extraction
6. ⏳ Start training

## References

- **Action Genome Paper**: [CVPR 2020](http://openaccess.thecvf.com/content_CVPR_2020/papers/Ji_Action_Genome_Actions_As_Compositions_of_Spatio-Temporal_Scene_Graphs_CVPR_2020_paper.pdf)
- **Action Genome Repo**: https://github.com/JingweiJ/ActionGenome
- **Charades Dataset**: https://prior.allenai.org/projects/charades
- **YOWO Paper**: https://arxiv.org/abs/2107.05848

---

**Status**: Waiting for video download to complete, then will extract frames and begin training.




