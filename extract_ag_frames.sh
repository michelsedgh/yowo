#!/bin/bash

# Script to extract Action Genome frames from Charades videos
# Uses the official Action Genome method

set -e

echo "======================================"
echo "Action Genome Frame Extraction"
echo "======================================"

# Check if videos are downloaded
if [ ! -f "/home/michel/yowo/Charades_v1_480.zip" ]; then
    echo "ERROR: Charades_v1_480.zip not found!"
    echo "Please download it first."
    exit 1
fi

# Unzip videos if not already done
if [ ! -d "/home/michel/yowo/data/ActionGenome/videos" ] || [ -z "$(ls -A /home/michel/yowo/data/ActionGenome/videos 2>/dev/null)" ]; then
    echo "Extracting videos..."
    cd /home/michel/yowo
    unzip -q Charades_v1_480.zip -d /home/michel/yowo/data/ActionGenome/videos_temp
    mv /home/michel/yowo/data/ActionGenome/videos_temp/Charades_v1_480/* /home/michel/yowo/data/ActionGenome/videos/
    rmdir /home/michel/yowo/data/ActionGenome/videos_temp/Charades_v1_480
    rmdir /home/michel/yowo/data/ActionGenome/videos_temp
    echo "Videos extracted to /home/michel/yowo/data/ActionGenome/videos/"
else
    echo "Videos already extracted."
fi

# Check for frame_list.txt
if [ ! -f "/home/michel/yowo/data/ActionGenome/annotations/frame_list.txt" ]; then
    echo "ERROR: frame_list.txt not found in annotations!"
    exit 1
fi

# Run the official Action Genome frame extraction script
echo "Extracting frames using Action Genome's official method..."
cd /home/michel/yowo
conda activate yowov2

python ActionGenome/tools/dump_frames.py \
    --video_dir /home/michel/yowo/data/ActionGenome/videos \
    --frame_dir /home/michel/yowo/data/ActionGenome/frames \
    --annotation_dir /home/michel/yowo/data/ActionGenome/annotations \
    --all_frames

echo "======================================"
echo "Frame extraction completed!"
echo "Frames saved to: /home/michel/yowo/data/ActionGenome/frames/"
echo "======================================"




