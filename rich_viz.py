#!/usr/bin/env python3
"""
Rich Visualization for Charades + Action Genome
Diving deep into the dataset to show everything.
"""
import os
import pickle
import cv2
import numpy as np
import csv
import subprocess
from pathlib import Path

def get_fps(video_path):
    try:
        cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=avg_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)]
        res = subprocess.check_output(cmd).decode().strip()
        if '/' in res:
            n, d = map(float, res.split('/'))
            return n / d if d != 0 else 24.0
        return float(res)
    except:
        return 24.0

def rich_viz(video_id='001YG', frame_idx='000089'):
    video_id_full = f"{video_id}.mp4"
    frame_key = f"{video_id_full}/{frame_idx}.png"
    
    # Paths
    root = Path('/home/michel/yowo/data/ActionGenome')
    frames_dir = root / 'frames'
    videos_dir = root / 'videos'
    ann_dir = root / 'annotations'
    output_dir = Path('/home/michel/yowo/vis_rich')
    output_dir.mkdir(exist_ok=True)
    
    # 1. Load Data
    with open(ann_dir / 'person_bbox.pkl', 'rb') as f:
        person_bboxes = pickle.load(f)
    with open(ann_dir / 'object_bbox_and_relationship.pkl', 'rb') as f:
        object_data = pickle.load(f)
    
    # Load Charades Actions
    active_actions = []
    fps = get_fps(videos_dir / video_id_full)
    time_sec = int(frame_idx) / fps

    with open(ann_dir / 'Charades_v1_train.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['id'] == video_id:
                actions_str = row['actions']
                if actions_str:
                    for a in actions_str.split(';'):
                        parts = a.split(' ')
                        if len(parts) == 3:
                            cls, start, end = parts[0], float(parts[1]), float(parts[2])
                            if start <= time_sec <= end:
                                active_actions.append(cls)
                break

    # 2. Load Image
    img_path = frames_dir / video_id_full / f"{frame_idx}.png"
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Error: {img_path} not found")
        return
    
    h, w = img.shape[:2]
    vis = img.copy()
    
    # 3. Draw Everything
    person_info = person_bboxes.get(frame_key)
    p_center = None
    if person_info and len(person_info['bbox']) > 0:
        # PERSON: xyxy
        x1, y1, x2, y2 = person_info['bbox'][0]
        # Scale
        aw, ah = person_info['bbox_size']
        sx, sy = w / aw, h / ah
        ix1, iy1, ix2, iy2 = int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)
        
        cv2.rectangle(vis, (ix1, iy1), (ix2, iy2), (0, 255, 0), 2)
        p_center = ((ix1+ix2)//2, (iy1+iy2)//2)
        
        # Action labels
        act_text = "PERSON: " + ", ".join(active_actions)
        cv2.putText(vis, act_text, (ix1, iy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # OBJECTS: xywh
    frame_objects = object_data.get(frame_key, [])
    for obj in frame_objects:
        if not obj.get('visible'): continue
        
        ox, oy, ow_val, oh_val = obj['bbox']
        # Scale
        iox, ioy, iow, ioh = int(ox*sx), int(oy*sy), int(ow_val*sx), int(oh_val*sy)
        # Convert xywh -> xyxy
        ox1, oy1, ox2, oy2 = iox, ioy, iox + iow, ioy + ioh
        
        color = (255, 255, 0)
        cv2.rectangle(vis, (ox1, oy1), (ox2, oy2), color, 2)
        cv2.putText(vis, obj['class'], (ox1, oy1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Relationships
        rels = []
        for r_type in ['attention_relationship', 'spatial_relationship', 'contacting_relationship']:
            if obj.get(r_type):
                rels.extend([r for r in obj[r_type] if r not in ['not_contacting', 'not_looking_at', 'unsure', 'none']])
        
        if rels and p_center:
            o_center = (ox1 + iow//2, oy1 + ioh//2)
            cv2.line(vis, p_center, o_center, (0, 0, 255), 1)
            rel_str = "/".join(rels)
            m_x, m_y = (p_center[0] + o_center[0]) // 2, (p_center[1] + o_center[1]) // 2
            cv2.putText(vis, rel_str, (m_x, m_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # Debug info
    cv2.putText(vis, f"Video: {video_id} Frame: {frame_idx}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, f"Time: {time_sec:.2f}s FPS: {fps:.2f}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Save
    out_path = output_dir / f"rich_{video_id}_{frame_idx}.png"
    cv2.imwrite(str(out_path), vis)
    print(f"Saved rich visualization to {out_path}")

if __name__ == "__main__":
    rich_viz('001YG', '000089')
    rich_viz('001YG', '000264')
    rich_viz('015XE', '000096')

