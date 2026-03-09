import random
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


# Augmentation for Training
class Augmentation(object):
    def __init__(self, img_size=224, jitter=0.2, hue=0.1, saturation=1.5, exposure=1.5):
        self.img_size = img_size
        self.jitter = jitter
        self.hue = hue
        self.saturation = saturation
        self.exposure = exposure


    def rand_scale(self, s):
        scale = random.uniform(1, s)

        if random.randint(0, 1): 
            return scale

        return 1./scale


    def _distort_cv2(self, video_clip_np):
        """Color distortion using cv2 HSV. Takes/returns list of RGB numpy arrays."""
        dhue = random.uniform(-self.hue, self.hue)
        dsat = self.rand_scale(self.saturation)
        dexp = self.rand_scale(self.exposure)

        hue_shift = int(dhue * 180)
        h_lut = np.array([(i + hue_shift) % 180 for i in range(180)], dtype=np.uint8)
        h_lut = np.pad(h_lut, (0, 76))
        s_lut = np.clip(np.arange(256, dtype=np.float32) * dsat, 0, 255).astype(np.uint8)
        v_lut = np.clip(np.arange(256, dtype=np.float32) * dexp, 0, 255).astype(np.uint8)

        result = []
        for img in video_clip_np:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            hsv[:, :, 0] = cv2.LUT(hsv[:, :, 0], h_lut)
            hsv[:, :, 1] = cv2.LUT(hsv[:, :, 1], s_lut)
            hsv[:, :, 2] = cv2.LUT(hsv[:, :, 2], v_lut)
            result.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))

        return result

    def _distort_pil(self, video_clip):
        """Color distortion using PIL HSV. Fallback when cv2 unavailable."""
        dhue = random.uniform(-self.hue, self.hue)
        dsat = self.rand_scale(self.saturation)
        dexp = self.rand_scale(self.exposure)

        sat_lut = bytes([max(0, min(255, int(i * dsat))) for i in range(256)])
        exp_lut = bytes([max(0, min(255, int(i * dexp))) for i in range(256)])
        hue_shift = int(dhue * 255)
        hue_lut = bytes([(i + hue_shift) % 256 for i in range(256)])

        video_clip_ = []
        for image in video_clip:
            image = image.convert('HSV')
            cs = list(image.split())
            cs[0] = cs[0].point(hue_lut)
            cs[1] = cs[1].point(sat_lut)
            cs[2] = cs[2].point(exp_lut)
            image = Image.merge(image.mode, tuple(cs))
            image = image.convert('RGB')
            video_clip_.append(image)

        return video_clip_

    def _crop_numpy(self, frames, pleft, ptop, crop_w, crop_h):
        """Numpy crop matching PIL Image.crop zero-fill for out-of-bounds regions."""
        if crop_h <= 0 or crop_w <= 0:
            return [np.zeros((max(1, crop_h), max(1, crop_w), 3), dtype=frames[0].dtype)
                    for _ in frames]

        h, w = frames[0].shape[:2]

        if pleft >= 0 and ptop >= 0 and pleft + crop_w <= w and ptop + crop_h <= h:
            return [f[ptop:ptop+crop_h, pleft:pleft+crop_w].copy() for f in frames]

        src_t, src_l = max(0, ptop), max(0, pleft)
        src_b, src_r = min(h, ptop + crop_h), min(w, pleft + crop_w)
        dst_t, dst_l = src_t - ptop, src_l - pleft
        ch, cw = src_b - src_t, src_r - src_l

        result = []
        for f in frames:
            out = np.zeros((crop_h, crop_w, 3), dtype=f.dtype)
            if ch > 0 and cw > 0:
                out[dst_t:dst_t+ch, dst_l:dst_l+cw] = f[src_t:src_b, src_l:src_r]
            result.append(out)
        return result


    def random_crop(self, video_clip, width, height):
        dw = int(width * self.jitter)
        dh = int(height * self.jitter)

        pleft  = random.randint(-dw, dw)
        pright = random.randint(-dw, dw)
        ptop   = random.randint(-dh, dh)
        pbot   = random.randint(-dh, dh)

        swidth  = width - pleft - pright
        sheight = height - ptop - pbot

        sx = float(swidth)  / width
        sy = float(sheight) / height

        dx = (float(pleft) / width) / sx
        dy = (float(ptop) / height) / sy

        # PIL crop uses exclusive right/bottom, original code passes swidth-1/sheight-1
        crop_w = swidth - 1
        crop_h = sheight - 1

        if isinstance(video_clip[0], np.ndarray):
            cropped = self._crop_numpy(video_clip, pleft, ptop, crop_w, crop_h)
        else:
            cropped = [img.crop((pleft, ptop, pleft + crop_w, ptop + crop_h))
                       for img in video_clip]

        return cropped, dx, dy, sx, sy


    def apply_bbox(self, target, ow, oh, dx, dy, sx, sy):
        sx, sy = 1./sx, 1./sy
        target[..., 0] = np.minimum(0.999, np.maximum(0, target[..., 0] / ow * sx - dx)) 
        target[..., 1] = np.minimum(0.999, np.maximum(0, target[..., 1] / oh * sy - dy)) 
        target[..., 2] = np.minimum(0.999, np.maximum(0, target[..., 2] / ow * sx - dx)) 
        target[..., 3] = np.minimum(0.999, np.maximum(0, target[..., 3] / oh * sy - dy)) 

        refine_target = []
        for i in range(target.shape[0]):
            tgt = target[i]
            bw = (tgt[2] - tgt[0]) * ow
            bh = (tgt[3] - tgt[1]) * oh

            if bw < 1. or bh < 1.:
                continue
            
            refine_target.append(tgt)

        refine_target = np.array(refine_target).reshape(-1, target.shape[-1])

        return refine_target


    def __call__(self, video_clip, target):
        if isinstance(video_clip[0], np.ndarray):
            oh, ow = video_clip[0].shape[:2]
        else:
            oh = video_clip[0].height
            ow = video_clip[0].width

        video_clip, dx, dy, sx, sy = self.random_crop(video_clip, ow, oh)

        flip = random.randint(0, 1)

        if isinstance(video_clip[0], np.ndarray):
            video_clip = [cv2.resize(f, (self.img_size, self.img_size),
                          interpolation=cv2.INTER_LINEAR) for f in video_clip]
            if flip:
                video_clip = [cv2.flip(f, 1) for f in video_clip]
            video_clip = self._distort_cv2(video_clip)
            clip = np.stack(video_clip)                                     # [T, H, W, C]
            video_clip = torch.from_numpy(
                np.ascontiguousarray(clip.transpose(3, 0, 1, 2)))          # [C, T, H, W]
        else:
            video_clip = [img.resize([self.img_size, self.img_size], Image.BILINEAR)
                         for img in video_clip]
            if flip:
                video_clip = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in video_clip]
            video_clip = self._distort_pil(video_clip)
            video_clip = [F.pil_to_tensor(image) for image in video_clip]

        if target is not None:
            target = self.apply_bbox(target, ow, oh, dx, dy, sx, sy)
            if flip:
                target[..., [0, 2]] = 1.0 - target[..., [2, 0]]
        else:
            target = np.array([])

        target = torch.as_tensor(target).float()

        return video_clip, target


# Transform for Testing
class BaseTransform(object):
    def __init__(self, img_size=224):
        self.img_size = img_size


    def __call__(self, video_clip, target=None, normalize=True):
        if isinstance(video_clip[0], np.ndarray):
            oh, ow = video_clip[0].shape[:2]
            video_clip = [cv2.resize(f, (self.img_size, self.img_size),
                         interpolation=cv2.INTER_LINEAR) for f in video_clip]
            clip = np.stack(video_clip)                                     # [T, H, W, C]
            video_clip = torch.from_numpy(
                np.ascontiguousarray(clip.transpose(3, 0, 1, 2))
            ).float().div_(255.0)                                           # [C, T, H, W]
        else:
            oh = video_clip[0].height
            ow = video_clip[0].width
            video_clip = [img.resize([self.img_size, self.img_size]) for img in video_clip]
            video_clip = [F.to_tensor(image) for image in video_clip]

        if target is not None:
            if normalize:
                target[..., [0, 2]] /= ow
                target[..., [1, 3]] /= oh
        else:
            target = np.array([])

        target = torch.as_tensor(target).float()

        return video_clip, target
