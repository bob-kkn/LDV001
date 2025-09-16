#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Panorama PII blurring (faces & plates) — FULL INTEGRATED (Speed-Optimized)
- Image / Video / Folder (recursive), original extension preserved
- YOLO or OpenCV Cascade detectors
- Horizontal tiling with phases, polar band reinforcement
- Soft-NMS, optional upscale detection, optional enhancement
- Multiprocess (--jobs) for folder mode
- YOLO batch tile inference, FAST preset, OpenCV/BLAS threads tuned
"""

import argparse, math, os, sys, time
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed

# ===== Threading / OpenCV tuning (minimize contention with multiprocessing) =====
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
try:
    cv2.setNumThreads(0)
except Exception:
    pass
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

# =============================
# Extensions
# =============================
IMG_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}

def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS

def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTS

def ensure_parent_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

# =============================
# Panorama tiling / NMS utilities
# =============================
def slice_horizontal_tiles(img, tiles=8, overlap_ratio=0.1, start_offset_px=0):
    H, W = img.shape[:2]
    tile_w = int(math.ceil(W / tiles))
    overlap = int(tile_w * overlap_ratio)
    starts = [ (i*tile_w - overlap + start_offset_px) % W for i in range(tiles) ]
    widths = [tile_w + 2*overlap] * tiles
    tiles_img, meta = [], []
    for sx, tw in zip(starts, widths):
        if sx + tw <= W:
            crop = img[:, sx:sx+tw]
        else:
            right = img[:, sx:W]
            left  = img[:, : (sx+tw) % W]
            crop = np.hstack([right, left])
        tiles_img.append(crop)
        meta.append((sx, tw))
    return tiles_img, meta, (W, H)

def boxes_tile_to_global(boxes_xyxy, sx, W):
    out = []
    for (x1,y1,x2,y2,score,clsid) in boxes_xyxy:
        out.append([ (sx + x1) % W, y1, (sx + x2) % W, y2, score, clsid ])
    return out

def wrap_split_boxes(boxes, W):
    out = []
    for b in boxes:
        x1,y1,x2,y2,sc,cl = b
        if x1 > x2:  # seam wrap
            out.append([x1, y1, x2 + W, y2, sc, cl])
            out.append([x1 - W, y1, x2, y2, sc, cl])
        else:
            out.append(b)
    return out

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a[:4]; bx1, by1, bx2, by2 = b[:4]
    ax1, ax2 = min(ax1, ax2), max(ax1, ax2)
    bx1, bx2 = min(bx1, bx2), max(bx1, bx2)
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)

def nms_classwise(boxes, iou_thr=0.5):
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    keep, used = [], [False]*len(boxes)
    for i, bi in enumerate(boxes):
        if used[i]: continue
        keep.append(bi); used[i] = True
        for j in range(i+1, len(boxes)):
            if used[j]: continue
            if bi[5] != boxes[j][5]:
                continue
            if iou_xyxy(bi, boxes[j]) >= iou_thr:
                used[j] = True
    return keep

def soft_nms_classwise(boxes, iou_thr=0.5, sigma=0.5, score_thr=0.05):
    import math
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    out = []
    while boxes:
        best = boxes.pop(0)
        out.append(best)
        keep = []
        for b in boxes:
            if b[5] != best[5]:
                keep.append(b); continue
            iou = iou_xyxy(best, b)
            if iou > iou_thr:
                b = b.copy()
                b[4] = b[4] * math.exp(-(iou**2) / sigma)
            if b[4] >= score_thr:
                keep.append(b)
        boxes = sorted(keep, key=lambda x: x[4], reverse=True)
    return out

# =============================
# Detectors
# =============================
class CascadeDet:
    def __init__(self, face_neighbors=5, plate_neighbors=4):
        face_xml  = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        plate_xml = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
        self.face_cascade  = cv2.CascadeClassifier(face_xml)
        self.plate_cascade = cv2.CascadeClassifier(plate_xml)
        self.face_neighbors = face_neighbors
        self.plate_neighbors = plate_neighbors

    def detect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=self.face_neighbors,
            flags=cv2.CASCADE_SCALE_IMAGE, minSize=(24,24)
        )
        plates = self.plate_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=self.plate_neighbors,
            flags=cv2.CASCADE_SCALE_IMAGE, minSize=(24,12)
        )
        out = []
        for (x, y, w, h) in faces:
            out.append([x, y, x+w, y+h, 0.90, 0])   # 0=face
        for (x, y, w, h) in plates:
            out.append([x, y, x+w, y+h, 0.85, 1])   # 1=plate
        return out

class YoloDet:
    def __init__(self, face_weight=None, plate_weight=None, conf=0.25, iou=0.5, imgsz=1280):
        from ultralytics import YOLO
        self.models = {}
        if face_weight:
            self.models['face'] = YOLO(face_weight)
        if plate_weight:
            self.models['plate'] = YOLO(plate_weight)
        self.conf, self.iou, self.imgsz = conf, iou, imgsz

    def detect(self, img):
        out = []
        if 'face' in self.models:
            res = self.models['face'](img, conf=self.conf, iou=self.iou, imgsz=self.imgsz, verbose=False)[0]
            for xyxy, sc in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                x1,y1,x2,y2 = xyxy.tolist()
                out.append([x1,y1,x2,y2,float(sc),0])
        if 'plate' in self.models:
            res = self.models['plate'](img, conf=self.conf, iou=self.iou, imgsz=self.imgsz, verbose=False)[0]
            for xyxy, sc in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                x1,y1,x2,y2 = xyxy.tolist()
                out.append([x1,y1,x2,y2,float(sc),1])
        return out

    # ===== Batch detection for speed =====
    def detect_batch(self, imgs: List[np.ndarray]):
        outs_face = [[] for _ in imgs]
        outs_plate = [[] for _ in imgs]
        if 'face' in self.models:
            res_list = self.models['face'](imgs, conf=self.conf, iou=self.iou,
                                           imgsz=self.imgsz, verbose=False)
            for i, r in enumerate(res_list):
                if len(r.boxes):
                    for xyxy, sc in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                        x1,y1,x2,y2 = xyxy.tolist()
                        outs_face[i].append([x1,y1,x2,y2,float(sc),0])
        if 'plate' in self.models:
            res_list = self.models['plate'](imgs, conf=self.conf, iou=self.iou,
                                            imgsz=self.imgsz, verbose=False)
            for i, r in enumerate(res_list):
                if len(r.boxes):
                    for xyxy, sc in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                        x1,y1,x2,y2 = xyxy.tolist()
                        outs_plate[i].append([x1,y1,x2,y2,float(sc),1])
        return [f + p for f, p in zip(outs_face, outs_plate)]

# =============================
# Preprocess / upscale wrapper
# =============================
def pre_enhance(img):
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    y = clahe.apply(y)
    yuv = cv2.merge([y,u,v])
    enh = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    blur = cv2.GaussianBlur(enh, (0,0), sigmaX=1.0)
    sharp = cv2.addWeighted(enh, 1.5, blur, -0.5, 0)
    return sharp

def detect_with_options(detector, img, enhance=False, upscale=1.0):
    src = img
    if enhance:
        src = pre_enhance(src)
    if upscale and upscale > 1.0:
        h, w = src.shape[:2]
        up = cv2.resize(src, (int(w*upscale), int(h*upscale)), interpolation=cv2.INTER_CUBIC)
        dets = detector.detect(up)
        out = []
        for x1,y1,x2,y2,sc,cl in dets:
            out.append([x1/upscale, y1/upscale, x2/upscale, y2/upscale, sc, cl])
        return out
    else:
        return detector.detect(src)

# =============================
# Blur / mosaic
# =============================
def apply_blur(img, box, method="gaussian", k=31, pixel=16):
    H, W = img.shape[:2]
    x1,y1,x2,y2 = box
    x1 = int(max(0, min(x1, W-1))); x2 = int(max(0, min(x2, W-1)))
    y1 = int(max(0, min(y1, H-1))); y2 = int(max(0, min(y2, H-1)))
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    if method == "pixelate":
        h, w = roi.shape[:2]
        pw = max(2, min(w, pixel)); ph = max(2, min(h, pixel))
        roi_small = cv2.resize(roi, (max(1, w//pw), max(1, h//ph)), interpolation=cv2.INTER_LINEAR)
        roi_pix = cv2.resize(roi_small, (w, h), interpolation=cv2.INTER_NEAREST)
        img[y1:y2, x1:x2] = roi_pix
    else:
        k = max(5, k | 1)
        blurred = cv2.GaussianBlur(roi, (k,k), 0)
        img[y1:y2, x1:x2] = blurred

def blur_with_wrap(img, box, method="gaussian", **kwargs):
    H, W = img.shape[:2]
    x1,y1,x2,y2 = box
    if x1 <= x2:
        apply_blur(img, (x1,y1,x2,y2), method=method, **kwargs)
    else:
        apply_blur(img, (x1,y1,W-1,y2), method=method, **kwargs)
        apply_blur(img, (0,y1,x2,y2), method=method, **kwargs)

# =============================
# Polar bands
# =============================
def detect_polar_bands(img, detector, bands=0, band_ratio=0.20, enhance=False, upscale=1.0):
    H, W = img.shape[:2]
    out = []
    band_h = int(H * band_ratio)
    bands = max(0, bands)
    if bands == 0 or band_h <= 8:
        return out
    # top bands
    for i in range(bands):
        y1 = max(0, i * band_h // max(1, bands))
        y2 = min(H, y1 + band_h)
        crop = img[y1:y2, :]
        dets = detect_with_options(detector, crop, enhance=enhance, upscale=upscale)
        for x1,y1c,x2,y2c,sc,cl in dets:
            out.append([x1, y1 + y1c, x2, y1 + y2c, sc, cl])
    # bottom bands
    for i in range(bands):
        y2 = H - max(0, i * band_h // max(1, bands))
        y1 = max(0, y2 - band_h)
        crop = img[y1:y2, :]
        dets = detect_with_options(detector, crop, enhance=enhance, upscale=upscale)
        for x1,y1c,x2,y2c,sc,cl in dets:
            out.append([x1, y1 + y1c, x2, y1 + y2c, sc, cl])
    return out

# =============================
# Main pipeline (one image)
# =============================
def process(pano_bgr, detector, tiles=8, overlap_ratio=0.10, nms_iou=0.55,
            method="gaussian", ksize=31, pixel=16, phases=1,
            polar_bands=0, polar_ratio=0.20, soft_nms=False,
            enhance=False, upscale=1.0):
    H, W = pano_bgr.shape[:2]
    all_boxes = []

    phases = max(1, int(phases))
    for ph in range(phases):
        start_offset_px = 0 if ph == 0 else int((W/tiles) * 0.5)
        tiles_img, meta, _ = slice_horizontal_tiles(
            pano_bgr, tiles=tiles, overlap_ratio=overlap_ratio, start_offset_px=start_offset_px
        )
        # ===== Batch path (fast) =====
        if isinstance(detector, YoloDet) and (not enhance) and (upscale <= 1.0):
            det_lists = detector.detect_batch(tiles_img)
            for (dets, (sx, _)) in zip(det_lists, meta):
                g = boxes_tile_to_global(dets, sx, W)
                all_boxes.extend(g)
        else:
            # Per-tile path (needed if enhance/upscale is on, or Cascade)
            for (crop, (sx, _)) in zip(tiles_img, meta):
                dets = detect_with_options(detector, crop, enhance=enhance, upscale=upscale)
                g = boxes_tile_to_global(dets, sx, W)
                all_boxes.extend(g)

    if polar_bands > 0 and polar_ratio > 0:
        pol = detect_polar_bands(pano_bgr, detector, bands=polar_bands,
                                 band_ratio=polar_ratio, enhance=enhance, upscale=upscale)
        all_boxes.extend(pol)

    boxes_expanded = wrap_split_boxes(all_boxes, W)
    if soft_nms:
        boxes_kept = soft_nms_classwise(boxes_expanded, iou_thr=nms_iou, sigma=0.5, score_thr=0.05)
    else:
        boxes_kept = nms_classwise(boxes_expanded, iou_thr=nms_iou)

    out = pano_bgr.copy()
    for x1,y1,x2,y2,sc,cl in boxes_kept:
        x1, x2 = int(x1) % W, int(x2) % W
        blur_with_wrap(out, (x1,y1,x2,y2),
                       method=("pixelate" if method=="pixelate" else "gaussian"),
                       k=ksize, pixel=pixel)
    return out, boxes_kept

# =============================
# Single file (image/video)
# =============================
def process_image_file(in_path: Path, out_path: Path, det, args) -> bool:
    img = cv2.imread(str(in_path))
    if img is None:
        print(f"[skip] cannot read image: {in_path}")
        return False
    out_img, boxes = process(
        img, det,
        tiles=args.tiles, overlap_ratio=args.overlap, nms_iou=args.nms,
        method=("pixelate" if args.pixelate else "gaussian"),
        ksize=args.ksize, pixel=args.pixel,
        phases=args.phases, polar_bands=args.polar_bands, polar_ratio=args.polar_ratio,
        soft_nms=args.soft_nms, enhance=args.enhance, upscale=args.upscale
    )
    ensure_parent_dir(out_path)
    ok = cv2.imwrite(str(out_path), out_img)
    if not ok:
        print(f"[err] failed to write: {out_path}")
        return False
    print(f"[ok] {in_path} -> {out_path} (detections={len(boxes)})")
    return True

def process_video_file(in_path: Path, out_path: Path, det, args) -> bool:
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        print(f"[skip] cannot open video: {in_path}")
        return False

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if fps_in <= 1e-2: fps_in = 30.0
    fps_out = args.video_fps if args.video_fps > 0 else fps_in

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*args.video_codec)  # e.g., 'mp4v','XVID','avc1'
    ensure_parent_dir(out_path)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps_out, (w, h))
    if not writer.isOpened():
        print(f"[err] failed to create writer: {out_path}")
        cap.release()
        return False

    frame_idx, kept = 0, 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret: break
        if args.video_step > 1 and (frame_idx % args.video_step != 0):
            writer.write(frame)  # skip frames: write original
            frame_idx += 1
            continue

        out_frame, _ = process(
            frame, det,
            tiles=args.tiles, overlap_ratio=args.overlap, nms_iou=args.nms,
            method=("pixelate" if args.pixelate else "gaussian"),
            ksize=args.ksize, pixel=args.pixel,
            phases=args.phases, polar_bands=args.polar_bands, polar_ratio=args.polar_ratio,
            soft_nms=args.soft_nms, enhance=args.enhance, upscale=args.upscale
        )
        writer.write(out_frame)
        frame_idx += 1; kept += 1

    writer.release()
    cap.release()
    dt = time.time() - t0
    print(f"[ok] {in_path} -> {out_path} (frames={frame_idx}, processed={kept}, {dt:.1f}s)")
    return True

# =============================
# Directory (recursive)
# =============================
def process_dir_serial(in_dir: Path, out_dir: Path, det, args) -> None:
    cnt, ok_cnt = 0, 0
    for root, _, files in os.walk(in_dir):
        root_path = Path(root)
        for fname in files:
            src = root_path / fname
            if not (is_image(src) or is_video(src)):
                continue
            rel = src.relative_to(in_dir)
            dst = out_dir / rel  # keep original extension
            ok = (process_image_file(src, dst, det, args) if is_image(src)
                  else process_video_file(src, dst, det, args))
            ok_cnt += int(ok); cnt += 1
    print(f"[done] total media: {cnt}, succeeded: {ok_cnt}, output root: {out_dir}")

# =============================
# Multiprocess workers
# =============================
_DET_CACHE = None

def _get_detector_cached(det_conf: dict):
    global _DET_CACHE
    if _DET_CACHE is not None:
        return _DET_CACHE
    use_yolo = (det_conf.get("yolo_face") is not None) or (det_conf.get("yolo_plate") is not None)
    if use_yolo:
        try:
            _DET_CACHE = YoloDet(face_weight=det_conf.get("yolo_face"),
                                 plate_weight=det_conf.get("yolo_plate"),
                                 conf=det_conf.get("yolo_conf", 0.25),
                                 iou=det_conf.get("yolo_iou", 0.50),
                                 imgsz=det_conf.get("yolo_imgsz", 1280))
        except Exception as e:
            print(f"[warn worker] YOLO init failed ({e}). Fallback to Cascade.")
            _DET_CACHE = CascadeDet(face_neighbors=det_conf.get("face_neighbors",5),
                                    plate_neighbors=det_conf.get("plate_neighbors",4))
    else:
        _DET_CACHE = CascadeDet(face_neighbors=det_conf.get("face_neighbors",5),
                                plate_neighbors=det_conf.get("plate_neighbors",4))
    return _DET_CACHE

def worker_process_path(task):
    src_s, dst_s, args_d, det_conf = task
    src, dst = Path(src_s), Path(dst_s)
    det = _get_detector_cached(det_conf)

    class A: pass
    args = A()
    for k,v in args_d.items(): setattr(args, k, v)

    try:
        if is_image(src):
            ok = process_image_file(src, dst, det, args)
        elif is_video(src):
            ok = process_video_file(src, dst, det, args)
        else:
            ok = False
    except Exception as e:
        print(f"[err] {src} -> {dst}: {e}")
        ok = False
    return ok

def process_dir_parallel(in_dir: Path, out_dir: Path, args, det_conf: dict):
    tasks = []
    for root, _, files in os.walk(in_dir):
        root_path = Path(root)
        for fname in files:
            src = root_path / fname
            if not (is_image(src) or is_video(src)):
                continue
            rel = src.relative_to(in_dir)
            dst = out_dir / rel
            args_d = dict(
                tiles=args.tiles, overlap=args.overlap, nms=args.nms,
                pixelate=args.pixelate, ksize=args.ksize, pixel=args.pixel,
                video_fps=args.video_fps, video_codec=args.video_codec, video_step=args.video_step,
                phases=args.phases, polar_bands=args.polar_bands, polar_ratio=args.polar_ratio,
                soft_nms=args.soft_nms, enhance=args.enhance, upscale=args.upscale
            )
            tasks.append((str(src), str(dst), args_d, det_conf))

    total = len(tasks); ok_cnt = 0
    if total == 0:
        print("[done] no media found.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(worker_process_path, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            ok = fut.result()
            ok_cnt += int(ok)
            if i % 10 == 0 or i == total:
                print(f"[progress] {i}/{total} done, ok={ok_cnt}")
    print(f"[done] total media: {total}, succeeded: {ok_cnt}, output root: {out_dir}")

# =============================
# CLI
# =============================
def parse_args():
    ap = argparse.ArgumentParser(description="Panorama PII blurring (faces & plates) — image/video/folder (fast)")
    ap.add_argument("--input", required=True, help="input equirectangular image/video OR folder")
    ap.add_argument("--out", default=None, help="output path (single file). default: <name>_blurred<ext>")
    ap.add_argument("--out-dir", default=None, help="output root dir (for folder). default: <input>_blurred")

    # Panorama options
    ap.add_argument("--tiles", type=int, default=8)
    ap.add_argument("--overlap", type=float, default=0.10)
    ap.add_argument("--nms", type=float, default=0.55)
    ap.add_argument("--pixelate", action="store_true", help="use mosaic instead of gaussian blur")
    ap.add_argument("--ksize", type=int, default=31, help="gaussian kernel (odd)")
    ap.add_argument("--pixel", type=int, default=16, help="mosaic block size")

    # Recall boosters
    ap.add_argument("--phases", type=int, default=1, help="horizontal tiling phases (1 or 2)")
    ap.add_argument("--polar-bands", type=int, default=0, help="extra vertical bands near poles")
    ap.add_argument("--polar-ratio", type=float, default=0.20, help="relative height per pole band (0~0.5)")
    ap.add_argument("--soft-nms", action="store_true", help="use Soft-NMS instead of hard NMS")
    ap.add_argument("--enhance", action="store_true", help="apply CLAHE+sharpen before detection")
    ap.add_argument("--upscale", type=float, default=1.0, help="detect on upscaled tiles (e.g., 1.25~1.5)")

    # YOLO (optional)
    ap.add_argument("--yolo-face", default=None, help="YOLO face weights .pt")
    ap.add_argument("--yolo-plate", default=None, help="YOLO plate weights .pt")
    ap.add_argument("--yolo-conf", type=float, default=0.25)
    ap.add_argument("--yolo-iou", type=float, default=0.50)
    ap.add_argument("--yolo-imgsz", type=int, default=1280)

    # Cascade tuning
    ap.add_argument("--face-nei", type=int, default=5, help="Cascade face minNeighbors")
    ap.add_argument("--plate-nei", type=int, default=4, help="Cascade plate minNeighbors")

    # Video options
    ap.add_argument("--video-fps", type=float, default=0.0, help="output FPS (0 = inherit)")
    ap.add_argument("--video-codec", type=str, default="mp4v", help="fourcc (e.g., mp4v, XVID, avc1)")
    ap.add_argument("--video-step", type=int, default=1, help="process every Nth frame (>=1)")

    # Multiprocess
    ap.add_argument("--jobs", type=int, default=1, help="parallel processes for folder mode")

    # FAST preset (speed-first)
    ap.add_argument("--fast", action="store_true", help="speed-first preset (good defaults for throughput)")
    return ap.parse_args()

def apply_fast_preset(args):
    # Speed-first balanced defaults
    args.phases = 1
    args.polar_bands = 0
    args.overlap = min(args.overlap, 0.10)
    args.upscale = 1.0
    args.enhance = False
    args.nms = min(args.nms, 0.55)
    # YOLO suggestions (not enforced, just clamp reasonably)
    if hasattr(args, "yolo_imgsz"): args.yolo_imgsz = min(args.yolo_imgsz, 1280)
    if hasattr(args, "yolo_conf"):  args.yolo_conf  = max(args.yolo_conf, 0.30)
    return args

def main():
    args = parse_args()
    if args.fast:
        args = apply_fast_preset(args)

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"not found: {in_path}")

    use_yolo = (args.yolo_face is not None) or (args.yolo_plate is not None)

    if in_path.is_file():
        # Single file mode
        if use_yolo:
            try:
                det = YoloDet(face_weight=args.yolo_face, plate_weight=args.yolo_plate,
                              conf=args.yolo_conf, iou=args.yolo_iou, imgsz=args.yolo_imgsz)
            except Exception as e:
                print(f"[warn] YOLO init failed ({e}). Fallback to Cascade.")
                det = CascadeDet(face_neighbors=args.face_nei, plate_neighbors=args.plate_nei)
        else:
            det = CascadeDet(face_neighbors=args.face_nei, plate_neighbors=args.plate_nei)

        if args.out is None:
            out_path = in_path.with_name(in_path.stem + "_blurred" + in_path.suffix)
        else:
            out_path = Path(args.out)

        if is_image(in_path):
            ok = process_image_file(in_path, out_path, det, args)
        elif is_video(in_path):
            ok = process_video_file(in_path, out_path, det, args)
        else:
            raise SystemExit(f"unsupported file type: {in_path.suffix}")
        if ok:
            print(f"done. saved -> {out_path}")
        return

    # Folder mode
    out_root = Path(args.out_dir) if args.out_dir else Path(str(in_path) + "_blurred")
    out_root.mkdir(parents=True, exist_ok=True)

    det_conf = dict(
        yolo_face=args.yolo_face,
        yolo_plate=args.yolo_plate,
        yolo_conf=args.yolo_conf,
        yolo_iou=args.yolo_iou,
        yolo_imgsz=args.yolo_imgsz,
        face_neighbors=args.face_nei,
        plate_neighbors=args.plate_nei
    )

    if args.jobs and args.jobs > 1:
        process_dir_parallel(in_path, out_root, args, det_conf)
    else:
        det = (YoloDet(face_weight=args.yolo_face, plate_weight=args.yolo_plate,
                       conf=args.yolo_conf, iou=args.yolo_iou, imgsz=args.yolo_imgsz)
               if use_yolo else CascadeDet(face_neighbors=args.face_nei, plate_neighbors=args.plate_nei))
        process_dir_serial(in_path, out_root, det, args)

if __name__ == "__main__":
    main()
