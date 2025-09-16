#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GUI for Panorama PII blurring tool.
- Main window: Input path, Output path, Run, Stop (+ Settings button)
- Settings dialog: all advanced options (tiles, overlap, nms, pixelate, YOLO, etc.)
- Threaded execution with graceful stop between files/frames
- Uses functions/classes from pano_blur.py (keep original CLI script intact)
"""

import os
import sys
import json
import queue
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# === Import your original module (save your code as pano_blur.py) ===
# It must expose: CascadeDet, YoloDet, process (frame/image), is_image, is_video
try:
    import cv2  # needed in UI for custom video loop with stop
    import pano_blur as core
except Exception as e:
    print("pano_blur.py import error:", e)
    raise

APP_TITLE = "로드뷰 번호판/얼굴 블러링 툴"
SETTINGS_FILE = Path.home() / ".pano_blur_gui_settings.json"

DEFAULT_INPUT  = Path("./input").resolve()
DEFAULT_OUTPUT = Path("./output").resolve()
DEFAULT_FACE   = Path("./models/yolov8x-face-lindevs.pt").resolve()
DEFAULT_PLATE  = Path("./models/best.pt").resolve()

# -------------------------
# Helpers
# -------------------------
def load_settings():
    first_run = not SETTINGS_FILE.exists()
    if not first_run:
        try:
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass

    # ---- 기본값 (요청하신 CLI와 동일) ----
    cfg = {
        "tiles": 10,          # ← tiles 10
        "overlap": 0.10,
        "nms": 0.55,
        "pixelate": False,
        "ksize": 31,
        "pixel": 16,
        "phases": 1,
        "polar_bands": 0,
        "polar_ratio": 0.20,
        "soft_nms": False,
        "enhance": False,
        "upscale": 1.0,

        # YOLO (요청 경로)
        "yolo_face": str(DEFAULT_FACE),
        "yolo_plate": str(DEFAULT_PLATE),
        "yolo_conf": 0.25,
        "yolo_iou": 0.50,
        "yolo_imgsz": 1280,

        # Cascade
        "face_nei": 5,
        "plate_nei": 4,

        # Video
        "video_fps": 0.0,
        "video_codec": "mp4v",
        "video_step": 1,

        # 병렬
        "jobs": 1,

        # FAST preset on
        "fast": True,         # ← FAST 활성화
    }

    # 최초 실행이면 저장까지 해 두면 다음부터 그대로 뜹니다.
    try:
        SETTINGS_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    return cfg

def save_settings(cfg: dict):
    try:
        SETTINGS_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print("settings save failed:", e)

def make_args_obj(cfg: dict):
    """Build a simple object with attributes like argparse Namespace."""
    class A: pass
    a = A()
    a.tiles = int(cfg["tiles"])
    a.overlap = float(cfg["overlap"])
    a.nms = float(cfg["nms"])
    a.pixelate = bool(cfg["pixelate"])
    a.ksize = int(cfg["ksize"])
    a.pixel = int(cfg["pixel"])
    a.phases = int(cfg["phases"])
    a.polar_bands = int(cfg["polar_bands"])
    a.polar_ratio = float(cfg["polar_ratio"])
    a.soft_nms = bool(cfg["soft_nms"])
    a.enhance = bool(cfg["enhance"])
    a.upscale = float(cfg["upscale"])
    a.video_fps = float(cfg["video_fps"])
    a.video_codec = str(cfg["video_codec"])
    a.video_step = int(cfg["video_step"])
    a.jobs = int(cfg["jobs"])
    a.yolo_face = cfg["yolo_face"] or None
    a.yolo_plate = cfg["yolo_plate"] or None
    a.yolo_conf = float(cfg["yolo_conf"])
    a.yolo_iou = float(cfg["yolo_iou"])
    a.yolo_imgsz = int(cfg["yolo_imgsz"])
    a.face_nei = int(cfg["face_nei"])
    a.plate_nei = int(cfg["plate_nei"])
    a.fast = bool(cfg["fast"])
    return a

# -------------------------
# Worker (with stop support)
# -------------------------
class Runner(threading.Thread):
    def __init__(self, input_path: Path, output_path: Path, cfg: dict, log_cb, done_cb, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.in_path = input_path
        self.out_path = output_path
        self.cfg = cfg
        self.log = log_cb
        self.done_cb = done_cb
        self.stop_event = stop_event

    def build_detector(self, args):
        use_yolo = (args.yolo_face is not None) or (args.yolo_plate is not None)
        if args.fast:
            # mimic CLI fast preset
            args.phases = 1
            args.polar_bands = 0
            args.overlap = min(args.overlap, 0.10)
            args.upscale = 1.0
            args.enhance = False
            args.nms = min(args.nms, 0.55)
            args.yolo_imgsz = min(args.yolo_imgsz, 1280)
            args.yolo_conf = max(args.yolo_conf, 0.30)
        if use_yolo:
            try:
                det = core.YoloDet(face_weight=args.yolo_face, plate_weight=args.yolo_plate,
                                   conf=args.yolo_conf, iou=args.yolo_iou, imgsz=args.yolo_imgsz)
            except Exception as e:
                self.log(f"[warn] YOLO init failed ({e}). Fallback to Cascade.")
                det = core.CascadeDet(face_neighbors=args.face_nei, plate_neighbors=args.plate_nei)
        else:
            det = core.CascadeDet(face_neighbors=args.face_nei, plate_neighbors=args.plate_nei)
        return det

    def run_image(self, img_path: Path, out_path: Path, det, args):
        # reuse original helper for simplicity
        ok = core.process_image_file(img_path, out_path, det, args)
        return ok

    def run_video(self, vid_path: Path, out_path: Path, det, args):
        """
        Custom video loop that checks stop_event each frame.
        This mirrors core.process_video_file() but adds graceful stop.
        """
        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            self.log(f"[skip] cannot open video: {vid_path}")
            return False

        fps_in = cap.get(cv2.CAP_PROP_FPS)
        if fps_in <= 1e-2: fps_in = 30.0
        fps_out = args.video_fps if args.video_fps > 0 else fps_in

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*args.video_codec)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_path), fourcc, fps_out, (w, h))
        if not writer.isOpened():
            self.log(f"[err] failed to create writer: {out_path}")
            cap.release()
            return False

        frame_idx, kept = 0, 0
        try:
            while True:
                if self.stop_event.is_set():
                    self.log("[stop] requested. finishing current loop.")
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                if args.video_step > 1 and (frame_idx % args.video_step != 0):
                    writer.write(frame)  # pass-through
                    frame_idx += 1
                    continue

                out_frame, _ = core.process(
                    frame, det,
                    tiles=args.tiles, overlap_ratio=args.overlap, nms_iou=args.nms,
                    method=("pixelate" if args.pixelate else "gaussian"),
                    ksize=args.ksize, pixel=args.pixel,
                    phases=args.phases, polar_bands=args.polar_bands, polar_ratio=args.polar_ratio,
                    soft_nms=args.soft_nms, enhance=args.enhance, upscale=args.upscale
                )
                writer.write(out_frame)
                frame_idx += 1; kept += 1
        finally:
            writer.release()
            cap.release()
        self.log(f"[ok] {vid_path} -> {out_path} (frames={frame_idx}, processed={kept})")
        return True

    def run_dir(self, in_dir: Path, out_dir: Path, det, args):
        # 1) 처리 계획(모든 미디어 목록) 먼저 수집 -> 고정 분모 확보
        plan = []
        for root, _, files in os.walk(in_dir):
            root_path = Path(root)
            for fname in files:
                src = root_path / fname
                if not (core.is_image(src) or core.is_video(src)):
                    continue
                rel = src.relative_to(in_dir)
                dst = out_dir / rel
                plan.append((src, dst))

        grand_total = len(plan)
        if grand_total == 0:
            self.log(f"[done] no media found in: {in_dir}")
            return

        ok_cnt = 0
        # 2) 실제 처리 루프 (stop 이벤트 대응)
        for i, (src, dst) in enumerate(plan, 1):
            if self.stop_event.is_set():
                self.log("[stop] requested. stopping before next file.")
                break

            dst.parent.mkdir(parents=True, exist_ok=True)
            ok = False
            try:
                if core.is_image(src):
                    ok = self.run_image(src, dst, det, args)
                else:
                    ok = self.run_video(src, dst, det, args)
            except Exception as e:
                self.log(f"[err] {src} -> {dst}: {e}")
                ok = False

            ok_cnt += int(ok)
            # 진행률: i/전체, 성공: ok_cnt
            self.log(f"[progress] {i}/{grand_total} processed, ok={ok_cnt}")

        # 3) 최종 요약
        processed = min(i, grand_total) if grand_total else 0
        self.log(f"[done] total media: {grand_total}, processed: {processed}, "
                 f"succeeded: {ok_cnt}, failed: {processed - ok_cnt}, output root: {out_dir}")

    def run(self):
        try:
            args = make_args_obj(self.cfg)
            det = self.build_detector(args)

            # Decide mode
            if self.in_path.is_file():
                # single file mode
                if self.out_path.is_dir():
                    out_path = self.out_path / (self.in_path.stem + "_blurred" + self.in_path.suffix)
                else:
                    out_path = self.out_path
                if core.is_image(self.in_path):
                    ok = self.run_image(self.in_path, out_path, det, args)
                elif core.is_video(self.in_path):
                    ok = self.run_video(self.in_path, out_path, det, args)
                else:
                    self.log(f"[err] unsupported file type: {self.in_path.suffix}")
                    ok = False
                if ok:
                    self.log(f"done. saved -> {out_path}")
            else:
                # folder mode
                out_dir = self.out_path if self.out_path else Path(str(self.in_path) + "_blurred")
                out_dir.mkdir(parents=True, exist_ok=True)
                self.run_dir(self.in_path, out_dir, det, args)
        except Exception as e:
            self.log(f"[runtime err] {e}")
        finally:
            self.done_cb()

# -------------------------
# Settings Dialog
# -------------------------
class SettingsDialog(tk.Toplevel):
    def __init__(self, master, cfg: dict, on_ok):
        super().__init__(master)
        self.title("설정")
        self.resizable(False, False)
        self.cfg = cfg.copy()
        self.on_ok = on_ok

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=10, pady=10)

        self.frames = {}
        def add_tab(name):
            f = ttk.Frame(nb)
            nb.add(f, text=name)
            self.frames[name] = f
            return f

        # ---- General
        gen = add_tab("일반")
        self._int_spin(gen, "tiles", "Tiles", 1, 64, 1, 0, 0)
        self._float_spin(gen, "overlap", "Overlap", 0.0, 0.5, 0.01, 0, 2)
        self._float_spin(gen, "nms", "NMS IoU", 0.1, 0.9, 0.01, 0, 4)
        self._check(gen, "pixelate", "Pixelate (mosaic)", 1, 0)
        self._int_spin(gen, "ksize", "Gaussian ksize", 3, 255, 2, 1, 2, help_="(odd number)")
        self._int_spin(gen, "pixel", "Mosaic block", 2, 128, 1, 1, 4)

        # ---- Recall / Advanced
        adv = add_tab("고급")
        self._int_spin(adv, "phases", "Phases (1-2)", 1, 2, 1, 0, 0)
        self._int_spin(adv, "polar_bands", "Polar bands", 0, 6, 1, 0, 2)
        self._float_spin(adv, "polar_ratio", "Polar ratio", 0.0, 0.5, 0.01, 0, 4)
        self._check(adv, "soft_nms", "Soft-NMS", 1, 0)
        self._check(adv, "enhance", "Pre-enhance (CLAHE+sharp)", 1, 2)
        self._float_spin(adv, "upscale", "Upscale", 1.0, 2.0, 0.05, 1, 0)

        # ---- YOLO
        yolo = add_tab("YOLO")
        self._file_row(yolo, "yolo_face", "Face weights (.pt)", 0, 0, filetypes=[("PyTorch weights","*.pt"),("All","*.*")])
        self._file_row(yolo, "yolo_plate", "Plate weights (.pt)", 1, 0, filetypes=[("PyTorch weights","*.pt"),("All","*.*")])
        self._float_spin(yolo, "yolo_conf", "YOLO conf", 0.05, 0.95, 0.01, 2, 0)
        self._float_spin(yolo, "yolo_iou", "YOLO IoU", 0.1, 0.95, 0.01, 2, 2)
        self._int_spin(yolo, "yolo_imgsz", "YOLO imgsz", 320, 2048, 32, 3, 0)

        # ---- Cascade
        cas = add_tab("Cascade")
        self._int_spin(cas, "face_nei", "Face minNeighbors", 1, 12, 1, 0, 0)
        self._int_spin(cas, "plate_nei", "Plate minNeighbors", 1, 12, 1, 0, 2)

        # ---- Video
        vid = add_tab("비디오")
        self._float_spin(vid, "video_fps", "Output FPS (0=inherit)", 0.0, 120.0, 1.0, 0, 0)
        self._entry(vid, "video_codec", "FOURCC (e.g., mp4v)", 0, 2, width=10)
        self._int_spin(vid, "video_step", "Process every Nth frame", 1, 16, 1, 1, 0)

        # ---- Preset
        preset = add_tab("프리셋")
        self._check(preset, "fast", "FAST preset (speed-first)", 0, 0)
        self._int_spin(preset, "jobs", "Parallel jobs (dir)", 1, 64, 1, 1, 0, help_="(GUI는 안정적 중지를 위해 단일 프로세스 권장)")

        # Action buttons
        btns = ttk.Frame(self)
        btns.pack(fill="x", padx=10, pady=(0,10))
        ttk.Button(btns, text="저장", command=self._save).pack(side="right", padx=(5,0))
        ttk.Button(btns, text="닫기", command=self.destroy).pack(side="right")

    # ---- UI elements builders ----
    def _entry(self, parent, key, label, r, c, width=20):
        ttk.Label(parent, text=label).grid(row=r, column=c, sticky="w", padx=4, pady=4)
        var = tk.StringVar(value=str(self.cfg.get(key, "")))
        ent = ttk.Entry(parent, textvariable=var, width=width)
        ent.grid(row=r, column=c+1, sticky="w", padx=4, pady=4)
        setattr(self, f"var_{key}", var)

    def _check(self, parent, key, label, r, c):
        var = tk.BooleanVar(value=bool(self.cfg.get(key, False)))
        cb = ttk.Checkbutton(parent, text=label, variable=var)
        cb.grid(row=r, column=c, sticky="w", padx=4, pady=4, columnspan=2)
        setattr(self, f"var_{key}", var)

    def _int_spin(self, parent, key, label, mn, mx, step, r, c, help_=None):
        ttk.Label(parent, text=label).grid(row=r, column=c, sticky="w", padx=4, pady=4)
        var = tk.IntVar(value=int(self.cfg.get(key, mn)))
        sp = ttk.Spinbox(parent, from_=mn, to=mx, increment=step, textvariable=var, width=8)
        sp.grid(row=r, column=c+1, sticky="w", padx=4, pady=4)
        if help_:
            ttk.Label(parent, text=help_, foreground="#777").grid(row=r, column=c+2, sticky="w")
        setattr(self, f"var_{key}", var)

    def _float_spin(self, parent, key, label, mn, mx, step, r, c, help_=None):
        ttk.Label(parent, text=label).grid(row=r, column=c, sticky="w", padx=4, pady=4)
        var = tk.DoubleVar(value=float(self.cfg.get(key, mn)))
        sp = ttk.Spinbox(parent, from_=mn, to=mx, increment=step, textvariable=var, width=8)
        sp.grid(row=r, column=c+1, sticky="w", padx=4, pady=4)
        if help_:
            ttk.Label(parent, text=help_, foreground="#777").grid(row=r, column=c+2, sticky="w")
        setattr(self, f"var_{key}", var)

    def _file_row(self, parent, key, label, r, c, filetypes):
        ttk.Label(parent, text=label).grid(row=r, column=c, sticky="w", padx=4, pady=4)
        var = tk.StringVar(value=str(self.cfg.get(key, "") or ""))
        ent = ttk.Entry(parent, textvariable=var, width=40)
        ent.grid(row=r, column=c+1, sticky="w", padx=4, pady=4)
        def browse():
            path = filedialog.askopenfilename(filetypes=filetypes)
            if path:
                var.set(path)
        ttk.Button(parent, text="찾기", command=browse).grid(row=r, column=c+2, padx=4)
        setattr(self, f"var_{key}", var)

    def _save(self):
        # collect values
        for k, v in self.__dict__.items():
            if not k.startswith("var_"): continue
            key = k[4:]
            val = v.get()
            self.cfg[key] = val
        save_settings(self.cfg)
        self.on_ok(self.cfg)
        messagebox.showinfo("저장됨", "설정이 저장되었습니다.")

# -------------------------
# Main App
# -------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("960x540")
        self.cfg = load_settings()

        self.var_input = tk.StringVar()
        self.var_output = tk.StringVar()

        # 입력/출력 경로 기본값 채우기 (비어 있을 때만)
        if not self.var_input.get():
            self.var_input.set(str(DEFAULT_INPUT))
        if not self.var_output.get():
            self.var_output.set(str(DEFAULT_OUTPUT))

        # Path row
        frm_top = ttk.Frame(self)
        frm_top.pack(fill="x", padx=10, pady=10)

        ttk.Label(frm_top, text="Input (파일 또는 폴더):").grid(row=0, column=0, sticky="w")
        ent_in = ttk.Entry(frm_top, textvariable=self.var_input, width=70)
        ent_in.grid(row=0, column=1, sticky="we", padx=6)
        ttk.Button(frm_top, text="파일", command=self.browse_input_file).grid(row=0, column=2, padx=2)
        ttk.Button(frm_top, text="폴더", command=self.browse_input_dir).grid(row=0, column=3, padx=2)

        ttk.Label(frm_top, text="Output 경로:").grid(row=1, column=0, sticky="w", pady=(6,0))
        ent_out = ttk.Entry(frm_top, textvariable=self.var_output, width=70)
        ent_out.grid(row=1, column=1, sticky="we", padx=6, pady=(6,0))
        ttk.Button(frm_top, text="폴더", command=self.browse_output_dir).grid(row=1, column=2, padx=2, pady=(6,0))

        frm_btn = ttk.Frame(self)
        frm_btn.pack(fill="x", padx=10, pady=(0,8))
        self.btn_run = ttk.Button(frm_btn, text="실행", command=self.on_run)
        self.btn_run.pack(side="left")
        self.btn_stop = ttk.Button(frm_btn, text="중지", command=self.on_stop, state="disabled")
        self.btn_stop.pack(side="left", padx=(6,0))
        ttk.Button(frm_btn, text="설정", command=self.open_settings).pack(side="right")

        # Log
        frm_log = ttk.Frame(self)
        frm_log.pack(fill="both", expand=True, padx=10, pady=10)
        self.txt = tk.Text(frm_log, height=18)
        self.txt.pack(fill="both", expand=True, side="left")
        sb = ttk.Scrollbar(frm_log, orient="vertical", command=self.txt.yview)
        sb.pack(side="right", fill="y")
        self.txt.configure(yscrollcommand=sb.set)

        # Worker state
        self.stop_event = threading.Event()
        self.worker: Runner | None = None

        # async log queue (thread-safe)
        self.log_q = queue.Queue()
        self.after(100, self._drain_log)

    # ---- UI actions ----
    def browse_input_file(self):
        path = filedialog.askopenfilename(title="입력 파일 선택")
        if path:
            self.var_input.set(path)

    def browse_input_dir(self):
        path = filedialog.askdirectory(title="입력 폴더 선택")
        if path:
            self.var_input.set(path)

    def browse_output_dir(self):
        path = filedialog.askdirectory(title="출력 폴더 선택")
        if path:
            self.var_output.set(path)

    def open_settings(self):
        SettingsDialog(self, self.cfg, self._on_settings_saved)

    def _on_settings_saved(self, new_cfg):
        self.cfg = new_cfg  # live update

    # ---- Run / Stop ----
    def on_run(self):
        if self.worker and self.worker.is_alive():
            messagebox.showwarning("실행 중", "이미 작업이 실행 중입니다.")
            return
        in_path = Path(self.var_input.get().strip())
        if not in_path.exists():
            messagebox.showerror("오류", "입력 경로가 올바르지 않습니다.")
            return

        out_str = self.var_output.get().strip()
        if in_path.is_file():
            # file mode: if output blank -> generate sibling _blurred
            if not out_str:
                out_path = in_path.with_name(in_path.stem + "_blurred" + in_path.suffix)
            else:
                out_path = Path(out_str)
                if out_path.is_dir():
                    out_path = out_path / (in_path.stem + "_blurred" + in_path.suffix)
        else:
            # dir mode: if output blank -> <input>_blurred
            out_path = Path(out_str) if out_str else Path(str(in_path) + "_blurred")

        # Start worker
        self.stop_event.clear()
        self.worker = Runner(in_path, out_path, self.cfg, self._log, self._on_done, self.stop_event)
        self.btn_run.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self._log(f"[start] input={in_path}  output={out_path}")
        self.worker.start()

    def on_stop(self):
        if self.worker and self.worker.is_alive():
            self.stop_event.set()
            self._log("[stop] 요청됨 (안전 중지). 파일/프레임 경계에서 멈춥니다.")
        else:
            self._log("[stop] 실행 중인 작업이 없습니다.")

    def _on_done(self):
        self._log("[done] 작업이 종료되었습니다.")
        self.btn_run.configure(state="normal")
        self.btn_stop.configure(state="disabled")

    # ---- Logging ----
    def _log(self, msg: str):
        # thread-safe enqueue
        self.log_q.put(str(msg))

    def _drain_log(self):
        try:
            while True:
                msg = self.log_q.get_nowait()
                self.txt.insert("end", msg + "\n")
                self.txt.see("end")
        except queue.Empty:
            pass
        self.after(100, self._drain_log)

# -------------------------
if __name__ == "__main__":
    app = App()
    app.mainloop()
