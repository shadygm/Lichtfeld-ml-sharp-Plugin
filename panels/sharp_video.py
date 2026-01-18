# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Sharp 4D Video Panel."""

import sys
import threading
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List

import numpy as np
import lichtfeld as lf

# Ensure plugin root is in path
_THIS_DIR = Path(__file__).parent.resolve()
_PLUGIN_ROOT = _THIS_DIR.parent
if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT))

import sharp_processor

class Stage(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    PLAYING = "playing"
    ERROR = "error"

@dataclass
class ProcessResult:
    success: bool
    ply_files: List[str] = field(default_factory=list)
    fps: float = 30.0
    error: Optional[str] = None

class ProcessingJob:
    def __init__(self, input_path: str, is_video: bool):
        self.input_path = input_path
        self.is_video = is_video
        self.progress = 0.0
        self.status = ""
        self.result = None
        self._thread = None
        self._lock = threading.Lock()

    def start(self, callback):
        self._thread = threading.Thread(target=self._run, args=(callback,), daemon=True)
        self._thread.start()

    def _run(self, callback):
        try:
            if self.is_video:
                processor = sharp_processor.SharpProcessor()
                
                def prog_cb(i, total, msg):
                    with self._lock:
                        self.status = msg
                        if total > 0:
                            self.progress = (i / total) * 100
                
                # Output dir is adjacent to video
                v_path = Path(self.input_path)
                out_dir = v_path.parent / (v_path.stem + "_gaussians")
                
                files, fps = processor.process_video(self.input_path, str(out_dir), prog_cb)
                
                # Unload model by deleting processor instance (assuming cleanup happens in __del__ or by GC)
                # If explicit unload needed, add method to processor. Here GC handles it.
                del processor
                
            else:
                # Direct PLY directory import
                with self._lock:
                    self.status = "Scanning PLY files..."
                    self.progress = 50.0
                
                p_path = Path(self.input_path)
                files = sorted([str(p) for p in p_path.glob("*.ply")])
                fps = 30.0 # Default for ply sequence
                
                if not files:
                    raise FileNotFoundError("No .ply files found in directory")

            with self._lock:
                self.result = ProcessResult(True, files, fps)
                self.progress = 100.0
                self.status = "Complete"
            
            callback(self.result)
            
        except Exception as e:
            logging.error(f"Processing failed: {e}")
            with self._lock:
                self.result = ProcessResult(False, error=str(e))
                self.status = f"Error: {e}"
            callback(self.result)

class SharpVideoPanel:
    panel_label = "Sharp 4D Video"
    panel_space = "SIDE_PANEL"
    panel_order = 5

    def __init__(self):
        self.input_path = str(Path.home() / "Videos")
        self.input_mode_video = True # True = Video, False = PLY Directory
        
        self.job = None
        self.ply_files = []
        self.playback_fps = 30.0
        self.current_frame_idx = 0
        self.last_frame_time = 0.0
        self.is_playing = False
        
        self.frame_cache = {} 
        self.cache_limit = 150 
        
        self.stage = Stage.IDLE

    def draw(self, layout):
        layout.heading("Sharp 4D Video")
        layout.label("Generate/Play 4DGS sequence")
        layout.separator()

        if self.stage == Stage.PROCESSING:
            if self.job:
                with self.job._lock:
                    layout.label(f"Status: {self.job.status}")
                    layout.progress_bar(self.job.progress / 100.0)
            return

        # Input Settings
        if layout.collapsing_header("Configuration", default_open=True):
            # Toggle Mode
            # layout.checkbox returns (changed, value)
            _, self.input_mode_video = layout.checkbox("Input is Video File", self.input_mode_video)
            
            label = "Video Path" if self.input_mode_video else "PLY Directory"
            dialog_label = "Select Video" if self.input_mode_video else "Select Directory"
            
            # path_input returns (changed, value)
            # folder_mode = not self.input_mode_video
            _, self.input_path = layout.path_input(label, self.input_path, not self.input_mode_video, dialog_label)
            
            _, self.playback_fps = layout.drag_float("Playback FPS", self.playback_fps, 1.0, 1.0, 120.0)

        if self.stage == Stage.IDLE or self.stage == Stage.ERROR:
            btn_label = "Process Video" if self.input_mode_video else "Load PLY Sequence"
            if layout.button(btn_label):
                self._start_processing()
            
            if self.stage == Stage.ERROR and self.job and self.job.result:
                layout.label(f"Error: {self.job.result.error}")

        # Playback Controls
        if self.ply_files:
            layout.separator()
            layout.heading("4D Sequence Playback")
            
            # Removed layout.row() as it's not supported. Using standard layout.
            if layout.button("Pause" if self.is_playing else "Play"):
                self.is_playing = not self.is_playing
            
            if layout.button("Reset Frame"):
                self.current_frame_idx = 0
                self._update_scene_frame(0)

            _, self.current_frame_idx = layout.drag_int(
                f"Frame {self.current_frame_idx+1}/{len(self.ply_files)}", 
                self.current_frame_idx, 0.5, 0, len(self.ply_files)-1
            )

            # Manual update trigger
            if not self.is_playing:
                self._update_scene_frame(self.current_frame_idx)

        # Background Playback Loop
        if self.is_playing and self.ply_files:
            now = time.time()
            frame_duration = 1.0 / self.playback_fps
            if now - self.last_frame_time >= frame_duration:
                self.current_frame_idx = (self.current_frame_idx + 1) % len(self.ply_files)
                self._update_scene_frame(self.current_frame_idx)
                self.last_frame_time = now

    def _start_processing(self):
        path = Path(self.input_path)
        if not path.exists():
             lf.log.error(f"Path not found: {self.input_path}")
             return

        self.job = ProcessingJob(self.input_path, self.input_mode_video)
        self.stage = Stage.PROCESSING
        self.job.start(self._on_complete)

    def _on_complete(self, result: ProcessResult):
        if result.success:
            self.ply_files = sorted(result.ply_files)
            # If loaded from PLY, respect the UI setting for FPS, or default to 30
            # If video, maybe we want to use video fps, but user override is fine
            
            self.stage = Stage.IDLE 
            self.current_frame_idx = 0
            self.is_playing = True
            self.stage = Stage.PLAYING
            
            # Start background preloading
            threading.Thread(target=self._preload_frames, daemon=True).start()
        else:
            self.stage = Stage.ERROR

    def _preload_frames(self):
        count = 0
        for p in self.ply_files:
            if count >= self.cache_limit: break
            if p not in self.frame_cache:
                try:
                    self.frame_cache[p] = sharp_processor.load_gaussian_ply(p)
                    count += 1
                except:
                    pass


    def _update_scene_frame(self, idx, node_name=None):
        if not self.ply_files:
            return

        node_name = node_name or "Sharp4D"
        path = Path(self.ply_files[idx])

        try:
            result = lf.io.load(str(path))
            splat = result.splat_data
            if splat is None:
                raise RuntimeError("No splat data returned")
        except Exception as e:
            lf.log.error(f"Failed to load splat frame {path}: {e}")
            return

        scene = lf.get_scene()
        if scene is None:
            lf.log.error("No active scene available.")
            self.stage = Stage.ERROR
            return

        new_node_name = f"{node_name}__next"

        lf.log.info(f"Adding frame {idx+1}/{len(self.ply_files)}: {path}")

        scene.add_splat(
            name=new_node_name,
            means=splat.means_raw,
            sh0=splat.sh0_raw,
            shN=splat.shN_raw,
            scaling=splat.scaling_raw,
            rotation=splat.rotation_raw,
            opacity=splat.opacity_raw,
            sh_degree=splat.active_sh_degree,
            scene_scale=splat.scene_scale,
        )

        old_node = scene.get_node(node_name)
        if old_node:
            scene.remove_node(old_node.name)

        scene.rename_node(new_node_name, node_name)
        scene.invalidate_cache()






