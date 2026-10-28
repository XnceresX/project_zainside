"""
safepass_studio.py
==================
SAFEPASS Autonomous CCTV & Webcam Person Re-Identification Studio.
Ported and integrated with https://github.com/meethu51/ReID-framework.git

Features:
  1. Autonomous Person Tracking & Re-ID: Persons are automatically detected and identified
     with persistent Global IDs (Person_0001, Person_0002, etc.) without requiring manual
     face or clothing input.
  2. Multi-Source Support: Live Laptop Webcams (Cam 0, Cam 1), CCTV Video Files (.mp4, .avi),
     and RTSP surveillance streams.
  3. Cross-Camera Re-ID: Combines OSNet appearance embeddings, temporal continuity, and
     topological handoff zones.
  4. Interactive Zone Drawing & Handoff Configuration.
"""

from __future__ import annotations

import os
import sys
import time
import threading
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ReID Framework Modules
from reid import (
    GlobalIDManager,
    PersonDetector,
    ReIDFeatureExtractor,
    SimpleTracker,
    load_zones,
    zone_confidence,
)
from reid.zone_editor import run_zone_editor
from cctv.reid_cctv import MultiCameraCCTVReIDEngine


class SafePassReIDStudio:
    """
    Main Desktop GUI Studio for Autonomous Person Re-ID across CCTV cameras & webcams.
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SAFEPASS - Autonomous Person Re-ID Studio")
        self.root.geometry("1060x780")
        self.root.minsize(920, 680)
        self.root.configure(bg="#0b0f19")

        # Global Re-ID State
        self.selected_sources: List[str] = ["0"]
        self.handoff_zone_file: Optional[str] = None
        self.engine: Optional[MultiCameraCCTVReIDEngine] = None
        self.is_running = False

        self._build_ui()

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", background="#0b0f19", foreground="#f1f5f9", font=("Segoe UI", 10))
        style.configure("TFrame", background="#0b0f19")
        style.configure("Card.TFrame", background="#111827", relief="ridge")
        style.configure("TCheckbutton", background="#111827", foreground="#94a3b8", font=("Segoe UI", 9))
        style.configure("TCombobox", fieldbackground="#1f2937", background="#374151", foreground="#ffffff")

        # Header Frame
        header_frame = tk.Frame(self.root, bg="#0b0f19", pady=12)
        header_frame.pack(fill="x", padx=20)

        title_lbl = tk.Label(
            header_frame,
            text="SAFEPASS AUTONOMOUS PERSON RE-ID STUDIO",
            font=("Segoe UI", 15, "bold"),
            fg="#00e5ff",
            bg="#0b0f19",
        )
        title_lbl.pack(anchor="w")

        subtitle_lbl = tk.Label(
            header_frame,
            text="Multi-Camera CCTV Tracking & Re-Identification Engine · Autonomous Person IDs (No Manual Input Required)",
            font=("Segoe UI", 9),
            fg="#94a3b8",
            bg="#0b0f19",
        )
        subtitle_lbl.pack(anchor="w")

        # Two-Column Layout Container
        main_container = tk.Frame(self.root, bg="#0b0f19")
        main_container.pack(fill="both", expand=True, padx=20, pady=5)

        left_col = tk.Frame(main_container, bg="#0b0f19")
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 10))

        right_col = tk.Frame(main_container, bg="#0b0f19", width=460)
        right_col.pack(side="right", fill="both", padx=(10, 0))

        # ---------------------------------------------------------------
        # LEFT: 1. Video Feeds & CCTV Input Sources
        # ---------------------------------------------------------------
        card1 = tk.LabelFrame(
            left_col,
            text=" 1. CCTV Video Feeds & Camera Sources ",
            bg="#111827",
            fg="#00e5ff",
            font=("Segoe UI", 10, "bold"),
            padx=15,
            pady=12,
            bd=1,
            relief="solid",
        )
        card1.pack(fill="x", pady=(0, 12))

        self.source_mode = tk.StringVar(value="webcam")

        rb_frame = tk.Frame(card1, bg="#111827")
        rb_frame.pack(fill="x", pady=(0, 10))

        tk.Radiobutton(
            rb_frame,
            text="Live Laptop Front Webcam (Camera 0)",
            variable=self.source_mode,
            value="webcam",
            command=self._on_source_mode_changed,
            bg="#111827",
            fg="#f8fafc",
            activebackground="#111827",
            activeforeground="#38bdf8",
            selectcolor="#1f2937",
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", pady=2)

        tk.Radiobutton(
            rb_frame,
            text="CCTV Video File(s) (.mp4, .avi, etc.)",
            variable=self.source_mode,
            value="files",
            command=self._on_source_mode_changed,
            bg="#111827",
            fg="#f8fafc",
            activebackground="#111827",
            activeforeground="#38bdf8",
            selectcolor="#1f2937",
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", pady=2)

        tk.Radiobutton(
            rb_frame,
            text="Custom RTSP / IP Camera Stream",
            variable=self.source_mode,
            value="rtsp",
            command=self._on_source_mode_changed,
            bg="#111827",
            fg="#f8fafc",
            activebackground="#111827",
            activeforeground="#38bdf8",
            selectcolor="#1f2937",
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", pady=2)

        # File Chooser Bar
        self.file_btn_frame = tk.Frame(card1, bg="#111827")
        self.file_btn_frame.pack(fill="x", pady=5)

        btn_browse_video = tk.Button(
            self.file_btn_frame,
            text="📁 Select CCTV Video File(s)...",
            command=self._on_browse_videos,
            bg="#1f2937",
            fg="#e2e8f0",
            font=("Segoe UI", 9, "bold"),
            padx=12,
            pady=5,
            relief="groove",
        )
        btn_browse_video.pack(side="left", padx=(0, 10))

        btn_clear_sources = tk.Button(
            self.file_btn_frame,
            text="Reset to Webcam",
            command=self._on_reset_sources,
            bg="#374151",
            fg="#94a3b8",
            font=("Segoe UI", 8),
            padx=8,
            pady=4,
            relief="flat",
        )
        btn_clear_sources.pack(side="left")

        # RTSP Entry Frame
        self.rtsp_frame = tk.Frame(card1, bg="#111827")
        tk.Label(self.rtsp_frame, text="RTSP Stream URL:", bg="#111827", fg="#cbd5e1", font=("Segoe UI", 9)).pack(side="left", padx=(0, 6))
        self.rtsp_var = tk.StringVar(value="rtsp://")
        self.rtsp_entry = tk.Entry(self.rtsp_frame, textvariable=self.rtsp_var, bg="#1f2937", fg="#ffffff", font=("Segoe UI", 9), width=32)
        self.rtsp_entry.pack(side="left", fill="x", expand=True)

        # Selected Sources Listbox
        tk.Label(card1, text="Active Video Feeds / Cameras:", bg="#111827", fg="#94a3b8", font=("Segoe UI", 8)).pack(anchor="w", pady=(8, 2))
        self.source_listbox = tk.Listbox(card1, bg="#0d131f", fg="#38bdf8", font=("Consolas", 9), height=4, bd=1, relief="sunken")
        self.source_listbox.pack(fill="x")
        self.source_listbox.insert("end", "Cam 1: Laptop Front Webcam (Index 0)")

        # ---------------------------------------------------------------
        # LEFT: 2. Re-ID Framework & Topology Engine
        # ---------------------------------------------------------------
        card2 = tk.LabelFrame(
            left_col,
            text=" 2. Re-ID Engine & Topology Configuration ",
            bg="#111827",
            fg="#00e5ff",
            font=("Segoe UI", 10, "bold"),
            padx=15,
            pady=12,
            bd=1,
            relief="solid",
        )
        card2.pack(fill="x", pady=(0, 10))

        cfg_grid = tk.Frame(card2, bg="#111827")
        cfg_grid.pack(fill="x", pady=4)

        # Detector Mode
        tk.Label(cfg_grid, text="Detection Backend:", bg="#111827", fg="#cbd5e1", font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w", pady=4)
        self.detector_var = tk.StringVar(value="Auto (HOG Pedestrian + Face/Body)")
        combo_det = ttk.Combobox(
            cfg_grid,
            textvariable=self.detector_var,
            values=["Auto (HOG Pedestrian + Face/Body)", "OpenCV HOG Pedestrian Only", "Color/Blob Signature Detector"],
            width=28,
            font=("Segoe UI", 9),
            state="readonly",
        )
        combo_det.grid(row=0, column=1, sticky="w", padx=10, pady=4)

        # Re-ID Embedding Backbone
        tk.Label(cfg_grid, text="Re-ID Backbone:", bg="#111827", fg="#cbd5e1", font=("Segoe UI", 9)).grid(row=1, column=0, sticky="w", pady=4)
        self.model_var = tk.StringVar(value="TorchReID OSNet (512-dim)")
        combo_model = ttk.Combobox(
            cfg_grid,
            textvariable=self.model_var,
            values=["TorchReID OSNet (512-dim)", "Fast Spatial Color/Geometry Stub"],
            width=28,
            font=("Segoe UI", 9),
            state="readonly",
        )
        combo_model.grid(row=1, column=1, sticky="w", padx=10, pady=4)

        # Person ID Prefix
        tk.Label(cfg_grid, text="Person ID Format:", bg="#111827", fg="#cbd5e1", font=("Segoe UI", 9)).grid(row=2, column=0, sticky="w", pady=4)
        self.prefix_var = tk.StringVar(value="Person_")
        combo_prefix = ttk.Combobox(
            cfg_grid,
            textvariable=self.prefix_var,
            values=["Person_", "G", "ID_"],
            width=12,
            font=("Segoe UI", 9),
        )
        combo_prefix.grid(row=2, column=1, sticky="w", padx=10, pady=4)

        # Thresholds
        thresh_frame = tk.Frame(card2, bg="#111827")
        thresh_frame.pack(fill="x", pady=6)

        tk.Label(thresh_frame, text="Re-ID Cosine Threshold:", bg="#111827", fg="#cbd5e1", font=("Segoe UI", 9)).pack(side="left")
        self.sim_thresh_var = tk.DoubleVar(value=0.52)
        tk.Scale(
            thresh_frame,
            variable=self.sim_thresh_var,
            from_=0.35,
            to=0.85,
            resolution=0.01,
            orient="horizontal",
            bg="#111827",
            fg="#00e5ff",
            highlightthickness=0,
            length=130,
        ).pack(side="left", padx=8)

        # Topological Handoff Zones
        zone_bar = tk.Frame(card2, bg="#111827")
        zone_bar.pack(fill="x", pady=(8, 2))

        btn_load_zone = tk.Button(
            zone_bar,
            text="🗺️ Load Handoff Zones...",
            command=self._on_load_zones,
            bg="#1f2937",
            fg="#e2e8f0",
            font=("Segoe UI", 8),
            padx=8,
            pady=3,
        )
        btn_load_zone.pack(side="left", padx=(0, 8))

        btn_draw_zone = tk.Button(
            zone_bar,
            text="✏️ Draw New Zone on Feed",
            command=self._on_draw_zone,
            bg="#1f2937",
            fg="#e2e8f0",
            font=("Segoe UI", 8),
            padx=8,
            pady=3,
        )
        btn_draw_zone.pack(side="left")

        self.lbl_zone_status = tk.Label(
            card2,
            text="Handoff Zones: None loaded (standard cross-camera Re-ID active)",
            bg="#111827",
            fg="#64748b",
            font=("Segoe UI", 8, "italic"),
        )
        self.lbl_zone_status.pack(anchor="w", pady=(4, 0))

        # ---------------------------------------------------------------
        # RIGHT: 3. Session Overview & Active Identity Roster
        # ---------------------------------------------------------------
        card_roster = tk.LabelFrame(
            right_col,
            text=" 3. Real-Time Re-ID Session & Controls ",
            bg="#111827",
            fg="#00e5ff",
            font=("Segoe UI", 10, "bold"),
            padx=15,
            pady=12,
            bd=1,
            relief="solid",
        )
        card_roster.pack(fill="both", expand=True)

        # Telemetry / Status Box
        self.info_box = tk.Text(
            card_roster,
            bg="#0d131f",
            fg="#f8fafc",
            font=("Consolas", 9),
            height=16,
            bd=1,
            relief="sunken",
            padx=10,
            pady=10,
        )
        self.info_box.pack(fill="both", expand=True, pady=(0, 15))
        self.info_box.config(state="disabled")

        # Launch Live ReID Button
        self.btn_launch = tk.Button(
            card_roster,
            text="🚀 LAUNCH PERSON RE-ID STREAM",
            command=self._on_launch_reid,
            bg="#10b981",
            fg="#ffffff",
            font=("Segoe UI", 11, "bold"),
            pady=10,
            relief="raised",
            cursor="hand2",
        )
        self.btn_launch.pack(fill="x", pady=(0, 8))

        # Update Session Summary Box
        self._update_info()

    def _on_source_mode_changed(self):
        mode = self.source_mode.get()
        if mode == "webcam":
            self.selected_sources = ["0"]
            self.rtsp_frame.pack_forget()
            self.file_btn_frame.pack(fill="x", pady=5)
            self._refresh_source_listbox(["Cam 1: Laptop Front Webcam (Index 0)"])
        elif mode == "files":
            self.rtsp_frame.pack_forget()
            self.file_btn_frame.pack(fill="x", pady=5)
            import glob
            found = sorted(glob.glob("data/videos/*.mp4") + glob.glob("data/videos/*.avi") + glob.glob("data/videos/*.mkv"))
            if found:
                self.selected_sources = list(found)
                items = [f"Cam {i+1}: {Path(f).name}" for i, f in enumerate(found)]
                self._refresh_source_listbox(items)
            else:
                self._on_browse_videos()
        elif mode == "rtsp":
            self.file_btn_frame.pack_forget()
            self.rtsp_frame.pack(fill="x", pady=5)
            self._refresh_source_listbox(["RTSP Stream (Configure URL above)"])
        self._update_info()

    def _on_browse_videos(self):
        files = filedialog.askopenfilenames(
            title="Select CCTV Video Files for Person Re-ID",
            filetypes=[("Video Files", "*.mp4;*.avi;*.mkv;*.mov;*.wmv"), ("All Files", "*.*")],
        )
        if files:
            self.selected_sources = list(files)
            items = [f"Cam {i+1}: {Path(f).name}" for i, f in enumerate(files)]
            self._refresh_source_listbox(items)
            self._update_info()

    def _on_reset_sources(self):
        self.source_mode.set("webcam")
        self.selected_sources = ["0"]
        self._refresh_source_listbox(["Cam 1: Laptop Front Webcam (Index 0)"])
        self._update_info()

    def _refresh_source_listbox(self, items: List[str]):
        self.source_listbox.delete(0, "end")
        for item in items:
            self.source_listbox.insert("end", item)

    def _on_load_zones(self):
        path = filedialog.askopenfilename(
            title="Load Handoff Zones JSON",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
        )
        if path:
            self.handoff_zone_file = path
            zones = load_zones(path)
            total = sum(len(v) for v in zones.values())
            self.lbl_zone_status.config(
                text=f"✓ Loaded {total} zone(s) from {Path(path).name}",
                fg="#38bdf8",
            )
            self._update_info()

    def _on_draw_zone(self):
        src = self.selected_sources[0] if self.selected_sources else "0"
        messagebox.showinfo(
            "Zone Editor Instructions",
            "Zone editor will open on the current feed.\n"
            "1. Drag mouse rectangle over the transition/exit doorway.\n"
            "2. Press [S] to save the zone.\n"
            "3. Press [Q] to close."
        )
        out_path = "data/cctv_zones.json"
        run_zone_editor(source=src, camera_id="cam1", output_path=out_path)
        if Path(out_path).exists():
            self.handoff_zone_file = out_path
            self.lbl_zone_status.config(text=f"✓ Saved & Loaded zone to {out_path}", fg="#10b981")
            self._update_info()

    def _update_info(self):
        src_text = "\n".join(f"   * {s}" for s in self.selected_sources)
        text = f"""======================================================
  SAFEPASS AUTONOMOUS RE-ID ENGINE
======================================================
1. IDENTIFICATION MODE:
   * Target Attribution: Autonomous Person IDs
   * Manual Face Input:  DISABLED (Automated Detection)
   * Manual Cloth Input: DISABLED (Automated Extraction)
   * ID Format:          {self.prefix_var.get()}0001, {self.prefix_var.get()}0002...

2. ACTIVE CAMERA FEEDS:
{src_text}

3. ALGORITHM STACK (ReID-Framework):
   * Tracking Layer:     Per-Camera Centroid SimpleTracker
   * Re-ID Feature:      {self.model_var.get()}
   * Similarity Thresh:  {self.sim_thresh_var.get():.2f}
   * Global Persistence: GlobalIDManager Multi-Camera Fusion
   * Topology Handoff:   {'ACTIVE' if self.handoff_zone_file else 'STANDARD SIMILARITY'}

4. CONTROLS IN LIVE FEED:
   * [SPACE]  Pause / Resume Feed
   * [S]      Save Annotated CCTV Frame Snapshot
   * [Q]      Quit and return to Studio
======================================================"""

        self.info_box.config(state="normal")
        self.info_box.delete("1.0", "end")
        self.info_box.insert("1.0", text)
        self.info_box.config(state="disabled")

    def _on_launch_reid(self):
        if self.source_mode.get() == "rtsp":
            rtsp_url = self.rtsp_var.get().strip()
            if not rtsp_url or rtsp_url == "rtsp://":
                messagebox.showerror("Invalid URL", "Please enter a valid RTSP stream URL.")
                return
            sources = [rtsp_url]
        else:
            sources = self.selected_sources

        use_osnet = "OSNet" in self.model_var.get()
        sim_thresh = self.sim_thresh_var.get()
        prefix = self.prefix_var.get().strip() or "Person_"

        self.root.withdraw()

        try:
            cam_ids = [f"cam{i+1}" for i in range(len(sources))]
            engine = MultiCameraCCTVReIDEngine(
                sources=sources,
                camera_ids=cam_ids,
                zone_file=self.handoff_zone_file,
                similarity_threshold=sim_thresh,
                handoff_threshold=0.48,
                use_osnet=use_osnet,
            )
            engine.global_ids.id_prefix = prefix
            engine.run(show=True)
        except Exception as e:
            messagebox.showerror("Execution Error", f"Failed to run Re-ID stream: {e}")
        finally:
            self.root.deiconify()
            self._update_info()


def main():
    root = tk.Tk()
    app = SafePassReIDStudio(root)
    root.mainloop()


if __name__ == "__main__":
    main()
