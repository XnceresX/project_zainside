"""
safepass_studio.py
==================
SAFEPASS Multimodal Person Intake & Live Identification Studio (Tracks P1 & P2).

Provides an interactive GUI to:
  1. Input Person Name and Height.
  2. Input Multiple Face Photos (Frontal, Left, Right, etc.) from files or live webcam capture
     to capture full 3D facial structure and features.
  3. Input Clothing & Appearance details (Upper & Lower garment colours, descriptions, or body photos).
  4. Forget/Reset previous test identities (Zain, gracious, etc.) so only the current input is used.
  5. Review the complete identity & appearance profile.
  6. Launch the live webcam feed and identify the person using both Face (P1) and Appearance (P2) models.
"""

from __future__ import annotations

import os
import sys
import time
import threading
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# SAFEPASS AI Modules
from face.detector import FaceDetector, DetectedFace
from face.extractor import FaceFeatureExtractor
from face.multiview import MultiViewEnrollmentEngine, MultiViewGalleryItem
from appearance.extractor import AppearanceExtractor
from appearance.color import ClothingColorExtractor, ColorFeatures
from appearance.quality import BodyQualityEstimator
from fusion.engine import FusionEngine
from face.webcam_live import (
    IdentityGallery,
    MultimodalGalleryItem,
    WebcamFaceIdentifier,
    extract_body_crop_from_face,
    draw_hud_header,
    draw_hud_footer,
    draw_corner_rect,
    draw_body_rect,
    draw_landmarks,
)


class MultimodalIntakeStudio:
    """
    Desktop GUI application for full multimodal profile intake, review, and live webcam verification.
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SAFEPASS - Multimodal Intake & Live Identification Studio")
        self.root.geometry("1100x820")
        self.root.minsize(980, 750)
        self.root.configure(bg="#0b0f19")

        # SAFEPASS Engines
        self.detector = FaceDetector(detection_confidence=0.55)
        self.face_extractor = FaceFeatureExtractor()
        self.enrollment_engine = MultiViewEnrollmentEngine(self.detector, self.face_extractor)
        self.appearance_extractor = AppearanceExtractor()
        self.color_extractor = ClothingColorExtractor()
        self.body_quality_estimator = BodyQualityEstimator()
        self.fusion_engine = FusionEngine(mode="dynamic")
        self.gallery = IdentityGallery()

        # State storage
        self.face_images: List[np.ndarray] = []           # Multiple loaded BGR face images
        self.face_crops: List[np.ndarray] = []            # Cropped faces
        self.face_thumbnails: List[ImageTk.PhotoImage] = [] # Tkinter display references
        self.body_image: Optional[np.ndarray] = None     # BGR body/clothes image
        self.body_thumbnail: Optional[ImageTk.PhotoImage] = None

        self._build_ui()

    def _build_ui(self):
        # Configure Styles
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", background="#0b0f19", foreground="#f1f5f9", font=("Segoe UI", 10))
        style.configure("TFrame", background="#0b0f19")
        style.configure("Card.TFrame", background="#111827", relief="ridge")
        style.configure("TCheckbutton", background="#111827", foreground="#94a3b8", font=("Segoe UI", 9))
        style.configure("TCombobox", fieldbackground="#1f2937", background="#374151", foreground="#ffffff")

        # Main Scrollable Canvas / Container
        header_frame = tk.Frame(self.root, bg="#0b0f19", pady=10)
        header_frame.pack(fill="x", padx=20)

        title_lbl = tk.Label(
            header_frame,
            text="SAFEPASS MULTIMODAL INTAKE & IDENTIFICATION STUDIO",
            font=("Segoe UI", 15, "bold"),
            fg="#00e5ff",
            bg="#0b0f19",
        )
        title_lbl.pack(anchor="w")

        subtitle_lbl = tk.Label(
            header_frame,
            text="Tracks P1 (Face Biometrics) & P2 (Appearance & Clothing Re-ID) · Multi-Photo Facial Structure Enrollment",
            font=("Segoe UI", 9),
            fg="#94a3b8",
            bg="#0b0f19",
        )
        subtitle_lbl.pack(anchor="w")

        # Two-Column Layout Container
        main_container = tk.Frame(self.root, bg="#0b0f19")
        main_container.pack(fill="both", expand=True, padx=20, pady=5)

        # Left Column: Inputs & Photo Selectors (width 58%)
        left_col = tk.Frame(main_container, bg="#0b0f19")
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Right Column: Review Card & Verification Controls (width 42%)
        right_col = tk.Frame(main_container, bg="#0b0f19", width=420)
        right_col.pack(side="right", fill="both", padx=(10, 0))

        # ---------------------------------------------------------------
        # LEFT: 1. Identity & Metadata Card
        # ---------------------------------------------------------------
        card1 = tk.LabelFrame(
            left_col,
            text=" 1. Person Identity & Physical Attributes ",
            bg="#111827",
            fg="#00e5ff",
            font=("Segoe UI", 10, "bold"),
            padx=15,
            pady=10,
            bd=1,
            relief="solid",
        )
        card1.pack(fill="x", pady=(0, 10))

        meta_grid = tk.Frame(card1, bg="#111827")
        meta_grid.pack(fill="x")

        # Name
        tk.Label(meta_grid, text="Full Name / Person ID *:", bg="#111827", fg="#cbd5e1", font=("Segoe UI", 10)).grid(row=0, column=0, sticky="w", pady=4)
        self.name_var = tk.StringVar(value="")
        self.name_var.trace_add("write", lambda *args: self._update_review())
        self.name_entry = tk.Entry(meta_grid, textvariable=self.name_var, bg="#1f2937", fg="#ffffff", font=("Segoe UI", 10), insertbackground="white", width=24)
        self.name_entry.grid(row=0, column=1, sticky="w", padx=10, pady=4)

        # Height
        tk.Label(meta_grid, text="Height (e.g. 175 cm / 5'9\"):", bg="#111827", fg="#cbd5e1", font=("Segoe UI", 10)).grid(row=0, column=2, sticky="w", pady=4, padx=(15, 0))
        self.height_var = tk.StringVar(value="175 cm")
        self.height_var.trace_add("write", lambda *args: self._update_review())
        self.height_entry = tk.Entry(meta_grid, textvariable=self.height_var, bg="#1f2937", fg="#38bdf8", font=("Segoe UI", 10), insertbackground="white", width=14)
        self.height_entry.grid(row=0, column=3, sticky="w", padx=10, pady=4)

        # Clear Previous Gallery Option
        self.reset_old_var = tk.BooleanVar(value=True)
        chk_reset = tk.Checkbutton(
            card1,
            text="Forget/Clear previous test identities (Zain, gracious, etc.) before saving",
            variable=self.reset_old_var,
            bg="#111827",
            fg="#f59e0b",
            activebackground="#111827",
            activeforeground="#fbbf24",
            selectcolor="#1f2937",
            font=("Segoe UI", 9, "bold"),
        )
        chk_reset.pack(anchor="w", pady=(8, 2))

        # ---------------------------------------------------------------
        # LEFT: 2. Facial Structure & Multiple Face Photos Card (Track P1)
        # ---------------------------------------------------------------
        card2 = tk.LabelFrame(
            left_col,
            text=" 2. Facial Structure & Multi-Angle Photos (Track P1 ArcFace) ",
            bg="#111827",
            fg="#00e5ff",
            font=("Segoe UI", 10, "bold"),
            padx=15,
            pady=10,
            bd=1,
            relief="solid",
        )
        card2.pack(fill="x", pady=(0, 10))

        tk.Label(
            card2,
            text="Add multiple photos (Frontal, Left side, Right side, Tilt) to build an accurate 3D facial feature structure:",
            bg="#111827",
            fg="#94a3b8",
            font=("Segoe UI", 9),
            wraplength=520,
            justify="left",
        ).pack(anchor="w", pady=(0, 8))

        btn_bar = tk.Frame(card2, bg="#111827")
        btn_bar.pack(anchor="w", fill="x", pady=(0, 8))

        btn_browse_face = tk.Button(
            btn_bar,
            text="📁 Browse Face Photos...",
            command=self._on_browse_face_photos,
            bg="#1f2937",
            fg="#e2e8f0",
            font=("Segoe UI", 9, "bold"),
            padx=12,
            pady=5,
            relief="groove",
        )
        btn_browse_face.pack(side="left", padx=(0, 10))

        btn_cam_face = tk.Button(
            btn_bar,
            text="📷 Capture Face Angles via Webcam",
            command=self._on_capture_face_webcam,
            bg="#0284c7",
            fg="#ffffff",
            font=("Segoe UI", 9, "bold"),
            padx=12,
            pady=5,
            relief="groove",
        )
        btn_cam_face.pack(side="left", padx=(0, 10))

        btn_clear_photos = tk.Button(
            btn_bar,
            text="Clear Photos",
            command=self._clear_face_photos,
            bg="#374151",
            fg="#94a3b8",
            font=("Segoe UI", 8),
            padx=8,
            pady=4,
            relief="flat",
        )
        btn_clear_photos.pack(side="left")

        # Scrollable Thumbnail Gallery for loaded face photos
        self.thumb_container = tk.Frame(card2, bg="#0d131f", height=95, bd=1, relief="sunken")
        self.thumb_container.pack(fill="x", pady=5)
        self.thumb_container.pack_propagate(False)

        self.thumb_lbl_info = tk.Label(
            self.thumb_container,
            text="No face photos added yet. Click 'Browse Face Photos...' or 'Capture Face Angles via Webcam'.",
            bg="#0d131f",
            fg="#64748b",
            font=("Segoe UI", 9, "italic"),
        )
        self.thumb_lbl_info.pack(expand=True)

        # ---------------------------------------------------------------
        # LEFT: 3. Clothing Appearance & Body Re-ID Card (Track P2)
        # ---------------------------------------------------------------
        card3 = tk.LabelFrame(
            left_col,
            text=" 3. Clothing & Body Appearance (Track P2 OSNet + HSV Color) ",
            bg="#111827",
            fg="#00e5ff",
            font=("Segoe UI", 10, "bold"),
            padx=15,
            pady=10,
            bd=1,
            relief="solid",
        )
        card3.pack(fill="x")

        cloth_grid = tk.Frame(card3, bg="#111827")
        cloth_grid.pack(fill="x", pady=(0, 8))

        # Upper Clothing
        tk.Label(cloth_grid, text="Upper Garment (Shirt/Top) Colour *:", bg="#111827", fg="#cbd5e1", font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w", pady=4)
        self.upper_color_var = tk.StringVar(value="black")
        self.upper_color_var.trace_add("write", lambda *args: self._update_review())
        self.combo_upper = ttk.Combobox(
            cloth_grid,
            textvariable=self.upper_color_var,
            values=["black", "white", "grey", "blue", "navy", "red", "pink", "green", "yellow", "orange", "purple", "brown"],
            width=14,
            font=("Segoe UI", 9),
        )
        self.combo_upper.grid(row=0, column=1, sticky="w", padx=10, pady=4)

        tk.Label(cloth_grid, text="Upper Description:", bg="#111827", fg="#94a3b8", font=("Segoe UI", 9)).grid(row=0, column=2, sticky="w", pady=4, padx=(10, 0))
        self.upper_desc_var = tk.StringVar(value="Black cotton shirt")
        self.upper_desc_var.trace_add("write", lambda *args: self._update_review())
        tk.Entry(cloth_grid, textvariable=self.upper_desc_var, bg="#1f2937", fg="#ffffff", font=("Segoe UI", 9), width=20, insertbackground="white").grid(row=0, column=3, sticky="w", padx=5, pady=4)

        # Lower Clothing
        tk.Label(cloth_grid, text="Lower Garment (Pants/Jeans) Colour *:", bg="#111827", fg="#cbd5e1", font=("Segoe UI", 9)).grid(row=1, column=0, sticky="w", pady=4)
        self.lower_color_var = tk.StringVar(value="blue")
        self.lower_color_var.trace_add("write", lambda *args: self._update_review())
        self.combo_lower = ttk.Combobox(
            cloth_grid,
            textvariable=self.lower_color_var,
            values=["blue", "black", "grey", "white", "brown", "green", "navy"],
            width=14,
            font=("Segoe UI", 9),
        )
        self.combo_lower.grid(row=1, column=1, sticky="w", padx=10, pady=4)

        tk.Label(cloth_grid, text="Lower Description:", bg="#111827", fg="#94a3b8", font=("Segoe UI", 9)).grid(row=1, column=2, sticky="w", pady=4, padx=(10, 0))
        self.lower_desc_var = tk.StringVar(value="Blue denim jeans")
        self.lower_desc_var.trace_add("write", lambda *args: self._update_review())
        tk.Entry(cloth_grid, textvariable=self.lower_desc_var, bg="#1f2937", fg="#ffffff", font=("Segoe UI", 9), width=20, insertbackground="white").grid(row=1, column=3, sticky="w", padx=5, pady=4)

        # Clothing / Body Photo buttons
        body_btn_frame = tk.Frame(card3, bg="#111827")
        body_btn_frame.pack(anchor="w", fill="x", pady=4)

        btn_browse_body = tk.Button(
            body_btn_frame,
            text="📁 Add Clothing/Body Photo...",
            command=self._on_browse_body_photo,
            bg="#1f2937",
            fg="#e2e8f0",
            font=("Segoe UI", 9),
            padx=10,
            pady=4,
            relief="groove",
        )
        btn_browse_body.pack(side="left", padx=(0, 10))

        btn_cam_body = tk.Button(
            body_btn_frame,
            text="📷 Auto-Detect Clothing from Webcam",
            command=self._on_capture_body_webcam,
            bg="#0284c7",
            fg="#ffffff",
            font=("Segoe UI", 9),
            padx=10,
            pady=4,
            relief="groove",
        )
        btn_cam_body.pack(side="left")

        self.lbl_body_status = tk.Label(
            card3,
            text="Tip: Auto-detection will scan your shirt & trousers live using HSV color histograms.",
            bg="#111827",
            fg="#64748b",
            font=("Segoe UI", 8, "italic"),
        )
        self.lbl_body_status.pack(anchor="w", pady=(4, 0))

        # ---------------------------------------------------------------
        # RIGHT: 4. Profile Review & Live Identification Card
        # ---------------------------------------------------------------
        card_review = tk.LabelFrame(
            right_col,
            text=" 4. Enrolled Profile Review & Live Camera Verification ",
            bg="#111827",
            fg="#00e5ff",
            font=("Segoe UI", 10, "bold"),
            padx=15,
            pady=12,
            bd=1,
            relief="solid",
        )
        card_review.pack(fill="both", expand=True)

        # Profile Summary Box
        self.review_box = tk.Text(
            card_review,
            bg="#0d131f",
            fg="#f8fafc",
            font=("Consolas", 9),
            height=16,
            bd=1,
            relief="sunken",
            padx=10,
            pady=10,
        )
        self.review_box.pack(fill="both", expand=True, pady=(0, 15))
        self.review_box.config(state="disabled")

        # Big Launch Live Identification Button
        self.btn_launch = tk.Button(
            card_review,
            text="🚀 CONFIRM PROFILE & START LIVE IDENTIFICATION",
            command=self._on_confirm_and_launch,
            bg="#10b981",
            fg="#ffffff",
            font=("Segoe UI", 11, "bold"),
            pady=10,
            relief="raised",
            cursor="hand2",
        )
        self.btn_launch.pack(fill="x", pady=(0, 8))

        # Reset / Clear All Button
        btn_wipe_all = tk.Button(
            card_review,
            text="🧹 Clear Entire Identity Database",
            command=self._on_wipe_database,
            bg="#374151",
            fg="#ef4444",
            font=("Segoe UI", 9),
            pady=4,
            relief="flat",
        )
        btn_wipe_all.pack(fill="x")

        # Initial review update
        self._update_review()

    # -------------------------------------------------------------------
    # Review Updating Logic
    # -------------------------------------------------------------------

    def _update_review(self):
        name = self.name_var.get().strip() or "[Awaiting Name Input]"
        height = self.height_var.get().strip() or "Not Specified"
        upper = self.upper_color_var.get().strip().upper()
        lower = self.lower_color_var.get().strip().upper()
        upper_desc = self.upper_desc_var.get().strip()
        lower_desc = self.lower_desc_var.get().strip()
        num_photos = len(self.face_images)
        has_body = "YES (Custom Crop/Detected)" if self.body_image is not None else "AUTO-DERIVED FROM FACE LOCATION"

        text = f"""======================================================
  SAFEPASS MULTIMODAL PROFILE REVIEW
======================================================
1. BIOMETRIC IDENTITY:
   * Target Name:     {name}
   * Estimated Height: {height}
   * Face Photos:      {num_photos} photo(s) loaded
     - ArcFace 512-d embeddings
     - 3D Multi-View Pose Synthesis: ACTIVE
     - Complete Facial Structure: {'READY' if num_photos > 0 else 'PENDING PHOTOS'}

2. CLOTHING & APPEARANCE PROFILE:
   * Upper Garment:    {upper} ({upper_desc})
   * Lower Garment:    {lower} ({lower_desc})
   * Body Crop:        {has_body}
   * OSNet Re-ID:      512-d Body Embedding ACTIVE
   * HSV Colour Model: 144-d Regional Histograms ACTIVE

3. MULTIMODAL FUSION ENGINE:
   * Fusion Strategy:  Learned Quality-Adaptive Dynamic
   * Biometric Weight: w_face (adapts to blur & pose)
   * Appearance Weight: w_app (adapts to body visibility)
   * Garment Weight:    w_color (adapts to clothing match)

4. DATABASE STATE:
   * Reset Old Data:   {'YES (Zain, gracious, etc. will be cleared)' if self.reset_old_var.get() else 'NO (merge with existing)'}
======================================================"""

        self.review_box.config(state="normal")
        self.review_box.delete("1.0", "end")
        self.review_box.insert("1.0", text)
        self.review_box.config(state="disabled")

    # -------------------------------------------------------------------
    # Photo Management & Thumbnail Preview
    # -------------------------------------------------------------------

    def _on_browse_face_photos(self):
        files = filedialog.askopenfilenames(
            title="Select Face Photos (Different Angles Encouraged)",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.webp"), ("All Files", "*.*")],
        )
        if not files:
            return

        added = 0
        for fpath in files:
            img = cv2.imread(fpath)
            if img is not None:
                self.face_images.append(img)
                # Detect face crop for thumbnail
                det = self.detector.detect_single(img)
                crop = det.face_crop if det is not None else cv2.resize(img, (112, 112))
                self.face_crops.append(crop)
                added += 1

        if added > 0:
            self._render_face_thumbnails()
            self._update_review()
            messagebox.showinfo("Photos Added", f"Loaded {added} face photo(s) successfully!\nDetected facial structures from provided images.")

    def _on_browse_body_photo(self):
        fpath = filedialog.askopenfilename(
            title="Select Clothing / Body Photo",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.webp"), ("All Files", "*.*")],
        )
        if not fpath:
            return
        img = cv2.imread(fpath)
        if img is not None:
            self.body_image = img
            # Auto-detect colors
            try:
                feats = self.color_extractor.extract(img)
                if feats.upper_label != "unknown":
                    self.upper_color_var.set(feats.upper_label)
                if feats.lower_label != "unknown":
                    self.lower_color_var.set(feats.lower_label)
                self.lbl_body_status.config(
                    text=f"✓ Detected: Upper [{feats.upper_label.upper()}], Lower [{feats.lower_label.upper()}] from selected photo.",
                    fg="#38bdf8",
                )
            except Exception:
                pass
            self._update_review()

    def _clear_face_photos(self):
        self.face_images.clear()
        self.face_crops.clear()
        self.face_thumbnails.clear()
        for widget in self.thumb_container.winfo_children():
            widget.destroy()
        self.thumb_lbl_info = tk.Label(
            self.thumb_container,
            text="No face photos added yet. Click 'Browse Face Photos...' or 'Capture Face Angles via Webcam'.",
            bg="#0d131f",
            fg="#64748b",
            font=("Segoe UI", 9, "italic"),
        )
        self.thumb_lbl_info.pack(expand=True)
        self._update_review()

    def _render_face_thumbnails(self):
        for widget in self.thumb_container.winfo_children():
            widget.destroy()
        self.face_thumbnails.clear()

        canvas = tk.Canvas(self.thumb_container, bg="#0d131f", highlightthickness=0)
        h_scroll = tk.Scrollbar(self.thumb_container, orient="horizontal", command=canvas.xview)
        scrollable_frame = tk.Frame(canvas, bg="#0d131f")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(xscrollcommand=h_scroll.set)

        canvas.pack(side="top", fill="both", expand=True)
        h_scroll.pack(side="bottom", fill="x")

        for idx, crop in enumerate(self.face_crops):
            thumb_bgr = cv2.resize(crop, (65, 65))
            thumb_rgb = cv2.cvtColor(thumb_bgr, cv2.COLOR_BGR2RGB)
            pi = ImageTk.PhotoImage(image=Image.fromarray(thumb_rgb))
            self.face_thumbnails.append(pi)

            thumb_card = tk.Frame(scrollable_frame, bg="#111827", bd=1, relief="ridge", padx=3, pady=3)
            thumb_card.pack(side="left", padx=4, pady=2)

            tk.Label(thumb_card, image=pi, bg="#111827").pack()
            tk.Label(thumb_card, text=f"Pose #{idx+1}", font=("Segoe UI", 7), fg="#94a3b8", bg="#111827").pack()

    # -------------------------------------------------------------------
    # Interactive Webcam Captures
    # -------------------------------------------------------------------

    def _on_capture_face_webcam(self):
        """
        Interactive multi-angle face capture tool using the laptop webcam.
        Allows taking snapshots of: Frontal, Left Yaw, Right Yaw, Pitch Up/Down.
        """
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open camera 0. Ensure no other app is using the webcam.")
            return

        win_title = "SAFEPASS Face Angle Capture (SPACE = Snap Photo, ESC = Done)"
        cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_title, 800, 520)

        captured_count = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame_mirrored = cv2.flip(frame, 1)
            display = frame_mirrored.copy()
            faces = self.detector.detect(frame_mirrored)

            h, w = display.shape[:2]

            # Top instructions bar
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (w, 54), (15, 18, 25), -1)
            cv2.addWeighted(overlay, 0.85, display, 0.15, 0, display)
            cv2.line(display, (0, 54), (w, 54), (0, 200, 255), 1, cv2.LINE_AA)

            msg = f"CAPTURED: {captured_count} ANGLES  |  [SPACE]: Snap Current Angle  |  [ESC]: Finish"
            cv2.putText(display, msg, (15, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 2, cv2.LINE_AA)

            for face in faces:
                draw_corner_rect(display, face.bbox, (0, 255, 200), thickness=2)
                if face.landmarks is not None:
                    draw_landmarks(display, face.landmarks)
                yaw = face.pose.yaw if face.pose else 0.0
                cv2.putText(
                    display,
                    f"Pose Yaw: {yaw:+2.0f} deg (Turn head slightly for full 3D capture)",
                    (face.bbox[0], max(face.bbox[1] - 8, 70)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (240, 240, 240),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow(win_title, display)
            key = cv2.waitKey(20) & 0xFF

            if key == 32:  # SPACEBAR = Capture
                if faces:
                    self.face_images.append(frame_mirrored.copy())
                    self.face_crops.append(faces[0].face_crop.copy())
                    captured_count += 1
                    # Flash feedback
                    cv2.rectangle(display, (0, 0), (w, h), (255, 255, 255), -1)
                    cv2.imshow(win_title, display)
                    cv2.waitKey(80)
            elif key in (27, ord('q'), ord('Q')):  # ESC = Done
                break

        cap.release()
        cv2.destroyWindow(win_title)

        if captured_count > 0:
            self._render_face_thumbnails()
            self._update_review()
            messagebox.showinfo("Face Angles Captured", f"Successfully captured {captured_count} facial structure photos!")

    def _on_capture_body_webcam(self):
        """
        Captures a frame from webcam, extracts clothing / torso crop,
        and auto-detects upper & lower garment colors using HSV histograms.
        """
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open camera 0.")
            return

        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            messagebox.showerror("Camera Error", "Failed to capture webcam frame.")
            return

        frame = cv2.flip(frame, 1)
        faces = self.detector.detect(frame)
        if not faces:
            # Fallback: take central region
            h, w = frame.shape[:2]
            body_crop = frame[int(h*0.2):int(h*0.8), int(w*0.25):int(w*0.75)]
        else:
            body_crop, _ = extract_body_crop_from_face(frame, faces[0].bbox)

        self.body_image = body_crop
        try:
            feats = self.color_extractor.extract(body_crop)
            if feats.upper_label != "unknown":
                self.upper_color_var.set(feats.upper_label)
            if feats.lower_label != "unknown":
                self.lower_color_var.set(feats.lower_label)
            self.lbl_body_status.config(
                text=f"✓ Auto-Detected from Webcam: Upper [{feats.upper_label.upper()}], Lower [{feats.lower_label.upper()}]",
                fg="#10b981",
            )
        except Exception as e:
            self.lbl_body_status.config(text=f"Detection error: {e}", fg="#ef4444")

        self._update_review()

    def _on_wipe_database(self):
        confirm = messagebox.askyesno(
            "Clear Database Confirmation",
            "Are you sure you want to completely clear all registered identities?\nThis will remove previous records (Zain, gracious, etc.).",
        )
        if confirm:
            self.gallery.clear()
            self._update_review()
            messagebox.showinfo("Database Cleared", "All identity database entries cleared successfully.")

    # -------------------------------------------------------------------
    # Final Confirm & Launch Live Webcam Identification
    # -------------------------------------------------------------------

    def _on_confirm_and_launch(self):
        name = self.name_var.get().strip()
        if not name:
            messagebox.showerror("Missing Information", "Please enter a Person Name / Identity ID.")
            self.name_entry.focus_set()
            return

        if not self.face_images:
            messagebox.showerror("Missing Photos", "Please add at least one face photo (or capture from webcam) to build the facial structure model.")
            return

        height = self.height_var.get().strip() or "175 cm"
        upper_col = self.upper_color_var.get().strip().lower() or "unknown"
        lower_col = self.lower_color_var.get().strip().lower() or "unknown"
        upper_desc = self.upper_desc_var.get().strip()
        lower_desc = self.lower_desc_var.get().strip()
        full_desc = f"{upper_desc}, {lower_desc}".strip(", ")

        # 1. Reset previous gallery if requested
        if self.reset_old_var.get():
            self.gallery.clear()
            print(f"[Studio] Cleared previous gallery records. Registering fresh identity '{name}'.")

        # 2. Extract Body Crop if not manually provided
        body_crop_to_use = self.body_image
        if body_crop_to_use is None:
            # Use first face photo to extract torso crop
            first_face_img = self.face_images[0]
            det = self.detector.detect_single(first_face_img)
            bbox = det.bbox if det is not None else (0, 0, first_face_img.shape[1], first_face_img.shape[0])
            body_crop_to_use, _ = extract_body_crop_from_face(first_face_img, bbox)

        # 3. Enroll Multi-Photo Facial Structure + Appearance
        try:
            # First photo enrollment
            enrolled_item = self.gallery.enroll_person(
                person_name=name,
                face_img=self.face_crops[0],
                enrollment_engine=self.enrollment_engine,
                appearance_extractor=self.appearance_extractor,
                color_extractor=self.color_extractor,
                body_crop=body_crop_to_use,
                upper_color=upper_col,
                lower_color=lower_col,
                appearance_desc=f"{full_desc} | Height: {height}",
            )
            enrolled_item.height = height

            # Accumulate all other provided face angle photos into the gallery
            for additional_crop in self.face_crops[1:]:
                sub_item = self.enrollment_engine.enroll(name, additional_crop)
                idx_offset = len(enrolled_item.face_item.embeddings)
                for k, v in sub_item.embeddings.items():
                    enrolled_item.face_item.embeddings[f"{k}_{idx_offset}"] = v
                for k, v in sub_item.aligned_crops.items():
                    enrolled_item.face_item.aligned_crops[f"{k}_{idx_offset}"] = v

            self.gallery.save()
            print(f"[Studio] Successfully enrolled '{name}' with {len(enrolled_item.face_item.embeddings)} multi-angle facial embeddings!")

        except Exception as e:
            messagebox.showerror("Enrollment Error", f"Failed to enroll identity: {e}")
            return

        # 4. Hide studio window and start live webcam surveillance
        self.root.withdraw()

        # Launch Live Webcam Engine
        identifier = WebcamFaceIdentifier(
            cam_index=0,
            threshold=0.70,
            mirror=True,
            show_landmarks=True,
        )
        identifier.gallery = self.gallery
        identifier.set_alert(f"[IDENTIFYING] Monitoring for '{name}' ({height}) via Face + Appearance!")

        try:
            identifier.run_gui()
        finally:
            self.root.deiconify()
            self._update_review()


def main():
    root = tk.Tk()
    app = MultimodalIntakeStudio(root)
    root.mainloop()


if __name__ == "__main__":
    main()
