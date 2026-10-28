/**
 * frontend/app.js
 * ================
 * SAFEPASS CCTV AI Command Center - Client Application.
 * Full-featured surveillance playback, dynamic camera management,
 * frame-accurate seeking, and real-time Re-ID visual HUD.
 */

class CameraPlayer {
    constructor(meta, referenceTarget, onUpdateCallback) {
        this.id = meta.id;
        this.name = meta.name;
        this.fps = meta.fps || 20;
        this.totalFrames = meta.total_frames || 100;
        this.summary = meta.summary || {};
        this.referenceTarget = referenceTarget;
        this.onUpdate = onUpdateCallback;

        this.currentFrame = 1;
        this.isPlaying = false;
        this.playbackRate = 1.0;
        this.analysisData = {};
        this.frameCache = new Map();
        this.animTimer = null;
        this.lastFrameTime = 0;

        this.element = null;
        this.canvas = null;
        this.ctx = null;
        this.slider = null;
        this.frameCounter = null;
        this.badge = null;
        this.playBtn = null;
        this.pauseOverlay = null;

        this.initDOM();
        this.loadAnalysis();
    }

    initDOM() {
        const card = document.createElement("div");
        card.className = "camera-card";
        card.id = `card-${this.id}`;

        // Header
        const header = document.createElement("div");
        header.className = "card-header";
        header.innerHTML = `
            <div class="card-title-group">
                <span class="cam-icon">📹</span>
                <span class="cam-name" id="name-${this.id}" title="Click to Rename">${this.name}</span>
            </div>
            <div class="card-badge badge-scan" id="badge-${this.id}">INITIALIZING</div>
            <div class="card-actions">
                <button class="btn-icon-action" id="fullscreen-${this.id}" title="Fullscreen View">⛶</button>
                <button class="btn-icon-action btn-del" id="delete-${this.id}" title="Delete Camera">✕</button>
            </div>
        `;

        // Viewport
        const viewport = document.createElement("div");
        viewport.className = "card-viewport";
        viewport.id = `viewport-${this.id}`;

        const canvas = document.createElement("canvas");
        canvas.className = "card-canvas";
        canvas.width = 960;
        canvas.height = 640;
        this.canvas = canvas;
        this.ctx = canvas.getContext("2d");

        // PIP for Reference Target
        const pip = document.createElement("div");
        pip.className = "viewport-pip";
        pip.id = `pip-${this.id}`;
        pip.innerHTML = `
            <img class="pip-img" id="pip-img-${this.id}" src="${this.referenceTarget ? this.referenceTarget.crop_b64 : ''}" alt="Ref">
            <div class="pip-label">REF TARGET</div>
        `;
        if (!this.referenceTarget || !this.referenceTarget.crop_b64) {
            pip.style.display = "none";
        }

        // Pause Overlay
        const overlay = document.createElement("div");
        overlay.className = "card-overlay-pause";
        overlay.id = `overlay-${this.id}`;
        overlay.innerHTML = `<div class="pause-icon-big">❚❚</div>`;
        this.pauseOverlay = overlay;

        viewport.appendChild(canvas);
        viewport.appendChild(pip);
        viewport.appendChild(overlay);

        // Controls
        const controls = document.createElement("div");
        controls.className = "card-controls";
        controls.innerHTML = `
            <div class="seek-container">
                <input type="range" class="seek-slider" id="seek-${this.id}" min="1" max="${this.totalFrames}" value="1">
                <div class="frame-counter" id="counter-${this.id}">F 1 / ${this.totalFrames} (00:00.0)</div>
            </div>
            <div class="card-btn-row">
                <div class="btn-group-left">
                    <button class="btn-ctrl btn-ctrl-play" id="play-${this.id}" title="Play/Pause">▶</button>
                    <button class="btn-ctrl btn-ctrl-skip" id="skip-back-5-${this.id}" title="Skip -5 Frames">⏪ -5f</button>
                    <button class="btn-ctrl btn-ctrl-skip" id="step-back-${this.id}" title="Step -1 Frame">◀ -1f</button>
                    <button class="btn-ctrl btn-ctrl-skip" id="step-fwd-${this.id}" title="Step +1 Frame">+1f ▶</button>
                    <button class="btn-ctrl btn-ctrl-skip" id="skip-fwd-5-${this.id}" title="Skip +5 Frames">+5f ⏩</button>
                </div>
                <div class="meta-info-badge" id="meta-${this.id}">
                    ${this.fps.toFixed(0)} FPS | 960x640
                </div>
            </div>
        `;

        card.appendChild(header);
        card.appendChild(viewport);
        card.appendChild(controls);

        this.element = card;
        this.badge = header.querySelector(`#badge-${this.id}`);
        this.slider = controls.querySelector(`#seek-${this.id}`);
        this.frameCounter = controls.querySelector(`#counter-${this.id}`);
        this.playBtn = controls.querySelector(`#play-${this.id}`);

        this.bindEvents(header, viewport, controls);
    }

    bindEvents(header, viewport, controls) {
        // Toggle play/pause on canvas click
        viewport.addEventListener("click", () => this.togglePlay());

        // Play/Pause button
        this.playBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            this.togglePlay();
        });

        // Step buttons
        controls.querySelector(`#skip-back-5-${this.id}`).addEventListener("click", (e) => {
            e.stopPropagation();
            this.step(-5);
        });
        controls.querySelector(`#step-back-${this.id}`).addEventListener("click", (e) => {
            e.stopPropagation();
            this.step(-1);
        });
        controls.querySelector(`#step-fwd-${this.id}`).addEventListener("click", (e) => {
            e.stopPropagation();
            this.step(1);
        });
        controls.querySelector(`#skip-fwd-5-${this.id}`).addEventListener("click", (e) => {
            e.stopPropagation();
            this.step(5);
        });

        // Scrubber Slider
        this.slider.addEventListener("input", (e) => {
            const targetF = parseInt(e.target.value, 10);
            this.pause();
            this.seekToFrame(targetF);
        });

        // Delete Button
        header.querySelector(`#delete-${this.id}`).addEventListener("click", (e) => {
            e.stopPropagation();
            if (confirm(`Remove '${this.name}' from active investigation?`)) {
                this.destroy();
                if (this.onUpdate) this.onUpdate("delete", this.id);
            }
        });

        // Rename inline
        const nameEl = header.querySelector(`#name-${this.id}`);
        nameEl.addEventListener("click", (e) => {
            e.stopPropagation();
            const newName = prompt("Rename Camera Location:", this.name);
            if (newName && newName.trim() && newName !== this.name) {
                this.name = newName.trim();
                nameEl.textContent = this.name;
                fetch(`/api/cameras/${this.id}`, {
                    method: "PATCH",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ name: this.name })
                });
            }
        });

        // Fullscreen
        header.querySelector(`#fullscreen-${this.id}`).addEventListener("click", (e) => {
            e.stopPropagation();
            if (!document.fullscreenElement) {
                viewport.requestFullscreen().catch(err => console.error(err));
            } else {
                document.exitFullscreen();
            }
        });
    }

    async loadAnalysis() {
        try {
            const res = await fetch(`/api/cameras/${this.id}/analysis`);
            if (res.ok) {
                const data = await res.json();
                this.analysisData = data.frames || {};
            }
        } catch (e) {
            console.warn(`Could not load analysis for ${this.id}:`, e);
        }
        // Load initial frame
        this.seekToFrame(1);
        this.updateCardBadge();
    }

    updateReference(ref) {
        this.referenceTarget = ref;
        const pip = this.element.querySelector(`#pip-${this.id}`);
        const pipImg = this.element.querySelector(`#pip-img-${this.id}`);
        if (ref && ref.crop_b64) {
            pipImg.src = ref.crop_b64;
            pip.style.display = "flex";
        } else {
            pip.style.display = "none";
        }
        this.drawCurrentFrame();
    }

    setSpeed(speed) {
        this.playbackRate = speed;
    }

    togglePlay() {
        if (this.isPlaying) {
            this.pause();
        } else {
            this.play();
        }
    }

    play() {
        if (this.isPlaying) return;
        this.isPlaying = true;
        this.playBtn.textContent = "❚❚";
        this.pauseOverlay.classList.remove("visible");
        this.lastFrameTime = performance.now();
        this.loop();
    }

    pause() {
        if (!this.isPlaying) return;
        this.isPlaying = false;
        this.playBtn.textContent = "▶";
        this.pauseOverlay.classList.add("visible");
        if (this.animTimer) {
            cancelAnimationFrame(this.animTimer);
            this.animTimer = null;
        }
    }

    step(delta) {
        this.pause();
        let target = this.currentFrame + delta;
        if (target < 1) target = 1;
        if (target > this.totalFrames) target = this.totalFrames;
        this.seekToFrame(target);
    }

    loop() {
        if (!this.isPlaying) return;
        const now = performance.now();
        const frameInterval = (1000 / (this.fps * this.playbackRate));

        if (now - this.lastFrameTime >= frameInterval) {
            this.lastFrameTime = now;
            let nextFrame = this.currentFrame + 1;
            if (nextFrame > this.totalFrames) {
                nextFrame = 1; // loop
            }
            this.seekToFrame(nextFrame);
        }

        this.animTimer = requestAnimationFrame(() => this.loop());
    }

    seekToFrame(frameIdx) {
        this.currentFrame = frameIdx;
        this.slider.value = frameIdx;

        // Update counter text
        const sec = (frameIdx / this.fps).toFixed(1);
        const totalSec = (this.totalFrames / this.fps).toFixed(1);
        this.frameCounter.textContent = `F ${frameIdx}/${this.totalFrames} (${sec}s / ${totalSec}s)`;

        // Check Cache or Fetch
        const cachedImg = this.frameCache.get(frameIdx);
        if (cachedImg && cachedImg.complete) {
            this.drawFrame(cachedImg, frameIdx);
        } else {
            const img = new Image();
            img.src = `/api/cameras/${this.id}/frame?f=${frameIdx}`;
            img.onload = () => {
                this.frameCache.set(frameIdx, img);
                // Simple cache size management
                if (this.frameCache.size > 200) {
                    const firstKey = this.frameCache.keys().next().value;
                    this.frameCache.delete(firstKey);
                }
                if (this.currentFrame === frameIdx) {
                    this.drawFrame(img, frameIdx);
                }
            };
        }

        // Prefetch next 3 frames for zero stutter
        for (let p = 1; p <= 3; p++) {
            const pf = frameIdx + p;
            if (pf <= this.totalFrames && !this.frameCache.has(pf)) {
                const pImg = new Image();
                pImg.src = `/api/cameras/${this.id}/frame?f=${pf}`;
                pImg.onload = () => this.frameCache.set(pf, pImg);
            }
        }
    }

    drawCurrentFrame() {
        const cached = this.frameCache.get(this.currentFrame);
        if (cached) {
            this.drawFrame(cached, this.currentFrame);
        }
    }

    drawFrame(img, frameIdx) {
        const ctx = this.ctx;
        const cw = this.canvas.width;
        const ch = this.canvas.height;

        ctx.clearRect(0, 0, cw, ch);
        ctx.drawImage(img, 0, 0, cw, ch);

        // Get detections for this frame
        const frameDets = this.analysisData[String(frameIdx)] || [];
        let hasMatch = false;
        let hasAlert = false;

        for (const det of frameDets) {
            const [x1, y1, x2, y2] = det.bbox;
            const bw = x2 - x1;
            const bh = y2 - y1;

            ctx.lineWidth = 3;
            ctx.strokeStyle = det.color || "#00e5ff";

            // Glowing bounding box
            ctx.shadowColor = det.color;
            ctx.shadowBlur = det.is_match ? 14 : (det.is_unknown ? 18 : 6);
            ctx.strokeRect(x1, y1, bw, bh);
            ctx.shadowBlur = 0; // reset

            // Corner crosshairs for surveillance aesthetic
            const cornerLen = Math.min(16, bw * 0.2);
            ctx.lineWidth = 4;
            ctx.beginPath();
            ctx.moveTo(x1, y1 + cornerLen); ctx.lineTo(x1, y1); ctx.lineTo(x1 + cornerLen, y1);
            ctx.moveTo(x2 - cornerLen, y1); ctx.lineTo(x2, y1); ctx.lineTo(x2, y1 + cornerLen);
            ctx.moveTo(x1, y2 - cornerLen); ctx.lineTo(x1, y2); ctx.lineTo(x1 + cornerLen, y2);
            ctx.moveTo(x2 - cornerLen, y2); ctx.lineTo(x2, y2); ctx.lineTo(x2, y2 - cornerLen);
            ctx.stroke();

            // Label Badge
            ctx.font = "bold 13px 'JetBrains Mono', monospace";
            const text = det.label;
            const tw = ctx.measureText(text).width;
            const badgeY = Math.max(24, y1 - 8);

            ctx.fillStyle = det.color;
            ctx.fillRect(x1, badgeY - 18, tw + 14, 22);

            ctx.fillStyle = (det.is_unknown) ? "#ffffff" : "#000000";
            ctx.fillText(text, x1 + 7, badgeY - 2);

            if (det.is_match) hasMatch = true;
            if (det.is_unknown) hasAlert = true;
        }

        // Live frame HUD overlay at top
        ctx.fillStyle = "rgba(10, 15, 24, 0.70)";
        ctx.fillRect(10, 10, 240, 26);
        ctx.strokeStyle = "rgba(0, 229, 255, 0.40)";
        ctx.lineWidth = 1;
        ctx.strokeRect(10, 10, 240, 26);

        ctx.font = "bold 11px 'JetBrains Mono', monospace";
        ctx.fillStyle = "#00e5ff";
        ctx.fillText(`CAM: ${this.id.toUpperCase()} | FRAME ${frameIdx}`, 20, 27);

        // Update card border & badge state
        this.element.classList.remove("card-matched", "card-alert", "card-ref");
        if (hasAlert) {
            this.element.classList.add("card-alert");
            this.badge.className = "card-badge badge-alert";
            this.badge.innerHTML = "ALERT: INTRUDER";
        } else if (hasMatch) {
            this.element.classList.add("card-matched");
            this.badge.className = "card-badge badge-match";
            this.badge.innerHTML = "MATCH CONFIRMED";
        } else if (this.summary.status === "REFERENCE") {
            this.element.classList.add("card-ref");
            this.badge.className = "card-badge badge-ref";
            this.badge.innerHTML = "REF TARGET";
        } else {
            this.badge.className = "card-badge badge-scan";
            this.badge.innerHTML = "SCANNING";
        }
    }

    updateCardBadge() {
        if (this.summary.status === "MATCH") {
            this.badge.className = "card-badge badge-match";
            this.badge.textContent = `MATCH (${(this.summary.score * 100).toFixed(1)}%)`;
        } else if (this.summary.status === "REJECTED_UNKNOWN") {
            this.badge.className = "card-badge badge-alert";
            this.badge.textContent = "ALERT: UNKNOWN";
        } else if (this.summary.status === "REFERENCE") {
            this.badge.className = "card-badge badge-ref";
            this.badge.textContent = "REF ENROLLED";
        }
    }

    destroy() {
        this.pause();
        if (this.element && this.element.parentNode) {
            this.element.parentNode.removeChild(this.element);
        }
        this.frameCache.clear();
    }
}


class CCTVCommandCenter {
    constructor() {
        this.cameras = [];
        this.players = new Map();
        this.referenceTarget = null;
        this.isPlayingAll = false;
        this.currentSpeed = 1.0;

        this.init();
    }

    async init() {
        this.bindGlobalControls();
        await this.loadCameras();
    }

    bindGlobalControls() {
        // Master Play / Pause
        const masterPlayBtn = document.getElementById("btn-master-play-pause");
        masterPlayBtn.addEventListener("click", () => this.toggleMasterPlay());

        // Master Skips
        document.getElementById("btn-master-rewind").addEventListener("click", () => {
            this.stepAll(-5);
        });
        document.getElementById("btn-master-step-back").addEventListener("click", () => {
            this.stepAll(-1);
        });
        document.getElementById("btn-master-step-fwd").addEventListener("click", () => {
            this.stepAll(1);
        });
        document.getElementById("btn-master-fwd").addEventListener("click", () => {
            this.stepAll(5);
        });
        document.getElementById("btn-master-reset").addEventListener("click", () => {
            this.resetAll();
        });

        // Speed Select
        document.getElementById("speed-select").addEventListener("change", (e) => {
            this.currentSpeed = parseFloat(e.target.value);
            for (const player of this.players.values()) {
                player.setSpeed(this.currentSpeed);
            }
            this.showToast(`Playback speed set to ${this.currentSpeed}x`);
        });

        // Modal triggers
        document.getElementById("btn-add-camera").addEventListener("click", () => {
            document.getElementById("add-camera-modal").classList.remove("hidden");
        });
        document.getElementById("btn-close-add-modal").addEventListener("click", () => {
            document.getElementById("add-camera-modal").classList.add("hidden");
        });
        document.getElementById("btn-cancel-add").addEventListener("click", () => {
            document.getElementById("add-camera-modal").classList.add("hidden");
        });

        // Verification Report Modal
        document.getElementById("btn-open-report").addEventListener("click", () => this.openReportModal());
        document.getElementById("btn-close-report-modal").addEventListener("click", () => {
            document.getElementById("report-modal").classList.add("hidden");
        });
        document.getElementById("btn-close-report-footer").addEventListener("click", () => {
            document.getElementById("report-modal").classList.add("hidden");
        });
        document.getElementById("btn-download-report").addEventListener("click", () => {
            const a = document.createElement("a");
            a.href = "/static/../cctv_verification_report.jpg";
            a.download = "SAFEPASS_ReID_Verification_Report.jpg";
            a.click();
            this.showToast("Report visual saved to your computer.", "success");
        });

        // Add Camera Form
        this.initAddCameraForm();
    }

    initAddCameraForm() {
        const form = document.getElementById("form-add-camera");
        const fileInput = document.getElementById("input-video-file");
        const dropZone = document.getElementById("drop-zone");
        const dropText = document.getElementById("drop-text");

        fileInput.addEventListener("change", (e) => {
            if (fileInput.files.length > 0) {
                dropText.innerHTML = `Selected: <span class="cyan-accent">${fileInput.files[0].name}</span>`;
            }
        });

        dropZone.addEventListener("dragover", (e) => {
            e.preventDefault();
            dropZone.classList.add("dragover");
        });
        dropZone.addEventListener("dragleave", () => {
            dropZone.classList.remove("dragover");
        });
        dropZone.addEventListener("drop", (e) => {
            e.preventDefault();
            dropZone.classList.remove("dragover");
            if (e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                dropText.innerHTML = `Selected: <span class="cyan-accent">${fileInput.files[0].name}</span>`;
            }
        });

        form.addEventListener("submit", async (e) => {
            e.preventDefault();
            if (!fileInput.files || fileInput.files.length === 0) {
                this.showToast("Please choose a video file first.", "error");
                return;
            }

            const name = document.getElementById("input-cam-name").value.trim();
            const file = fileInput.files[0];

            const progressContainer = document.getElementById("upload-progress-container");
            const progressBar = document.getElementById("progress-bar-fill");
            const progressStatus = document.getElementById("progress-status");

            progressContainer.classList.remove("hidden");
            progressBar.style.width = "35%";
            progressStatus.textContent = "Uploading video to command server...";

            const formData = new FormData();
            formData.append("file", file);
            if (name) formData.append("name", name);

            try {
                progressBar.style.width = "65%";
                progressStatus.textContent = "Running Neural Person Detector & OSNet Re-ID...";

                const res = await fetch("/api/cameras/upload", {
                    method: "POST",
                    body: formData
                });

                if (!res.ok) {
                    const err = await res.json();
                    throw new Error(err.detail || "Upload error");
                }

                progressBar.style.width = "100%";
                progressStatus.textContent = "Re-ID analysis complete! Ingesting camera...";

                setTimeout(async () => {
                    document.getElementById("add-camera-modal").classList.add("hidden");
                    progressContainer.classList.add("hidden");
                    progressBar.style.width = "0%";
                    form.reset();
                    dropText.innerHTML = `Drag & drop video file here or <span class="cyan-accent">Browse</span>`;
                    await this.loadCameras();
                    this.showToast(`Camera '${name || file.name}' added successfully!`, "success");
                }, 800);

            } catch (err) {
                console.error(err);
                progressContainer.classList.add("hidden");
                this.showToast(`Upload failed: ${err.message}`, "error");
            }
        });
    }

    async loadCameras() {
        const grid = document.getElementById("camera-grid");
        try {
            const res = await fetch("/api/cameras");
            if (!res.ok) throw new Error("Could not fetch cameras");
            const data = await res.json();

            this.cameras = data.cameras || [];
            this.referenceTarget = data.reference_target;

            // Update KPI banner
            this.updateKPIBanner();

            // Clear old players not in new list
            const currentIds = new Set(this.cameras.map(c => c.id));
            for (const [id, player] of this.players.entries()) {
                if (!currentIds.has(id)) {
                    player.destroy();
                    this.players.delete(id);
                }
            }

            // Create or update players
            for (const cam of this.cameras) {
                if (!this.players.has(cam.id)) {
                    const player = new CameraPlayer(cam, this.referenceTarget, (action, id) => {
                        if (action === "delete") {
                            this.handleCameraDelete(id);
                        }
                    });
                    this.players.set(cam.id, player);
                    grid.appendChild(player.element);
                } else {
                    this.players.get(cam.id).updateReference(this.referenceTarget);
                }
            }
        } catch (e) {
            console.error("Error loading cameras:", e);
            this.showToast("Could not load camera list. Please verify backend server.", "error");
        }
    }

    async handleCameraDelete(camId) {
        try {
            await fetch(`/api/cameras/${camId}`, { method: "DELETE" });
            this.players.delete(camId);
            this.cameras = this.cameras.filter(c => c.id !== camId);
            this.updateKPIBanner();
            this.showToast(`Camera feed '${camId}' deleted.`, "success");
        } catch (e) {
            this.showToast(`Error deleting camera: ${e.message}`, "error");
        }
    }

    updateKPIBanner() {
        document.getElementById("kpi-cam-count").textContent = `${this.cameras.length} Feeds`;

        if (this.referenceTarget && this.referenceTarget.id) {
            document.getElementById("kpi-ref-name").textContent = `${this.referenceTarget.id} (${this.referenceTarget.cloth_type || 'White Shirt'})`;
            const thumb = document.getElementById("kpi-ref-thumb");
            if (this.referenceTarget.crop_b64) {
                thumb.src = this.referenceTarget.crop_b64;
                thumb.classList.remove("hidden");
            }
        }

        let matchCount = 0;
        let alertCount = 0;
        for (const cam of this.cameras) {
            const status = cam.summary ? cam.summary.status : "";
            if (status === "MATCH" || status === "REFERENCE") matchCount++;
            if (status === "REJECTED_UNKNOWN") alertCount++;
        }

        document.getElementById("kpi-match-count").textContent = `${matchCount} Feeds Verified`;
        document.getElementById("kpi-alert-count").textContent = `${alertCount} Intruder Rejected`;
    }

    toggleMasterPlay() {
        this.isPlayingAll = !this.isPlayingAll;
        const icon = document.getElementById("master-play-icon");
        const text = document.getElementById("master-play-text");

        if (this.isPlayingAll) {
            icon.textContent = "❚❚";
            text.textContent = "PAUSE ALL";
            for (const player of this.players.values()) {
                player.play();
            }
        } else {
            icon.textContent = "▶";
            text.textContent = "PLAY ALL";
            for (const player of this.players.values()) {
                player.pause();
            }
        }
    }

    stepAll(delta) {
        this.isPlayingAll = false;
        document.getElementById("master-play-icon").textContent = "▶";
        document.getElementById("master-play-text").textContent = "PLAY ALL";
        for (const player of this.players.values()) {
            player.step(delta);
        }
    }

    resetAll() {
        this.isPlayingAll = false;
        document.getElementById("master-play-icon").textContent = "▶";
        document.getElementById("master-play-text").textContent = "PLAY ALL";
        for (const player of this.players.values()) {
            player.pause();
            player.seekToFrame(1);
        }
        this.showToast("Reset all cameras to Frame 1.");
    }

    async openReportModal() {
        const modal = document.getElementById("report-modal");
        const body = document.getElementById("report-modal-body");
        modal.classList.remove("hidden");
        body.innerHTML = `<div class="progress-status">Generating Cross-Camera Verification Report...</div>`;

        try {
            const res = await fetch("/api/reid/summary");
            if (!res.ok) throw new Error("Could not load verification report");
            const data = await res.json();

            const ref = data.reference_target;
            const summaries = data.camera_summaries || [];
            const alerts = data.security_alerts || [];

            let html = `
                <div class="report-section">
                    <div class="report-section-title">
                        <span>🎯</span> PRIMARY REFERRAL SUBJECT: ${ref ? ref.id : 'Person_01'} (${ref ? ref.cloth_type : 'White Shirt'})
                        | Enrolled from: ${ref ? ref.source_cam_name : 'Camera 1'}
                    </div>
                    <div class="report-card-grid">
            `;

            for (const item of summaries) {
                const s = item.summary || {};
                const isMatch = s.status === "MATCH" || s.status === "REFERENCE";
                const isReject = s.status === "REJECTED_UNKNOWN";

                html += `
                    <div class="report-item-card ${isMatch ? 'matched' : (isReject ? 'rejected' : '')}">
                        <div class="report-crop-wrapper">
                            ${s.crop_b64 ? `<img src="${s.crop_b64}" class="report-crop-img" alt="Subject">` : `<span style="color:#64748b;">Not Observed</span>`}
                        </div>
                        <div class="report-cam-name">${item.cam_name}</div>
                        <div class="report-badge-status ${isMatch ? 'match' : (isReject ? 'reject' : '')}">
                            ${isMatch ? (s.status === 'REFERENCE' ? 'ENROLLED (F' + s.frame + ')' : `MATCH: ${(s.score * 100).toFixed(1)}% (F${s.frame})`) : (isReject ? 'REJECTED: UNKNOWN' : 'NOT OBSERVED')}
                        </div>
                    </div>
                `;
            }

            html += `</div></div>`;

            // Security Alert Section
            if (alerts.length > 0) {
                html += `
                    <div class="report-section">
                        <div class="report-section-title" style="color: #ef4444;">
                            <span>🚨</span> SECURITY ALERTS: UNKNOWN INDIVIDUALS (ZERO MATCH WITH REFERENCE TARGET)
                        </div>
                `;
                for (const al of alerts) {
                    html += `
                        <div class="report-alert-box">
                            <img src="${al.crop_b64}" class="report-alert-img" alt="Intruder">
                            <div class="report-alert-details">
                                <div class="report-alert-title">UNKNOWN SUBJECT DETECTED: ${al.cam_name} (Frame ${al.frame})</div>
                                <div class="report-alert-desc">
                                    <strong>Rejection Reason:</strong> ${al.reason || 'Appearance/Luminance Mismatch'}<br>
                                    <strong>Raw Posture Similarity:</strong> ${(al.raw_sim * 100).toFixed(1)}% (Hard-Gated & Rejected)<br>
                                    <strong>Verification Verdict:</strong> <span style="color: #ef4444; font-weight: 700;">DIFFERENT INDIVIDUAL (Mask/Black Outfit) - 0.0% Match</span>
                                </div>
                            </div>
                        </div>
                    `;
                }
                html += `</div>`;
            }

            body.innerHTML = html;

        } catch (e) {
            console.error(e);
            body.innerHTML = `<div class="progress-status" style="color:#ef4444;">Failed to load report: ${e.message}</div>`;
        }
    }

    showToast(message, type = "info") {
        const container = document.getElementById("toast-container");
        const toast = document.createElement("div");
        toast.className = `toast ${type === 'error' ? 'toast-error' : (type === 'success' ? 'toast-success' : '')}`;
        toast.textContent = message;
        container.appendChild(toast);
        setTimeout(() => {
            if (toast.parentNode) toast.parentNode.removeChild(toast);
        }, 3500);
    }
}

// Initialize on DOM load
window.addEventListener("DOMContentLoaded", () => {
    window.app = new CCTVCommandCenter();
});
