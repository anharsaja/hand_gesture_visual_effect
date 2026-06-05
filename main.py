# Entry point – loop utama & rendering
"""
main.py
=======
Entry point aplikasi Computer Vision real-time.

Pipeline per frame:
    1. Baca frame dari webcam  (flip horizontal = mode cermin)
    2. HandTracker.process()   → deteksi landmark
    3. InteractionManager.update_areas()   → hitung polygon
    4. InteractionManager.check_interactions() → deteksi jari di area
    5. InteractionManager.get_area_masks()  → buat mask compositing
    6. FilterManager.apply_all_active()    → terapkan filter ke frame
    7. HandTracker.draw_landmarks()        → gambar skeleton
    8. InteractionManager.draw_areas()     → gambar overlay area
    9. draw_hud()                           → gambar HUD / info layar
    10. cv2.imshow()                        → tampilkan

Keyboard shortcut:
    Q / ESC  : Keluar
    S        : Swap tangan primer/sekunder
    H        : Toggle tampilan HUD
    F        : Toggle semua filter on/off
    R        : Reset (nonaktifkan semua filter)
    1-5      : Ganti ukuran blur (kecil → besar)
"""

import sys
import time
import cv2
import numpy as np
from collections import deque
from typing import Deque, List

# ─── Import modul lokal ───────────────────────────────────────────────────────
from hand_tracker import HandTracker, HandData
from filter_manager import FilterManager
from interaction_manager import InteractionManager

# ═════════════════════════════════════════════════════════════════════════════
#  Konfigurasi
# ═════════════════════════════════════════════════════════════════════════════


class Config:
    # Kamera
    CAMERA_INDEX: int = 0
    FRAME_WIDTH: int = 1280
    FRAME_HEIGHT: int = 720
    TARGET_FPS: int = 30

    # MediaPipe
    MAX_HANDS: int = 2
    DETECT_CONFIDENCE: float = 0.7
    TRACK_CONFIDENCE: float = 0.6
    MODEL_COMPLEXITY: int = 0  # 0 = ringan (30fps), 1 = akurat

    # Interaksi
    PRIMARY_HAND: str = "Right"  # tangan yang memiliki area

    # UI
    SHOW_HUD: bool = True
    SHOW_FPS: bool = True
    WINDOW_NAME: str = "Hand Gesture Filter  |  Q=Quit  S=Swap  H=HUD"

    # Filter
    FILTER_BLEND_ALPHA: float = 0.88  # intensitas filter (0–1)
    BLUR_KSIZE: int = 25  # ukuran kernel gaussian blur


# ═════════════════════════════════════════════════════════════════════════════
#  FPS Counter
# ═════════════════════════════════════════════════════════════════════════════


class FPSCounter:
    """Rolling average FPS menggunakan deque."""

    def __init__(self, window: int = 30):
        self._times: Deque[float] = deque(maxlen=window)
        self._last = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now - self._last)
        self._last = now
        if len(self._times) < 2:
            return 0.0
        return 1.0 / (sum(self._times) / len(self._times))


# ═════════════════════════════════════════════════════════════════════════════
#  HUD Renderer
# ═════════════════════════════════════════════════════════════════════════════


def draw_hud(
    frame: np.ndarray,
    fps: float,
    active_areas: List[str],
    primary_hand: str,
    show_hud: bool,
    filters_enabled: bool,
    interaction_log: List[str],
) -> None:
    """
    Gambar HUD informatif di tepi frame.

    Informasi yang ditampilkan:
    - FPS
    - Tangan primer
    - Area aktif + nama filter
    - Log interaksi terakhir
    - Keyboard shortcut
    - Indikator filter on/off
    """
    if not show_hud:
        # Mode minimal: hanya FPS
        cv2.putText(
            frame,
            f"{fps:.0f} FPS",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 100),
            2,
            cv2.LINE_AA,
        )
        return

    h, w = frame.shape[:2]

    # ── Panel kiri atas ───────────────────────────────────────────────────
    panel_x, panel_y = 10, 10
    line_h = 22

    lines = [
        ("=== HAND GESTURE FILTER ===", (200, 200, 255)),
        (f"FPS : {fps:5.1f}", (0, 255, 150) if fps >= 25 else (0, 100, 255)),
        (f"PRIMARY HAND : {primary_hand}", (220, 220, 50)),
        (
            f"FILTERS : {'ON' if filters_enabled else 'OFF'}",
            (0, 255, 0) if filters_enabled else (0, 0, 255),
        ),
        ("", (200, 200, 200)),
        ("ACTIVE AREAS:", (200, 200, 200)),
    ]

    filter_names = {"A": "Gaussian Blur", "B": "Grayscale", "C": "Canny Edge"}
    area_colors = {
        "A": (255, 180, 80),
        "B": (80, 255, 130),
        "C": (80, 160, 255),
    }

    for area_id in ["A", "B", "C"]:
        is_active = area_id in active_areas
        status = "● AKTIF" if is_active else "○"
        color = area_colors[area_id] if is_active else (100, 100, 100)
        lines.append((f"  {status} Area {area_id}: {filter_names[area_id]}", color))

    lines += [
        ("", (200, 200, 200)),
        ("SHORTCUTS:", (180, 180, 255)),
        ("  Q/ESC : Keluar", (160, 160, 160)),
        ("  S     : Swap Hand", (160, 160, 160)),
        ("  H     : Toggle HUD", (160, 160, 160)),
        ("  F     : Toggle Filter", (160, 160, 160)),
        ("  R     : Reset", (160, 160, 160)),
    ]

    # Gambar background semi-transparan
    bg_h = len(lines) * line_h + 16
    bg_w = 270
    sub = frame[panel_y : panel_y + bg_h, panel_x : panel_x + bg_w]
    if sub.shape[0] > 0 and sub.shape[1] > 0:
        black = np.zeros_like(sub)
        cv2.addWeighted(black, 0.55, sub, 0.45, 0, sub)
        frame[panel_y : panel_y + bg_h, panel_x : panel_x + bg_w] = sub

    # Gambar teks
    for i, (text, color) in enumerate(lines):
        if text:
            cv2.putText(
                frame,
                text,
                (panel_x + 8, panel_y + 20 + i * line_h),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )

    # ── Log interaksi (kanan atas) ────────────────────────────────────────
    if interaction_log:
        log_x = w - 280
        for j, log in enumerate(interaction_log[-4:]):  # tampilkan 4 terakhir
            cv2.putText(
                frame,
                f"> {log}",
                (log_x, 30 + j * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 200),
                1,
                cv2.LINE_AA,
            )

    # ── Petunjuk cara pakai (bawah) ────────────────────────────────────────
    instruction = "Rentangkan tangan kanan | Sentuh area dengan jari tangan kiri"
    cv2.putText(
        frame,
        instruction,
        (w // 2 - 300, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  Inisialisasi komponen
# ═════════════════════════════════════════════════════════════════════════════


def init_camera(config: Config) -> cv2.VideoCapture:
    """Buka kamera dan set resolusi/FPS target."""
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Tidak bisa membuka kamera index {config.CAMERA_INDEX}. "
            "Pastikan webcam terhubung."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)
    # Buffer kecil agar latency rendah
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


# ═════════════════════════════════════════════════════════════════════════════
#  Main loop
# ═════════════════════════════════════════════════════════════════════════════


def main():
    cfg = Config()

    # ── Inisialisasi ──────────────────────────────────────────────────────
    print("[INFO] Memulai aplikasi Hand Gesture Filter...")
    print("[INFO] Inisialisasi kamera...")
    cap = init_camera(cfg)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Resolusi: {actual_w}x{actual_h}")

    print("[INFO] Inisialisasi MediaPipe Hands...")
    tracker = HandTracker(
        max_hands=cfg.MAX_HANDS,
        detection_confidence=cfg.DETECT_CONFIDENCE,
        tracking_confidence=cfg.TRACK_CONFIDENCE,
        model_complexity=cfg.MODEL_COMPLEXITY,
    )

    print("[INFO] Inisialisasi Filter Manager...")
    filter_mgr = FilterManager()

    print("[INFO] Inisialisasi Interaction Manager...")
    interact_mgr = InteractionManager(primary_handedness=cfg.PRIMARY_HAND)

    fps_counter = FPSCounter(window=30)
    show_hud = cfg.SHOW_HUD
    filters_enabled = True

    # Buat window
    cv2.namedWindow(cfg.WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(cfg.WINDOW_NAME, actual_w, actual_h)

    print("[INFO] Aplikasi berjalan. Tekan Q atau ESC untuk keluar.")
    print("=" * 55)

    # ── Loop utama ────────────────────────────────────────────────────────
    while True:
        # ── 1. Baca frame ─────────────────────────────────────────────
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame tidak terbaca, mencoba lagi...")
            continue

        # Flip horizontal → mode cermin (lebih intuitif)
        frame = cv2.flip(frame, 1)

        # ── 2. Deteksi tangan ─────────────────────────────────────────
        hands: List[HandData] = tracker.process(frame)

        # ── 3. Update polygon area ────────────────────────────────────
        interact_mgr.update_areas(hands)

        # ── 4. Cek interaksi (jari sekunder di dalam area) ───────────
        interaction_results = interact_mgr.check_interactions(hands)

        # Sinkronisasi status aktif ke FilterManager
        if filters_enabled:
            for area_id, is_active in interaction_results.items():
                filter_mgr.set_area_active(area_id, is_active)
        else:
            filter_mgr.reset_all()

        # ── 5. Buat mask compositing ──────────────────────────────────
        area_masks = interact_mgr.get_area_masks(frame.shape)

        # ── 6. Terapkan filter ke frame ───────────────────────────────
        if filters_enabled:
            frame = filter_mgr.apply_all_active(
                frame,
                area_masks,
                blend_alpha=cfg.FILTER_BLEND_ALPHA,
            )

        # ── 7. Gambar skeleton tangan ─────────────────────────────────
        tracker.draw_landmarks(frame, hands)

        # ── 8. Gambar overlay area ────────────────────────────────────
        interact_mgr.draw_areas(frame, show_inactive=True)

        # ── 9. Gambar HUD ─────────────────────────────────────────────
        fps = fps_counter.tick()
        draw_hud(
            frame=frame,
            fps=fps,
            active_areas=interact_mgr.get_active_areas(),
            primary_hand=interact_mgr.primary_handedness,
            show_hud=show_hud,
            filters_enabled=filters_enabled,
            interaction_log=interact_mgr.interaction_log,
        )

        # ── 10. Tampilkan frame ────────────────────────────────────────
        cv2.imshow(cfg.WINDOW_NAME, frame)

        # ── 11. Event keyboard ────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q"), 27):  # Q atau ESC → keluar
            print("[INFO] Keluar...")
            break

        elif key in (ord("s"), ord("S")):  # S → swap tangan
            interact_mgr.swap_primary()
            print(f"[INFO] Primary hand → {interact_mgr.primary_handedness}")

        elif key in (ord("h"), ord("H")):  # H → toggle HUD
            show_hud = not show_hud

        elif key in (ord("f"), ord("F")):  # F → toggle filter
            filters_enabled = not filters_enabled
            if not filters_enabled:
                filter_mgr.reset_all()
            print(f"[INFO] Filters {'enabled' if filters_enabled else 'disabled'}")

        elif key in (ord("r"), ord("R")):  # R → reset
            filter_mgr.reset_all()
            print("[INFO] Filters reset")

        elif key == ord("1"):  # 1 → blur halus
            cfg.FILTER_BLEND_ALPHA = 0.5
        elif key == ord("2"):  # 2 → blur sedang
            cfg.FILTER_BLEND_ALPHA = 0.75
        elif key == ord("3"):  # 3 → blur penuh
            cfg.FILTER_BLEND_ALPHA = 1.0

    # ── Bersihkan resource ────────────────────────────────────────────────
    print("[INFO] Membersihkan resource...")
    tracker.release()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Selesai.")


# ═════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INFO] Dihentikan oleh pengguna.")
        cv2.destroyAllWindows()
        sys.exit(0)
