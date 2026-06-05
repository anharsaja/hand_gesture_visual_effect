# Polygon area, point-in-polygon, overlay
"""
interaction_manager.py
======================
Modul yang mengelola:
    1. Definisi area filter (polygon antar jari) per tangan
    2. Point-in-polygon detection (ujung jari tangan lain di dalam area)
    3. Pembuatan mask untuk compositing filter
    4. Rendering overlay area (polygon + label transparan)

Arsitektur:
    InteractionManager
        └── Menyimpan dict area_id → FingerGapArea
    FingerGapArea
        └── Polygon dinamis dari 4 titik jari + dua MCP

Area per tangan kanan (bisa dibalik untuk kiri):
    A = celah antara INDEX_TIP  dan MIDDLE_TIP
    B = celah antara MIDDLE_TIP dan RING_TIP
    C = celah antara RING_TIP   dan PINKY_TIP
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from hand_tracker import HandData, LandmarkIndex

# ─────────────────────────────────────────────
#  Konstanta
# ─────────────────────────────────────────────

# Definisi pasangan jari untuk setiap area
# Format: (TIP_kiri, TIP_kanan, MCP_kiri, MCP_kanan)
AREA_FINGER_PAIRS: Dict[str, Tuple[int, int, int, int]] = {
    "A": (
        LandmarkIndex.INDEX_FINGER_TIP,  # ujung telunjuk
        LandmarkIndex.MIDDLE_FINGER_TIP,  # ujung tengah
        LandmarkIndex.INDEX_FINGER_MCP,  # pangkal telunjuk
        LandmarkIndex.MIDDLE_FINGER_MCP,  # pangkal tengah
    ),
    "B": (
        LandmarkIndex.MIDDLE_FINGER_TIP,  # ujung tengah
        LandmarkIndex.RING_FINGER_TIP,  # ujung manis
        LandmarkIndex.MIDDLE_FINGER_MCP,  # pangkal tengah
        LandmarkIndex.RING_FINGER_MCP,  # pangkal manis
    ),
    "C": (
        LandmarkIndex.RING_FINGER_TIP,  # ujung manis
        LandmarkIndex.PINKY_TIP,  # ujung kelingking
        LandmarkIndex.RING_FINGER_MCP,  # pangkal manis
        LandmarkIndex.PINKY_MCP,  # pangkal kelingking
    ),
}

# Warna area (BGR) + alpha overlay
AREA_STYLES: Dict[str, Dict] = {
    "A": {"color": (255, 120, 30), "alpha": 0.30, "label": "AREA A  [BLUR]"},
    "B": {"color": (30, 210, 90), "alpha": 0.30, "label": "AREA B  [GRAY]"},
    "C": {"color": (30, 90, 255), "alpha": 0.30, "label": "AREA C  [EDGE]"},
}

# Warna aktif (lebih cerah saat jari masuk)
AREA_ACTIVE_COLOR: Dict[str, Tuple[int, int, int]] = {
    "A": (255, 200, 80),
    "B": (80, 255, 130),
    "C": (80, 160, 255),
}


# ─────────────────────────────────────────────
#  Dataclass: satu area polygon
# ─────────────────────────────────────────────


@dataclass
class FingerGapArea:
    """
    Merepresentasikan satu area celah antar dua jari.

    Polygon dibentuk dari 4 titik:
      [TIP_kiri, TIP_kanan, MCP_kanan, MCP_kiri]
    Ini menghasilkan trapezoid mengikuti bentuk celah jari.
    """

    area_id: str
    polygon: Optional[np.ndarray] = None  # shape (4,2), dtype int32
    is_active: bool = False
    centroid: Optional[Tuple[int, int]] = None

    def update_from_hand(self, hand: HandData, pair: Tuple[int, int, int, int]) -> None:
        """
        Hitung ulang polygon berdasarkan posisi landmark tangan saat ini.

        Parameters
        ----------
        hand : HandData dengan landmark terkini
        pair : (TIP_A, TIP_B, MCP_A, MCP_B) – indeks landmark
        """
        tip_a_idx, tip_b_idx, mcp_a_idx, mcp_b_idx = pair

        tip_a = hand.landmarks[tip_a_idx]
        tip_b = hand.landmarks[tip_b_idx]
        mcp_a = hand.landmarks[mcp_a_idx]
        mcp_b = hand.landmarks[mcp_b_idx]

        # Polygon searah jarum jam: TIP_A → TIP_B → MCP_B → MCP_A
        self.polygon = np.array(
            [
                [tip_a[0], tip_a[1]],
                [tip_b[0], tip_b[1]],
                [mcp_b[0], mcp_b[1]],
                [mcp_a[0], mcp_a[1]],
            ],
            dtype=np.int32,
        )

        # Hitung centroid untuk label
        cx = int(np.mean(self.polygon[:, 0]))
        cy = int(np.mean(self.polygon[:, 1]))
        self.centroid = (cx, cy)

    def contains_point(self, point: Tuple[int, int]) -> bool:
        """
        Cek apakah sebuah titik berada di dalam polygon ini.

        Menggunakan cv2.pointPolygonTest (efisien, C++ di balik layar).
        Returns True jika di dalam atau di tepi polygon.
        """
        if self.polygon is None:
            return False
        # pointPolygonTest: > 0 → dalam, 0 → tepi, < 0 → luar
        result = cv2.pointPolygonTest(
            self.polygon, (float(point[0]), float(point[1])), False
        )
        return result >= 0

    def to_mask(self, frame_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Buat binary mask (uint8) dari polygon ini.

        Parameters
        ----------
        frame_shape : (height, width) frame

        Returns
        -------
        Mask uint8 dengan nilai 255 di dalam polygon, 0 di luar.
        None jika polygon belum dihitung.
        """
        if self.polygon is None:
            return None
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [self.polygon], 255)
        return mask


# ─────────────────────────────────────────────
#  InteractionManager
# ─────────────────────────────────────────────


class InteractionManager:
    """
    Mengelola semua area interaktif dan deteksi interaksi.

    Alur kerja per frame:
    1. update_areas(hand_primary)  → hitung polygon dari posisi tangan
    2. check_interactions(hand_secondary) → cek ujung jari di dalam area
    3. get_area_masks(frame_shape) → buat mask untuk compositing
    4. draw_areas(frame)           → render overlay di frame

    Parameters
    ----------
    primary_handedness : 'Right' atau 'Left' – tangan yang memiliki area
    """

    def __init__(self, primary_handedness: str = "Right"):
        self.primary_handedness = primary_handedness
        self.secondary_handedness = "Left" if primary_handedness == "Right" else "Right"

        # Inisialisasi tiga area
        self.areas: Dict[str, FingerGapArea] = {
            area_id: FingerGapArea(area_id=area_id) for area_id in AREA_FINGER_PAIRS
        }

        # Cache mask per area (dihitung ulang hanya jika polygon berubah)
        self._mask_cache: Dict[str, Optional[np.ndarray]] = {
            k: None for k in AREA_FINGER_PAIRS
        }
        self._last_polygons: Dict[str, Optional[np.ndarray]] = {
            k: None for k in AREA_FINGER_PAIRS
        }

        # Frame shape untuk mask
        self._frame_shape: Tuple[int, int] = (480, 640)

        # Titik-titik trigger (ujung jari tangan sekunder yang aktif)
        self.trigger_points: List[Tuple[int, int]] = []

        # Log status interaksi untuk display
        self.interaction_log: List[str] = []

    # ─────────────────────────────────────────
    #  Update area polygon
    # ─────────────────────────────────────────

    def update_areas(self, hands: List[HandData]) -> Optional[HandData]:
        """
        Cari tangan primer dan perbarui polygon semua area.

        Parameters
        ----------
        hands : list semua tangan yang terdeteksi

        Returns
        -------
        HandData tangan primer jika ditemukan, None jika tidak
        """
        primary_hand = self._find_hand(hands, self.primary_handedness)

        if primary_hand is None:
            # Reset semua area jika tangan primer tidak terlihat
            for area in self.areas.values():
                area.polygon = None
                area.centroid = None
                area.is_active = False
            return None

        # Perbarui polygon setiap area
        for area_id, area in self.areas.items():
            pair = AREA_FINGER_PAIRS[area_id]
            area.update_from_hand(primary_hand, pair)

        return primary_hand

    # ─────────────────────────────────────────
    #  Cek interaksi
    # ─────────────────────────────────────────

    def check_interactions(self, hands: List[HandData]) -> Dict[str, bool]:
        """
        Cek apakah ujung jari tangan sekunder berada di dalam area manapun.

        Parameters
        ----------
        hands : list semua tangan terdeteksi

        Returns
        -------
        Dict area_id → bool (True = ada jari di dalam area)
        """
        secondary_hand = self._find_hand(hands, self.secondary_handedness)
        results: Dict[str, bool] = {k: False for k in self.areas}

        self.trigger_points = []
        self.interaction_log = []

        if secondary_hand is None:
            for area in self.areas.values():
                area.is_active = False
            return results

        # Ambil semua ujung jari tangan sekunder sebagai trigger
        fingertips = secondary_hand.get_fingertips()
        probe_points = list(fingertips.values())

        for area_id, area in self.areas.items():
            if area.polygon is None:
                continue

            hit = False
            for pt in probe_points:
                if area.contains_point(pt):
                    hit = True
                    self.trigger_points.append(pt)
                    self.interaction_log.append(
                        f"{secondary_hand.handedness} finger → {area_id}"
                    )

            area.is_active = hit
            results[area_id] = hit

        return results

    # ─────────────────────────────────────────
    #  Mask untuk compositing
    # ─────────────────────────────────────────

    def get_area_masks(
        self,
        frame_shape: Tuple[int, int],
    ) -> Dict[str, Optional[np.ndarray]]:
        """
        Kembalikan mask binary untuk setiap area.
        Menggunakan cache – mask hanya dihitung ulang jika polygon berubah.

        Parameters
        ----------
        frame_shape : (height, width) atau (height, width, channels)

        Returns
        -------
        Dict area_id → mask uint8 (atau None)
        """
        self._frame_shape = frame_shape[:2]
        masks: Dict[str, Optional[np.ndarray]] = {}

        for area_id, area in self.areas.items():
            poly = area.polygon

            # Cek apakah polygon berubah dari cache
            prev = self._last_polygons[area_id]
            if poly is None:
                masks[area_id] = None
                self._mask_cache[area_id] = None
                self._last_polygons[area_id] = None
                continue

            if prev is None or not np.array_equal(poly, prev):
                # Hitung ulang mask
                self._mask_cache[area_id] = area.to_mask(frame_shape)
                self._last_polygons[area_id] = poly.copy()

            masks[area_id] = self._mask_cache[area_id]

        return masks

    # ─────────────────────────────────────────
    #  Rendering overlay
    # ─────────────────────────────────────────

    def draw_areas(
        self,
        frame: np.ndarray,
        show_inactive: bool = True,
    ) -> None:
        """
        Gambar overlay semua area di atas frame (in-place).
        Area aktif ditampilkan lebih terang + border lebih tebal.

        Parameters
        ----------
        frame         : frame BGR (dimodifikasi in-place)
        show_inactive : jika False, hanya area aktif yang digambar
        """
        overlay = frame.copy()

        for area_id, area in self.areas.items():
            if area.polygon is None:
                continue
            if not show_inactive and not area.is_active:
                continue

            style = AREA_STYLES[area_id]
            color = AREA_ACTIVE_COLOR[area_id] if area.is_active else style["color"]
            alpha = 0.50 if area.is_active else style["alpha"]

            # Fill polygon transparan
            cv2.fillPoly(overlay, [area.polygon], color)

            # Border polygon
            border_thick = 3 if area.is_active else 1
            cv2.polylines(
                frame,
                [area.polygon],
                isClosed=True,
                color=color,
                thickness=border_thick,
                lineType=cv2.LINE_AA,
            )

        # Blend overlay ke frame dengan alpha
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        # Label teks di tengah setiap area
        for area_id, area in self.areas.items():
            if area.centroid is None:
                continue
            if not show_inactive and not area.is_active:
                continue

            style = AREA_STYLES[area_id]
            color = AREA_ACTIVE_COLOR[area_id] if area.is_active else style["color"]
            label = style["label"]
            cx, cy = area.centroid

            # Shadow teks
            cv2.putText(
                frame,
                label,
                (cx - 41, cy + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            # Teks utama
            cv2.putText(
                frame,
                label,
                (cx - 40, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

            # Indikator aktif
            if area.is_active:
                cv2.circle(frame, (cx + 55, cy - 2), 6, (0, 255, 100), -1, cv2.LINE_AA)

        # Gambar titik trigger (ujung jari yang menyentuh area)
        for pt in self.trigger_points:
            cv2.circle(frame, pt, 10, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, pt, 4, (0, 255, 255), -1, cv2.LINE_AA)

    # ─────────────────────────────────────────
    #  Helper
    # ─────────────────────────────────────────

    def _find_hand(
        self,
        hands: List[HandData],
        handedness: str,
    ) -> Optional[HandData]:
        """
        Cari HandData dengan handedness tertentu dari list.
        Kembalikan None jika tidak ditemukan.
        """
        for hand in hands:
            if hand.handedness == handedness:
                return hand
        return None

    def swap_primary(self) -> None:
        """
        Tukar tangan primer dan sekunder.
        Berguna saat pengguna mau pakai tangan kiri sebagai area.
        """
        self.primary_handedness, self.secondary_handedness = (
            self.secondary_handedness,
            self.primary_handedness,
        )
        # Reset cache
        self._mask_cache = {k: None for k in AREA_FINGER_PAIRS}
        self._last_polygons = {k: None for k in AREA_FINGER_PAIRS}

    def get_active_areas(self) -> List[str]:
        """Kembalikan list area_id yang sedang aktif."""
        return [aid for aid, a in self.areas.items() if a.is_active]
