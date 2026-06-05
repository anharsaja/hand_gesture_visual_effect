# Registry filter OpenCV + compositing
"""
filter_manager.py
=================
Modul yang menyimpan dan mengelola semua filter visual OpenCV.

Desain:
    - Setiap filter adalah callable (fungsi) yang menerima frame BGR
    dan mengembalikan frame BGR hasil filter.
    - FilterManager menyimpan registry filter sehingga mudah
    ditambah/dihapus tanpa mengubah kode di tempat lain.
    - Mendukung "overlay partial" – filter hanya diterapkan pada
    region tertentu (mask), bukan seluruh frame.

Filter bawaan:
    BLUR      → Gaussian Blur
    GRAYSCALE → Konversi ke grayscale lalu kembali ke BGR
    EDGE      → Canny Edge Detection
    NONE      → Identitas (tidak ada efek)
"""

import cv2
import numpy as np
from typing import Callable, Dict, Optional, Tuple

# ─────────────────────────────────────────────
#  Tipe alias
# ─────────────────────────────────────────────
FilterFunc = Callable[[np.ndarray], np.ndarray]


# ─────────────────────────────────────────────
#  Fungsi-fungsi filter
# ─────────────────────────────────────────────


def filter_none(frame: np.ndarray) -> np.ndarray:
    """Tidak ada efek – kembalikan frame asli."""
    return frame


def filter_gaussian_blur(
    frame: np.ndarray,
    ksize: int = 25,
    sigma: float = 0,
) -> np.ndarray:
    """
    Gaussian Blur.

    Parameters
    ----------
    ksize : ukuran kernel (harus ganjil, lebih besar = lebih blur)
    sigma : standar deviasi Gaussian (0 = auto dari ksize)
    """
    # Pastikan ksize ganjil
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(frame, (ksize, ksize), sigma)


def filter_grayscale(frame: np.ndarray) -> np.ndarray:
    """
    Konversi ke grayscale dan kembalikan sebagai BGR 3-channel
    agar pipeline tidak perlu menangani channel berbeda.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Kembalikan ke BGR (3 channel) agar bisa di-blend dengan frame asli
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def filter_canny_edge(
    frame: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
    aperture: int = 3,
) -> np.ndarray:
    """
    Canny Edge Detection.

    Parameters
    ----------
    low_threshold  : threshold bawah hysteresis
    high_threshold : threshold atas hysteresis
    aperture       : ukuran kernel Sobel (3, 5, atau 7)

    Proses:
    1. Konversi ke grayscale
    2. Gaussian pre-blur untuk kurangi noise
    3. Canny edge
    4. Invert + kembalikan ke BGR
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, low_threshold, high_threshold, apertureSize=aperture)
    # Invert agar tepi gelap di latar putih → lebih estetis
    edges_inv = cv2.bitwise_not(edges)
    return cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2BGR)


def filter_pixelate(frame: np.ndarray, pixel_size: int = 16) -> np.ndarray:
    """
    Filter ekstra: Pixelate / mosaic effect.
    Berguna sebagai contoh cara menambah filter baru.

    Parameters
    ----------
    pixel_size : ukuran satu "pixel" mosaic
    """
    h, w = frame.shape[:2]
    # Kecilkan lalu besarkan kembali → efek pixelate
    small = cv2.resize(
        frame, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR
    )
    result = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return result


def filter_sepia(frame: np.ndarray) -> np.ndarray:
    """
    Filter ekstra: Sepia tone.
    Contoh filter warna berbasis matrix transform.
    """
    kernel = np.array(
        [
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189],
        ],
        dtype=np.float32,
    )
    sepia = cv2.transform(frame.astype(np.float32), kernel)
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)
    return sepia


# ─────────────────────────────────────────────
#  Filter Registry & Manager
# ─────────────────────────────────────────────


class FilterManager:
    """
    Pengelola registry filter.

    Cara pakai:
        fm = FilterManager()
        result = fm.apply('BLUR', frame)

        # Tambah filter kustom:
        fm.register('MY_FILTER', my_filter_func)
        result = fm.apply('MY_FILTER', frame)

        # Terapkan filter hanya pada region (mask):
        result = fm.apply_with_mask('EDGE', frame, polygon_mask)
    """

    # Nama filter default untuk setiap area
    AREA_FILTER_MAP: Dict[str, str] = {
        "A": "BLUR",
        "B": "GRAYSCALE",
        "C": "EDGE",
    }

    # Warna overlay area (BGR) – dipakai untuk polygon transparan
    AREA_COLORS: Dict[str, Tuple[int, int, int]] = {
        "A": (255, 100, 50),  # Biru-ungu  (BGR)
        "B": (50, 220, 100),  # Hijau terang
        "C": (50, 100, 255),  # Oranye-merah
    }

    def __init__(self):
        # Registry: nama → fungsi filter
        self._registry: Dict[str, FilterFunc] = {
            "NONE": filter_none,
            "BLUR": filter_gaussian_blur,
            "GRAYSCALE": filter_grayscale,
            "EDGE": filter_canny_edge,
            "PIXELATE": filter_pixelate,
            "SEPIA": filter_sepia,
        }

        # Status aktif per area (True = filter sedang aktif)
        self.active_areas: Dict[str, bool] = {
            "A": False,
            "B": False,
            "C": False,
        }

    # ─────────────────────────────────────────
    #  Registrasi
    # ─────────────────────────────────────────

    def register(self, name: str, func: FilterFunc) -> None:
        """
        Tambahkan filter baru ke registry.

        Parameters
        ----------
        name : nama unik filter (huruf besar disarankan)
        func : callable (np.ndarray → np.ndarray)
        """
        self._registry[name.upper()] = func

    def list_filters(self):
        """Kembalikan daftar nama filter yang tersedia."""
        return list(self._registry.keys())

    # ─────────────────────────────────────────
    #  Penerapan Filter
    # ─────────────────────────────────────────

    def apply(self, filter_name: str, frame: np.ndarray) -> np.ndarray:
        """
        Terapkan filter ke seluruh frame.

        Parameters
        ----------
        filter_name : nama filter di registry
        frame       : frame BGR input

        Returns
        -------
        Frame BGR hasil filter (copy baru, frame asli tidak diubah)
        """
        name = filter_name.upper()
        if name not in self._registry:
            return frame.copy()
        return self._registry[name](frame.copy())

    def apply_with_mask(
        self,
        filter_name: str,
        frame: np.ndarray,
        mask: np.ndarray,
        blend_alpha: float = 0.85,
    ) -> np.ndarray:
        """
        Terapkan filter HANYA pada region yang ditandai oleh mask.
        Di luar mask, frame asli dipertahankan.

        Parameters
        ----------
        filter_name : nama filter
        frame       : frame BGR asli (tidak diubah)
        mask        : binary mask uint8 (255 = area filter, 0 = area asli)
        blend_alpha : intensitas efek (0.0 = tidak ada, 1.0 = penuh)

        Returns
        -------
        Frame BGR baru hasil compositing
        """
        filtered = self.apply(filter_name, frame)

        # Blend hanya di dalam mask
        mask_3ch = cv2.merge([mask, mask, mask])  # jadikan 3 channel
        mask_f = mask_3ch.astype(np.float32) / 255.0

        # result = filtered * alpha * mask + frame * (1 - alpha * mask)
        result = (
            filtered.astype(np.float32) * mask_f * blend_alpha
            + frame.astype(np.float32) * (1.0 - mask_f * blend_alpha)
        ).astype(np.uint8)

        return result

    def apply_all_active(
        self,
        frame: np.ndarray,
        area_masks: Dict[str, Optional[np.ndarray]],
        blend_alpha: float = 0.85,
    ) -> np.ndarray:
        """
        Terapkan semua filter dari area yang sedang aktif secara berurutan.

        Parameters
        ----------
        frame       : frame BGR asli
        area_masks  : dict area_id → mask (atau None jika area tidak ada)
        blend_alpha : intensitas efek

        Returns
        -------
        Frame BGR hasil semua compositing aktif
        """
        result = frame.copy()

        for area_id, is_active in self.active_areas.items():
            if not is_active:
                continue

            mask = area_masks.get(area_id)
            if mask is None:
                continue

            filter_name = self.AREA_FILTER_MAP.get(area_id, "NONE")
            result = self.apply_with_mask(filter_name, result, mask, blend_alpha)

        return result

    # ─────────────────────────────────────────
    #  Status Area
    # ─────────────────────────────────────────

    def set_area_active(self, area_id: str, active: bool) -> None:
        """Aktifkan / nonaktifkan filter sebuah area."""
        if area_id in self.active_areas:
            self.active_areas[area_id] = active

    def reset_all(self) -> None:
        """Nonaktifkan semua area (frame bersih)."""
        for key in self.active_areas:
            self.active_areas[key] = False

    def get_active_filter_name(self, area_id: str) -> str:
        """Kembalikan nama filter yang terkait area, atau 'NONE'."""
        return self.AREA_FILTER_MAP.get(area_id, "NONE")
