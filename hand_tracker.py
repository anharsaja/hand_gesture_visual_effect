# MediaPipe wrapper & skeleton drawing
"""
hand_tracker.py
===============
Kompatibel dengan MediaPipe 0.10+ (Task API) DAN 0.9.x (solutions API).
Deteksi otomatis versi yang terinstall, lalu pakai API yang sesuai.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import mediapipe as mp


# ─────────────────────────────────────────────
#  Deteksi versi MediaPipe
# ─────────────────────────────────────────────
_MP_VERSION = tuple(int(x) for x in mp.__version__.split(".")[:2])
_USE_TASK_API = _MP_VERSION >= (0, 10)
print(f"[HandTracker] MediaPipe {mp.__version__} → {'Task API' if _USE_TASK_API else 'Solutions API'}")


# ─────────────────────────────────────────────
#  Konstanta indeks landmark MediaPipe
# ─────────────────────────────────────────────
class LandmarkIndex:
    WRIST               = 0
    THUMB_CMC           = 1
    THUMB_MCP           = 2
    THUMB_IP            = 3
    THUMB_TIP           = 4
    INDEX_FINGER_MCP    = 5
    INDEX_FINGER_PIP    = 6
    INDEX_FINGER_DIP    = 7
    INDEX_FINGER_TIP    = 8
    MIDDLE_FINGER_MCP   = 9
    MIDDLE_FINGER_PIP   = 10
    MIDDLE_FINGER_DIP   = 11
    MIDDLE_FINGER_TIP   = 12
    RING_FINGER_MCP     = 13
    RING_FINGER_PIP     = 14
    RING_FINGER_DIP     = 15
    RING_FINGER_TIP     = 16
    PINKY_MCP           = 17
    PINKY_PIP           = 18
    PINKY_DIP           = 19
    PINKY_TIP           = 20


@dataclass
class HandData:
    """Satu tangan terdeteksi – 21 landmark dalam piksel."""
    handedness: str                       # 'Left' | 'Right'
    landmarks:  List[Tuple[int, int]]     # 21 titik (x, y)
    score:      float = 1.0

    def get_landmark(self, index: int) -> Tuple[int, int]:
        return self.landmarks[index]

    def get_fingertips(self) -> Dict[str, Tuple[int, int]]:
        return {
            "index":  self.landmarks[LandmarkIndex.INDEX_FINGER_TIP],
            "middle": self.landmarks[LandmarkIndex.MIDDLE_FINGER_TIP],
            "ring":   self.landmarks[LandmarkIndex.RING_FINGER_TIP],
            "pinky":  self.landmarks[LandmarkIndex.PINKY_TIP],
            "thumb":  self.landmarks[LandmarkIndex.THUMB_TIP],
        }


# ─────────────────────────────────────────────
#  Backend: Task API (mediapipe >= 0.10)
# ─────────────────────────────────────────────
class _TaskAPIBackend:
    """Wrapper MediaPipe Task API untuk >= 0.10."""

    def __init__(self, max_hands, detect_conf, track_conf, model_complexity):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        import urllib.request, os, tempfile

        # Download model lite jika belum ada
        model_path = os.path.join(tempfile.gettempdir(), "hand_landmarker.task")
        if not os.path.exists(model_path):
            print("[HandTracker] Mengunduh hand_landmarker.task (~9MB)...")
            url = (
                "https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )
            urllib.request.urlretrieve(url, model_path)
            print("[HandTracker] Download selesai.")

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detect_conf,
            min_hand_presence_confidence=detect_conf,
            min_tracking_confidence=track_conf,
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(options)
        self._ts_ms = 0   # timestamp simulasi (ms)

    def process(self, frame_rgb: np.ndarray, frame_shape) -> List[HandData]:
        from mediapipe.tasks.python import vision as mp_vision
        import mediapipe as mp

        h, w = frame_shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        self._ts_ms += 33   # ~30 fps simulasi
        result = self._detector.detect_for_video(mp_image, self._ts_ms)

        hands_out: List[HandData] = []
        if not result.hand_landmarks:
            return hands_out

        for lm_list, handedness_list in zip(result.hand_landmarks, result.handedness):
            # Konversi ke piksel
            landmarks_px = []
            for lm in lm_list:
                x = max(0, min(int(lm.x * w), w - 1))
                y = max(0, min(int(lm.y * h), h - 1))
                landmarks_px.append((x, y))

            # Task API: label sudah dari perspektif cermin (kebalikan solutions)
            raw_label = handedness_list[0].category_name   # 'Left' / 'Right'
            label     = "Right" if raw_label == "Left" else "Left"
            score     = handedness_list[0].score

            hands_out.append(HandData(handedness=label, landmarks=landmarks_px, score=score))

        return hands_out

    def close(self):
        self._detector.close()


# ─────────────────────────────────────────────
#  Backend: Solutions API (mediapipe < 0.10)
# ─────────────────────────────────────────────
class _SolutionsBackend:
    """Wrapper MediaPipe solutions API untuk < 0.10."""

    def __init__(self, max_hands, detect_conf, track_conf, model_complexity):
        _mp_hands = mp.solutions.hands
        self._hands = _mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=model_complexity,
            min_detection_confidence=detect_conf,
            min_tracking_confidence=track_conf,
        )

    def process(self, frame_rgb: np.ndarray, frame_shape) -> List[HandData]:
        h, w = frame_shape[:2]
        frame_rgb.flags.writeable = False
        results = self._hands.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if not results.multi_hand_landmarks:
            return []

        hands_out: List[HandData] = []
        for hand_lm, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            landmarks_px = []
            for lm in hand_lm.landmark:
                x = max(0, min(int(lm.x * w), w - 1))
                y = max(0, min(int(lm.y * h), h - 1))
                landmarks_px.append((x, y))

            raw_label = hand_info.classification[0].label
            label     = "Right" if raw_label == "Left" else "Left"
            score     = hand_info.classification[0].score

            hands_out.append(HandData(handedness=label, landmarks=landmarks_px, score=score))

        return hands_out

    def close(self):
        self._hands.close()


# ─────────────────────────────────────────────
#  HandTracker (public class)
# ─────────────────────────────────────────────
class HandTracker:
    """
    Deteksi tangan real-time. Kompatibel MediaPipe 0.9 & 0.10+.

    Cara pakai:
        tracker = HandTracker(max_hands=2)
        hands   = tracker.process(frame_bgr)
        tracker.draw_landmarks(frame_bgr, hands)
    """

    CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17),
    ]

    def __init__(
        self,
        max_hands:            int   = 2,
        detection_confidence: float = 0.7,
        tracking_confidence:  float = 0.6,
        model_complexity:     int   = 0,
    ):
        if _USE_TASK_API:
            self._backend = _TaskAPIBackend(
                max_hands, detection_confidence, tracking_confidence, model_complexity
            )
        else:
            self._backend = _SolutionsBackend(
                max_hands, detection_confidence, tracking_confidence, model_complexity
            )

    def process(self, frame_bgr: np.ndarray) -> List[HandData]:
        """Proses frame BGR → list HandData."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self._backend.process(frame_rgb, frame_bgr.shape)

    def draw_landmarks(
        self,
        frame:      np.ndarray,
        hands_data: List[HandData],
        dot_color:  Tuple[int,int,int] = (0, 255, 180),
        line_color: Tuple[int,int,int] = (255, 255, 255),
        dot_radius: int = 5,
        line_thick: int = 2,
    ) -> None:
        """Gambar skeleton + landmark di frame (in-place)."""
        for hand in hands_data:
            pts = hand.landmarks

            for s, e in self.CONNECTIONS:
                cv2.line(frame, pts[s], pts[e], line_color, line_thick, cv2.LINE_AA)

            for pt in pts:
                cv2.circle(frame, pt, dot_radius + 2, (20, 20, 20), -1, cv2.LINE_AA)
                cv2.circle(frame, pt, dot_radius,     dot_color,    -1, cv2.LINE_AA)

            wrist = pts[LandmarkIndex.WRIST]
            cv2.putText(
                frame,
                f"{hand.handedness} ({hand.score:.2f})",
                (wrist[0] - 30, wrist[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (220, 220, 50), 1, cv2.LINE_AA,
            )

    def release(self) -> None:
        """Bebaskan resource."""
        self._backend.close()
