# 🖐 Hand Gesture Filter
**Aplikasi Computer Vision real-time** – deteksi dua tangan dari webcam,  
buat area interaktif antar jari, dan aktifkan filter visual dengan gerakan tangan.

---

## 📁 Struktur Folder

```
hand_gesture_cv/
│
├── main.py                 # Entry point – webcam loop & rendering
├── hand_tracker.py         # Deteksi tangan & landmark extraction (MediaPipe)
├── filter_manager.py       # Registry & penerapan filter OpenCV
├── interaction_manager.py  # Polygon area, point-in-polygon, overlay
│
├── requirements.txt        # Dependency Python
└── README.md               # Dokumentasi ini
```

---

## 📦 Dependency

| Library | Versi Min | Fungsi |
|---|---|---|
| `opencv-python` | 4.8.0 | Capture webcam, drawing, filter image |
| `mediapipe` | 0.10.0 | Hand landmark detection (21 titik/tangan) |
| `numpy` | 1.24.0 | Array computing, mask, polygon |

---

## 🏗 Arsitektur

```
main.py
  │
  ├─→ HandTracker          (hand_tracker.py)
  │     └─ MediaPipe Hands → 21 landmark per tangan → HandData
  │
  ├─→ InteractionManager   (interaction_manager.py)
  │     ├─ update_areas()  → hitung polygon A/B/C dari posisi jari
  │     ├─ check_interactions() → point-in-polygon detection
  │     ├─ get_area_masks() → binary mask untuk compositing
  │     └─ draw_areas()    → overlay polygon + label
  │
  └─→ FilterManager        (filter_manager.py)
        ├─ apply_all_active() → compositing filter dengan mask
        └─ Registry: BLUR, GRAYSCALE, EDGE, PIXELATE, SEPIA
```

### Alur Data per Frame

```
Webcam Frame
    │
    ▼
HandTracker.process()
    │ → List[HandData]  (maks 2 tangan)
    ▼
InteractionManager.update_areas()     ← polygon ikuti jari tangan kanan
    │
InteractionManager.check_interactions() ← jari tangan kiri masuk area?
    │ → Dict[area_id → bool]
    ▼
FilterManager.apply_all_active()      ← compositing filter di mask area aktif
    │ → frame dengan efek
    ▼
HandTracker.draw_landmarks()          ← skeleton tangan
    │
InteractionManager.draw_areas()       ← overlay polygon + label
    │
draw_hud()                            ← FPS, status, shortcut
    │
cv2.imshow()
```

---

## 🖐 Area Filter

```
       Tangan Kanan (terbuka ke depan kamera)

  INDEX  MIDDLE  RING  PINKY
    │      │      │      │
    │  [A] │  [B] │  [C] │
    │      │      │      │
   MCP    MCP    MCP    MCP

  Area A (Biru)  = celah INDEX  ↔ MIDDLE  → Gaussian Blur
  Area B (Hijau) = celah MIDDLE ↔ RING    → Grayscale
  Area C (Merah) = celah RING   ↔ PINKY   → Canny Edge Detection
```

Polygon setiap area dibentuk dari **4 titik**:
```
[TIP_kiri, TIP_kanan, MCP_kanan, MCP_kiri]
```
→ trapezoid yang mengikuti bentuk celah jari secara dinamis.

---

## ▶️ Instalasi & Menjalankan

### 1. Clone / unduh proyek
```bash
# Jika dari git:
git clone <repo_url>
cd hand_gesture_cv

# Atau ekstrak ZIP dan masuk ke folder
```

### 2. Buat virtual environment (direkomendasikan)
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate
```

### 3. Install dependency
```bash
pip install -r requirements.txt
```

### 4. Jalankan aplikasi
```bash
python main.py
```

> **Catatan**: Pastikan webcam tidak dipakai aplikasi lain.  
> Jika kamera index 0 tidak jalan, ganti `CAMERA_INDEX = 1` di `main.py`.

---

## ⌨️ Keyboard Shortcut

| Tombol | Fungsi |
|---|---|
| `Q` / `ESC` | Keluar dari aplikasi |
| `S` | Swap tangan primer/sekunder (kanan ↔ kiri) |
| `H` | Toggle HUD (tampilan informasi) |
| `F` | Toggle semua filter on/off |
| `R` | Reset – nonaktifkan semua filter |
| `1` | Intensitas filter rendah (50%) |
| `2` | Intensitas filter sedang (75%) |
| `3` | Intensitas filter penuh (100%) |

---

## 🎯 Cara Pakai

1. Buka kedua tangan di depan kamera
2. **Tangan kanan**: buka dan rentangkan jari → area A/B/C muncul di antara jari
3. **Tangan kiri**: gerakkan jari telunjuk ke dalam salah satu area
   - Masuk Area A → Gaussian Blur aktif di area tersebut
   - Masuk Area B → Grayscale aktif
   - Masuk Area C → Canny Edge aktif
4. Area aktif ditandai dengan polygon lebih terang + indikator hijau

---

## 🔧 Menambah Filter Baru

```python
# Di filter_manager.py, tambah fungsi:
def filter_my_custom(frame: np.ndarray) -> np.ndarray:
    # ... efek kustom Anda ...
    return result

# Daftarkan di __init__ FilterManager:
self._registry["MY_FILTER"] = filter_my_custom

# Map ke area di AREA_FILTER_MAP:
AREA_FILTER_MAP = {
    "A": "MY_FILTER",   # ganti filter Area A
    ...
}
```

---

## 🔧 Menambah Area Baru

```python
# Di interaction_manager.py:
AREA_FINGER_PAIRS["D"] = (
    LandmarkIndex.THUMB_TIP,
    LandmarkIndex.INDEX_FINGER_TIP,
    LandmarkIndex.THUMB_MCP,
    LandmarkIndex.INDEX_FINGER_MCP,
)

AREA_STYLES["D"] = {
    "color": (200, 50, 200),
    "alpha": 0.30,
    "label": "AREA D  [NEW]",
}
```

---

## ⚡ Tips Performa

- **Model complexity 0** sudah cukup akurat untuk real-time; jangan naikkan ke 1 kecuali perlu
- **Resolusi 1280×720** adalah default; turunkan ke `640×480` jika FPS < 20
- **Buffer kamera = 1** (`CAP_PROP_BUFFERSIZE`) mengurangi latency
- Polygon mask di-**cache** – hanya dihitung ulang jika posisi tangan berubah
- `frame.flags.writeable = False` sebelum `hands.process()` mengurangi copy internal MediaPipe

---

## 🐛 Troubleshooting

| Masalah | Solusi |
|---|---|
| Kamera tidak terbuka | Coba `CAMERA_INDEX = 1` atau `2` di `Config` |
| FPS rendah | Turunkan resolusi ke 640×480, atau set `MODEL_COMPLEXITY = 0` |
| Tangan tidak terdeteksi | Pastikan pencahayaan cukup; naikkan `DETECT_CONFIDENCE = 0.5` |
| Area tidak muncul | Tangan primer (default: kanan) harus terlihat penuh di kamera |
| Import error | Pastikan virtual env aktif dan `pip install -r requirements.txt` sudah dijalankan |

---

## 📄 Lisensi

MIT License – bebas digunakan dan dimodifikasi.
