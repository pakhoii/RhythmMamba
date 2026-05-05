import cv2
import torch
import numpy as np
import time
from collections import deque
from scipy.signal import butter, filtfilt, welch
from neural_methods.model.RhythmMamba import RhythmMamba

# ======================
# CONFIG
# ======================
# MODEL_PATH = "PreTrainedModels/UBFC_cross_RhythmMamba.pth"
# MODEL_PATH = "PreTrainedModels/UBFC_SizeW128_SizeH128_ClipLength160_DataTypeStandardized_DataAugNone_LabelTypeStandardized_Crop_faceTrue_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse/RhythmMambaVer2_Epoch25.pth"
MODEL_PATH = "PreTrainedModels/UBFC_SizeW128_SizeH128_ClipLength160_DataTypeStandardized_DataAugNone_LabelTypeStandardized_Crop_faceTrue_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse/RhythmMambaVer3_Epoch27.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FRAME_BUFFER = 160
INFER_STRIDE = 15
BPM_SMOOTH = 5

# ======================
# LOAD MODEL
# ======================
model = RhythmMamba().to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()
print(f"✓ Model loaded on {DEVICE}")

# ======================
# POST-PROCESS
# ======================
def _detrend(input_signal, lambda_value=100):
    from scipy.sparse import spdiags
    signal_length = input_signal.shape[0]
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, signal_length - 2, signal_length).toarray()
    return np.dot((H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)

def get_hr(y, sr=30, min_bpm=45, max_bpm=150):
    p, q = welch(y, sr, nfft=int(1e5/sr), nperseg=np.min((len(y)-1, 256)))
    mask = (p > min_bpm/60) & (p < max_bpm/60)
    if np.sum(mask) == 0:
        return 0.0
    return float(p[mask][np.argmax(q[mask])] * 60)

def estimate_hr(ppg, fs=30):
    """Pipeline chính xác như repo gốc."""
    try:
        sig = _detrend(ppg, 100)
        b, a = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
        sig = filtfilt(b, a, np.double(sig))
        return get_hr(sig, sr=fs)
    except Exception as e:
        print(f"⚠ HR estimation failed: {e}")
        return 0.0

# ======================
# FACE DETECTOR
# ======================
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ======================
# WEBCAM LOOP
# ======================
cap = cv2.VideoCapture(0)

# Lấy FPS từ camera
camera_fps = cap.get(cv2.CAP_PROP_FPS)
if camera_fps <= 0:
    camera_fps = 30.0

frames = deque(maxlen=FRAME_BUFFER)
times = deque(maxlen=FRAME_BUFFER)

tracked_bbox = None
bpm = 0.0
bpm_hist = deque(maxlen=BPM_SMOOTH)
inference_counter = 0
detection_interval = 30  # Re-detect mặt mỗi 30 frames

print(f"✓ Camera FPS: {camera_fps}")
print(f"✓ Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    current_time = time.time()
    h_frame, w_frame = frame.shape[:2]

    # 1. FACE DETECTION
    if inference_counter % detection_interval == 0 or tracked_bbox is None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            tracked_bbox = faces[0]
        else:
            tracked_bbox = None

    # 2. EXTRACT & PREPROCESS FRAME
    if tracked_bbox is not None:
        x, y, w, h = tracked_bbox
        
        # Clamp to frame bounds
        x = max(0, min(x, w_frame - w))
        y = max(0, min(y, h_frame - h))
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        face_crop = frame[y:y+h, x:x+w]

        if face_crop.size > 0:
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (128, 128))
            
            # ✅ FIX: Normalize từng frame riêng (giống training)
            face_resized = face_resized.astype(np.float32) / 255.0
            mean = face_resized.mean()
            std = face_resized.std()
            face_resized = (face_resized - mean) / (std + 1e-7)

            frames.append(face_resized)
            times.append(current_time)

    # 3. INFERENCE
    if len(frames) == FRAME_BUFFER:
        inference_counter += 1

        if inference_counter % INFER_STRIDE == 0:
            # ✅ Prepare batch
            clip = np.array(frames, dtype=np.float32)  # [160, 128, 128, 3]
            clip = np.transpose(clip, (0, 3, 1, 2))     # [160, 3, 128, 128]
            clip = np.expand_dims(clip, 0)              # [1, 160, 3, 128, 128]
            clip = torch.from_numpy(clip).float().to(DEVICE)

            # ✅ Run model
            with torch.no_grad():
                ppg = model(clip)  # [1, 160]

            ppg = ppg.squeeze().cpu().numpy()  # [160]

            # ✅ FIX: Normalize output (giống training line 74)
            ppg = (ppg - np.mean(ppg)) / (np.std(ppg) + 1e-7)

            # ✅ Estimate HR
            raw_bpm = estimate_hr(ppg, fs=camera_fps)
            
            if raw_bpm > 0 and 40 <= raw_bpm <= 200:  # Sanity check
                bpm_hist.append(raw_bpm)
                bpm = float(np.median(bpm_hist))
                print(f"Raw BPM: {raw_bpm:.1f} | Smoothed: {bpm:.1f}")

    # 4. DISPLAY
    color = (30, 200, 100) if bpm > 0 else (100, 100, 100)
    cv2.putText(frame, f"HR: {bpm:.1f} BPM" if bpm > 0 else "Detecting...",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    cv2.putText(frame, f"Buffer: {len(frames)}/{FRAME_BUFFER}",
                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 50), 2)
    cv2.putText(frame, f"FPS: {camera_fps:.1f}",
                (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 150, 200), 2)

    cv2.imshow("RhythmMamba HR Monitor", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
print("✓ Done")