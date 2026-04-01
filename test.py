import cv2
import torch
import numpy as np
import time
from scipy.signal import butter, filtfilt, welch
from neural_methods.model.RhythmMamba import RhythmMamba

# ======================
# CONFIG
# ======================
MODEL_PATH = "PreTrainedModels/PURE_cross_RhythmMamba.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FRAME_BUFFER = 160

# ======================
# LOAD MODEL
# ======================
model = RhythmMamba().to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()

# ======================
# FILTER (Chuẩn RhythmMamba: Bậc 2, 0.75 - 2.5 Hz)
# ======================
def bandpass(signal, low, high, fs):
    nyq = 0.5 * fs
    low /= nyq
    high /= nyq
    # Dùng bậc 2 (order=2) theo đúng paper
    b, a = butter(2, [low, high], btype='band')
    return filtfilt(b, a, signal)

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

frames = []
times = []
tracked_bbox = None
bpm = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    
    # 1. Khóa Bounding Box: Chỉ dò mặt ở frame ĐẦU TIÊN của mỗi chu kỳ
    if tracked_bbox is None or len(frames) == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            # Ưu tiên lấy khuôn mặt to nhất (tránh nhận diện nhầm chi tiết nền)
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            tracked_bbox = faces[0]
        else:
            tracked_bbox = None

    # 2. Xử lý Trích xuất Frame (Nằm ngoài vòng lặp for)
    if tracked_bbox is not None:
        x, y, w, h = tracked_bbox
        
        # Vẽ khung để quan sát
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Cắt khuôn mặt
        face_crop = frame[y:y+h, x:x+w]
        
        # Tránh lỗi crash nếu bbox lọt ra ngoài viền màn hình
        if face_crop.size > 0: 
            # QUAN TRỌNG: Chuyển BGR thành RGB
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (128, 128))
            
            frames.append(face_resized)
            times.append(current_time)

    # 3. Đạt đủ 160 frames -> Tiến hành Inference
    if len(frames) == FRAME_BUFFER:
        
        # Tính toán FPS thực tế (dynamic FPS)
        elapsed_time = times[-1] - times[0]
        actual_fps = (FRAME_BUFFER - 1) / elapsed_time if elapsed_time > 0 else 30.0

        # Chuẩn bị Tensor (1, 160, 3, 128, 128)
        clip = np.array(frames, dtype=np.float32) / 255.0
        clip = clip.transpose(0, 3, 1, 2)
        clip = np.expand_dims(clip, 0)
        clip = torch.from_numpy(clip).to(DEVICE)

        # Chạy Model
        with torch.no_grad():
            ppg = model(clip)
        
        ppg = ppg.squeeze().cpu().numpy()

        # Tiền xử lý tín hiệu: Bộ lọc Butterworth 0.75 - 2.5 Hz
        filtered = bandpass(ppg, 0.75, 2.5, actual_fps)

        # Ước tính HR bằng thuật toán Welch (Power Spectral Density)
        freqs, psd = welch(filtered, fs=actual_fps, nperseg=len(filtered))
        bpm = freqs[np.argmax(psd)] * 60

        # Reset lại buffer và bbox để đo chu kỳ 160 frames tiếp theo
        frames = []
        times = []
        tracked_bbox = None

    # 4. Hiển thị thông tin lên màn hình
    if bpm > 0:
        cv2.putText(frame, f"HR: {bpm:.1f} BPM", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    cv2.putText(frame, f"Buffer: {len(frames)}/{FRAME_BUFFER}", (20, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("RhythmMamba HR Monitor", frame)

    if cv2.waitKey(1) & 0xFF == 27: # Nhấn ESC để thoát
        break

cap.release()
cv2.destroyAllWindows()