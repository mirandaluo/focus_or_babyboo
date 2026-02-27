import cv2
import numpy as np
import mediapipe as mp
import os
import time
import subprocess
import tkinter as tk

#video render
try:
    from PIL import Image, ImageTk
    pil_available = True
except Exception:
    pil_available = False

#ultralytics YOLO for phone detection but highkey doesnt work!! 
try:
    from ultralytics import YOLO
    ultralytics_available = True
except Exception:
    ultralytics_available = False

#settings
video_path = "/Users/mirluo29/Downloads/virus.mp4" #change my user to yours if ur gonna use this
pitch_down_threshold = 20.0 # lower = more sensitive
turn_away_threshold = 40.0 # degrees yaw away from camera
hysteresis = 6.0 #prevents flicker
caption_text = "credit for this idea: @fruitydumpsterxoxo on tt"
cam_index = 0

#detection tuning for face/eye signals
eye_closed_ear = 0.18 #smaller = more sensitive
gaze_down_ratio = 0.62 #higher = more sensitive to looking down
eyes_closed_seconds = 4.0 #trigger if eyes closed this long
gaze_down_frames_required = 3 #require gaze down for N consecutive frames
phone_detect_every = 1   # un phone detection every N frames
phone_confidence = 0.15

#YOLO phone detection (doesn't work :()
yolo_model_path = "yolov8s.pt" #larger model = better detection
draw_phone_boxes = True
use_rect_fallback = True
rect_min_area_ratio = 0.01 #fraction of frame area
rect_max_area_ratio = 0.85
rect_aspect_min = 0.30
rect_aspect_max = 3.5
rect_canny_low = 30
rect_canny_high = 120

#reminder window and audio playback controls
reminder_topmost = True
reminder_scale = 1.0  #1.0 = original video size
use_audio = True
ffplay_path = "ffplay"  #make sure ffplay is in path
use_tkinter_window = False  #set True only if Tk works
#mediapipe the goat
#face mesh for head pose and eye landmarks
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
#generic face model points for head pose
model_points = np.array([
    (0.0,   0.0,   0.0),  #nose tip
    (0.0, -63.6, -12.5), #chin
    (-43.3, 32.7, -26.0), #left eye outer corner
    (43.3,  32.7, -26.0),#right eye outer corner
    (-28.9,-28.9, -24.1),#left mouth corner
    (28.9, -28.9, -24.1)#right mouth corner
], dtype=np.float64)

landmark_ids = [1, 152, 33, 263, 61, 291] #nose, chin, left eye, right eye, left mouth, right mouth

def rotation_to_euler(R):
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2]) #pitch
        y = np.arctan2(-R[2, 0], sy) #yaw
        z = np.arctan2(R[1, 0], R[0, 0]) #roll
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.degrees(x), np.degrees(y), np.degrees(z)

#eye landmark for ear and iris ratio.
left_eye_inner = 133
left_eye_outer = 33
left_eye_up = 159
left_eye_down = 145
right_eye_inner = 362
right_eye_outer = 263
right_eye_up = 386
right_eye_down = 374
left_iris = [468, 469, 470, 471, 472]
right_iris = [473, 474, 475, 476, 477]

def _eye_aspect_ratio(lm, w, h, left=True):
    if left:
        p1 = lm[left_eye_outer]
        p2 = lm[left_eye_inner]
        p3 = lm[left_eye_up]
        p4 = lm[left_eye_down]
    else:
        p1 = lm[right_eye_outer]
        p2 = lm[right_eye_inner]
        p3 = lm[right_eye_up]
        p4 = lm[right_eye_down]

    p1 = np.array([p1.x * w, p1.y * h])
    p2 = np.array([p2.x * w, p2.y * h])
    p3 = np.array([p3.x * w, p3.y * h])
    p4 = np.array([p4.x * w, p4.y * h])

    eye_width = np.linalg.norm(p1 - p2)
    eye_height = np.linalg.norm(p3 - p4)
    if eye_width <= 1e-6:
        return 0.0
    return float(eye_height / eye_width)

def _gaze_down_ratio(lm, w, h, left=True):
    if left:
        iris_pts = left_iris
        upper = left_eye_up
        lower = left_eye_down
    else:
        iris_pts = right_iris
        upper = right_eye_up
        lower = right_eye_down

    iris = np.array([(lm[i].x * w, lm[i].y * h) for i in iris_pts], dtype=np.float32)
    iris_center = iris.mean(axis=0)
    upper_pt = np.array([lm[upper].x * w, lm[upper].y * h], dtype=np.float32)
    lower_pt = np.array([lm[lower].x * w, lm[lower].y * h], dtype=np.float32)

    eye_height = np.linalg.norm(lower_pt - upper_pt)
    if eye_height <= 1e-6:
        return 0.5
    ratio = np.linalg.norm(iris_center - upper_pt) / eye_height
    return float(ratio)


def get_attention_signals(frame_bgr):
    """Returns pitch_down, yaw_deg, gaze_down, eyes_closed based on a single face."""
    #run nediapipe face mesh to get landmarks
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None, None, False, False

    lm = result.multi_face_landmarks[0].landmark
    #map selected landmarks into pixel coordinates for solvePnP
    image_points = np.array([(lm[i].x * w, lm[i].y * h) for i in landmark_ids], dtype=np.float64)

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    #estimate head pose from 2D->3D
    ok, rvec, tvec = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not ok:
        return None, None, False, False

    #rotate vec to euler
    R, _ = cv2.Rodrigues(rvec)
    pitch, yaw, roll = rotation_to_euler(R)

    #flip sign so "looking down" is usually positive
    pitch_down = -pitch
    pitch_down = float(pitch_down)

    #eye aspect ratio for blink/closed-eye detection
    left_ear = _eye_aspect_ratio(lm, w, h, left=True)
    right_ear = _eye_aspect_ratio(lm, w, h, left=False)
    ear = (left_ear + right_ear) * 0.5
    eyes_closed = ear < eye_closed_ear

    #iris position ratio for gaze down detection.
    left_gaze = _gaze_down_ratio(lm, w, h, left=True
    right_gaze = _gaze_down_ratio(lm, w, h, left=False)
    gaze_ratio = (left_gaze + right_gaze) * 0.5
    gaze_down = gaze_ratio > gaze_down_ratio

    return pitch_down, float(yaw), gaze_down, eyes_closed

def detect_large_rectangle(frame_bgr):
    """large rectangle meant to mimic phone but doesnt work :("""
    if not use_rect_fallback:
        return False, []
    h, w = frame_bgr.shape[:2]
    frame_area = float(h * w)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, rect_canny_low, rect_canny_high)
    edges = cv2.dilate(edges, None, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < frame_area * rect_min_area_ratio or area > frame_area * rect_max_area_ratio:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        x, y, bw, bh = cv2.boundingRect(approx)
        if bh == 0 or bw == 0:
            continue
        aspect = bw / float(bh)
        if aspect < rect_aspect_min or aspect > rect_aspect_max:
            continue
        boxes.append(((x, y, x + bw, y + bh), area / frame_area))
    return (len(boxes) > 0), boxes

def draw_caption(frame, text):
    """Overlay the caption bar on the reminder video frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = 10
    y = frame.shape[0] - 10
    #background box
    cv2.rectangle(frame, (x - 5, y - th - 5), (x + tw + 5, y + 5), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

class ReminderWindow:
    """Tkinter window for showing reminder frames on top of other apps."""
    def __init__(self, title="focus RIGHT NOW!", topmost=True):
        if not pil_available:
            raise RuntimeError("PIL (Pillow) is required for the reminder window.")

        self.should_quit = False
        self.root = tk.Tk()
        self.root.title(title)
        self.root.attributes("-topmost", bool(topmost))
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<Escape>", self._on_key)
        self.root.bind("q", self._on_key)
        self.root.bind("f", self._on_key)
        self.label = tk.Label(self.root, bg="black")
        self.label.pack()
        self._photo = None

    def _on_close(self):
        self.should_quit = True

    def _on_key(self, _event):
        self.should_quit = True

    def update_frame(self, frame_bgr):
        frame = frame_bgr
        if reminder_scale != 1.0:
            frame = cv2.resize(
                frame,
                (int(frame.shape[1] * reminder_scale), int(frame.shape[0] * reminder_scale)),
                interpolation=cv2.INTER_AREA
            )

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        self._photo = ImageTk.PhotoImage(img)
        self.label.configure(image=self._photo)
        self.root.update_idletasks()
        self.root.update()

    def destroy(self):
        try:
            self.root.destroy()
        except Exception:
            pass

def play_video_frame(video_cap, window_name=None):
    ok, frame = video_cap.read()
    if not ok or frame is None:
        return None

    draw_caption(frame, caption_text)
    if window_name:
        cv2.imshow(window_name, frame)
    return frame

def start_audio():
    if not use_audio:
        return None
    try:
        return subprocess.Popen(
            [ffplay_path, "-loglevel", "quiet", "-nodisp", "-loop", "-1", video_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except Exception:
        return None

def stop_audio(proc):
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=1.0)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


class PhoneDetector:#doesnt work!!
    def __init__(self):
        self.available = False
        self.model = None
        if ultralytics_available and os.path.exists(yolo_model_path):
            try:
                self.model = YOLO(yolo_model_path)
                self.available = True
            except Exception:
                self.available = False

    def detect_phone(self, frame_bgr):
        if not self.available:
            return False, []
        results = self.model.predict(frame_bgr, conf=phone_confidence, classes=[67], verbose=False)
        boxes_out = []
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes:
                    xyxy = b.xyxy[0].cpu().numpy().astype(int)
                    conf = float(b.conf[0].cpu().numpy())
                    boxes_out.append((xyxy, conf))
        return (len(boxes_out) > 0), boxes_out


def main():
    """main loop: webcam -> detect distraction -> show reminder video"""
    if not os.path.exists(video_path):
        print(f'Video file not found at: "{video_path}"')
        print('Fix video_path to the correct location (ex: /Users/.../Downloads/virus.mp4).')
        return
    if not pil_available:
        print("Pillow is required for the reminder window. Install with: pip install pillow")
        return

    #cam
    cam = cv2.VideoCapture(cam_index)
    if not cam.isOpened():
        print("Could not open webcam. Check permissions or try cam_index=1.")
        return

    distracted = False
    video_cap = None
    reminder_window = None
    audio_proc = None
    # Optional phone detector (YOLO).
    phone_detector = PhoneDetector()
    if not phone_detector.available:
        print("Phone detection(lwk doesnt work but pls do these steps anyways: install ultralytics and download yolov8s.pt")

    frame_idx = 0
    last_phone = False
    phone_boxes = []
    rect_boxes = []
    eyes_closed_since = None
    gaze_down_frames = 0

    should_quit = False

    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                break

            pitch_down, yaw_deg, gaze_down, eyes_closed = get_attention_signals(frame)

            #update temporal filters to avoid blink false positives
            if eyes_closed:
                if eyes_closed_since is None:
                    eyes_closed_since = time.time()
            else:
                eyes_closed_since = None

            if gaze_down:
                gaze_down_frames += 1
            else:
                gaze_down_frames = 0

            eyes_closed_active = False
            if eyes_closed_since is not None:
                eyes_closed_active = (time.time() - eyes_closed_since) >= eyes_closed_seconds
            gaze_down_active = gaze_down_frames >= gaze_down_frames_required

            #decide distracted/focused with hysteresis
            if pitch_down is None or yaw_deg is None:
                new_distracted = False
            else:
                pitch_trigger = pitch_down > (pitch_down_threshold - hysteresis if distracted else pitch_down_threshold)
                gaze_trigger = gaze_down_active
                eyes_trigger = eyes_closed_active
                yaw_trigger = abs(yaw_deg) > (turn_away_threshold - hysteresis if distracted else turn_away_threshold)
                #require 2 of 3 signals to reduce false positives
                signal_count = int(pitch_trigger) + int(gaze_trigger) + int(eyes_trigger)
                new_distracted = signal_count >= 2
                if yaw_trigger:
                    new_distracted = True

            #jhone detection  using YOLO
            if phone_detector.available:
                if frame_idx % phone_detect_every == 0:
                    last_phone, phone_boxes = phone_detector.detect_phone(frame)
                if last_phone:
                    new_distracted = True
            else:
                phone_boxes = []

            #rectangle fallback (runs when YOLO misses), might be even worse tho.
            rect_detected = False
            if frame_idx % phone_detect_every == 0:
                rect_detected, rect_boxes = detect_large_rectangle(frame)
            if rect_detected:
                new_distracted = True

            # start video when distracted
            if new_distracted and not distracted:
                distracted = True

                if video_cap is not None:
                    video_cap.release()
                    video_cap = None

                video_cap = cv2.VideoCapture(video_path)
                if not video_cap.isOpened():
                    print(f'OpenCV: Could not open video file: "{video_path}"')
                    video_cap.release()
                    video_cap = None
                    distracted = False
                else:
                    if use_tkinter_window:
                        reminder_window = ReminderWindow(topmost=reminder_topmost)
                    else:
                        cv2.namedWindow("REMINDER", cv2.WINDOW_NORMAL)
                        try:
                            cv2.setWindowProperty("REMINDER", cv2.WND_PROP_TOPMOST, 1)
                        except cv2.error:
                            pass
                    audio_proc = start_audio()

            #stop video when refocused
            elif (not new_distracted) and distracted:
                distracted = False
                if video_cap is not None:
                    video_cap.release()
                    video_cap = None
                stop_audio(audio_proc)
                audio_proc = None
                if reminder_window is not None:
                    reminder_window.destroy()
                    reminder_window = None

            #debug texts and cam prev
            preview = frame.copy()
            if pitch_down is not None:
                cv2.putText(preview, f"pitch_down: {pitch_down:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if yaw_deg is not None:
                cv2.putText(preview, f"yaw: {yaw_deg:.1f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(preview, f"gaze_down: {gaze_down}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(preview, f"eyes_closed: {eyes_closed}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(preview, f"phone: {last_phone}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if draw_phone_boxes and phone_boxes:
                for (xyxy, conf) in phone_boxes:
                    x1, y1, x2, y2 = xyxy
                    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 200, 255), 2)
                    cv2.putText(preview, f"phone {conf:.2f}", (x1, max(0, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            if draw_phone_boxes and rect_boxes:
                for (xyxy, score) in rect_boxes:
                    x1, y1, x2, y2 = xyxy
                    cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 180, 0), 2)
                    cv2.putText(preview, f"rect {score:.2f}", (x1, max(0, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)
            cv2.putText(preview, f"state: {'DISTRACTED' if distracted else 'FOCUSED'}", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("WEBCAM", preview)

            #reminder video loop
            if distracted and video_cap is not None:
                frame_out = play_video_frame(video_cap, window_name=None if use_tkinter_window else "REMINDER")
                if frame_out is None:
                    #loop video from start while still distracted
                    video_cap.release()
                    video_cap = cv2.VideoCapture(video_path)
                    if not video_cap.isOpened():
                        print(f'OpenCV: Could not reopen video file: "{video_path}"')
                        video_cap.release()
                        video_cap = None
                        distracted = False
                else:
                    if reminder_window is not None:
                        reminder_window.update_frame(frame_out)
                        if reminder_window.should_quit:
                            should_quit = True
            #quit
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('f'), 27):
                should_quit = True
            if should_quit:
                break
            frame_idx += 1
    finally:
        cam.release()
        if video_cap is not None:
            video_cap.release()
        stop_audio(audio_proc)
        if reminder_window is not None:
            reminder_window.destroy()
        if not use_tkinter_window:
            try:
                cv2.destroyWindow("focus RIGHT NOW!!")
            except cv2.error:
                pass
        cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
