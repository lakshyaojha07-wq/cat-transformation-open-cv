"""
Gesture-Controlled Cat Switcher (Tasks API version)
====================================================

Same behavior as before: shows a base cat, switches to a specific cat photo
when your hand makes a matching gesture.

This version uses MediaPipe's newer "Tasks" API (HandLandmarker) instead of
the old `mp.solutions.hands` API, which has been removed from recent
MediaPipe releases (0.10.30+ / 1.0.x). If you're on an older MediaPipe
(with `mp.solutions` available), the previous script also works — but this
one is the future-proof version and is what you should use going forward.

Gestures -> cats (edit CATS config below to change any of this):
    Fist                          -> gesture_fist.png      (punching-cat)
    Index finger up, vertical     -> gesture_thinking.png  (smart-cat)
    Index finger up, tilted/horiz -> gesture_shush.jpg      (cat_shushing)
    Two hands, wrists together    -> gesture_prayer.jpg     (begging-cat)
    Two hands, index tips close   -> gesture_shy.jpg        (shy_cat)
    (no match / open hand)        -> base.jpg               (cat_staring)

Setup:
    pip install opencv-python mediapipe

First run downloads a small (~8MB) hand-tracking model file automatically
(needs an internet connection once).

Usage:
    python gesture_cat_switcher_v2.py
"""

import os
import math
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

# ---------------------------------------------------------------------------
# Config: which image file shows for which gesture
# ---------------------------------------------------------------------------
CATS_DIR = "cats"
BASE_IMAGE = "base.jpg"

GESTURE_IMAGES = {
    "fist": "gesture_fist.png",
    "thinking": "gesture_thinking.png",
    "shush": "gesture_shush.jpg",
    "prayer": "gesture_prayer.jpg",
    "shy": "gesture_shy.jpg",
}

DISPLAY_SIZE = (500, 500)
CAM_INDEX = 0

SMOOTHING_FRAMES = 4
TWO_HAND_WRIST_DIST = 0.18
TWO_HAND_FINGERTIP_DIST = 0.12
VERTICAL_ANGLE_MAX_DEG = 35

# ---------------------------------------------------------------------------
# Model file: auto-downloaded once on first run
# ---------------------------------------------------------------------------
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# Standard 21-point hand skeleton connections, for drawing only
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    print(f"Downloading hand-tracking model to {MODEL_PATH} (one-time, ~8MB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")
    except Exception as e:
        raise RuntimeError(
            f"Could not download the hand-tracking model automatically ({e}).\n"
            f"Download it yourself from:\n  {MODEL_URL}\n"
            f"and save it as '{MODEL_PATH}' next to this script."
        )


# ---------------------------------------------------------------------------
# Load cat images
# ---------------------------------------------------------------------------
def load_images():
    images = {}
    base_path = os.path.join(CATS_DIR, BASE_IMAGE)
    base_img = cv2.imread(base_path)
    if base_img is None:
        raise FileNotFoundError(f"Base image not found: {base_path}")
    images["base"] = cv2.resize(base_img, DISPLAY_SIZE)

    for key, fname in GESTURE_IMAGES.items():
        path = os.path.join(CATS_DIR, fname)
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: missing {path}, that gesture will fall back to base.")
            continue
        images[key] = cv2.resize(img, DISPLAY_SIZE)

    return images


# ---------------------------------------------------------------------------
# Finger-state helpers
# Note: with the Tasks API, a detected hand is just a plain list of 21
# landmark objects (each with .x, .y, .z) — no ".landmark" wrapper like the
# old API had.
# ---------------------------------------------------------------------------
def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def finger_states(lm):
    """lm: list of 21 landmarks for one hand. Returns [thumb, index, middle, ring, pinky]."""
    palm_ref = lm[17]
    thumb_tip, thumb_ip = lm[4], lm[3]
    thumb_extended = dist(thumb_tip, palm_ref) > dist(thumb_ip, palm_ref) * 1.1

    states = [thumb_extended]
    tip_ids = [8, 12, 16, 20]
    pip_ids = [6, 10, 14, 18]
    for tip_i, pip_i in zip(tip_ids, pip_ids):
        states.append(lm[tip_i].y < lm[pip_i].y)
    return states


def index_angle_from_vertical(lm):
    mcp, tip = lm[5], lm[8]
    dx = tip.x - mcp.x
    dy = mcp.y - tip.y
    return math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))


def classify_gesture(hands_data):
    """hands_data: list of (landmarks, finger_states) tuples, 0-2 entries."""
    if len(hands_data) == 2:
        (lm_a, fs_a), (lm_b, fs_b) = hands_data
        wrist_a, wrist_b = lm_a[0], lm_b[0]

        if dist(wrist_a, wrist_b) < TWO_HAND_WRIST_DIST:
            if sum(fs_a[1:]) >= 3 and sum(fs_b[1:]) >= 3:
                return "prayer"

        index_only_a = fs_a[1] and not any(fs_a[2:])
        index_only_b = fs_b[1] and not any(fs_b[2:])
        if index_only_a and index_only_b:
            if dist(lm_a[8], lm_b[8]) < TWO_HAND_FINGERTIP_DIST:
                return "shy"

        return "base"

    if len(hands_data) == 1:
        lm, fs = hands_data[0]
        thumb, index, middle, ring, pinky = fs

        if not any([index, middle, ring, pinky]):
            return "fist"

        if index and not any([middle, ring, pinky]):
            angle = index_angle_from_vertical(lm)
            return "thinking" if angle < VERTICAL_ANGLE_MAX_DEG else "shush"

        return "base"

    return "base"


def draw_landmarks(frame, lm, w, h):
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    ensure_model()
    images = load_images()

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        num_hands=2,
        running_mode=RunningMode.VIDEO,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    detector = HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    current_gesture = "base"
    candidate_gesture = "base"
    candidate_count = 0
    start_time = time.time()

    print("Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - start_time) * 1000)

        result = detector.detect_for_video(mp_image, timestamp_ms)

        hands_data = []
        for lm in result.hand_landmarks:
            draw_landmarks(frame, lm, w, h)
            hands_data.append((lm, finger_states(lm)))

        gesture = classify_gesture(hands_data)

        if gesture == candidate_gesture:
            candidate_count += 1
        else:
            candidate_gesture = gesture
            candidate_count = 1

        if candidate_count >= SMOOTHING_FRAMES:
            current_gesture = candidate_gesture

        display_key = current_gesture if current_gesture in images else "base"
        cv2.putText(frame, f"Gesture: {current_gesture}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Webcam (hand tracking)", frame)
        cv2.imshow("Cat", images[display_key])

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
