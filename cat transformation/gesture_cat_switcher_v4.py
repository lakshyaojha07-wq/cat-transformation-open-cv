"""
Gesture-Controlled Cat Switcher (Tasks API version, v4)
========================================================

Changes from v3, based on real-world testing:
- Two new gestures added: finger gun -> hamster photo, OK/scuba sign ->
  scuba cat photo.
- Prayer gesture no longer also requires a specific finger-extended count.
  When your palms press together, one hand partially blocks the camera's
  view of the other, which makes finger-by-finger detection unreliable —
  so prayer now just checks "two hands, wrists close together."
- A short grace period now covers brief hand-tracking dropouts: if your
  hand is briefly not detected for a frame or two, the last gesture is
  kept instead of immediately snapping back to the base cat. Only a
  longer gap resets to base.
- Detection confidence thresholds lowered slightly to help hands get
  picked up more consistently.

Gestures -> cats:
    Fist                              -> gesture_fist.png    (punching-cat)
    Index + middle extended (peace)    -> gesture_thinking.png (smart-cat)
    Index only extended                -> gesture_shush.jpg   (cat_shushing)
    Thumb + index extended (finger gun) -> gesture_gun.jpg     (hamster)
    Thumb+index tips touching, other
      three fingers up (OK/scuba sign) -> gesture_scuba.jpg   (scuba cat)
    Two hands, wrists together          -> gesture_prayer.jpg  (begging-cat)
    Two hands, index tips close         -> gesture_shy.jpg     (shy_cat)
    (no match / open hand)              -> base.jpg            (cat_staring)

Setup:
    pip install opencv-python mediapipe

First run downloads a small (~8MB) hand-tracking model file automatically
(needs an internet connection once).

Usage:
    python gesture_cat_switcher_v4.py
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
    "gun": "gesture_gun.jpg",
    "scuba": "gesture_scuba.jpg",
}

DISPLAY_SIZE = (500, 500)
CAM_INDEX = 0

SMOOTHING_FRAMES = 4
TWO_HAND_WRIST_DIST = 0.25          # loosened further - prayer no longer
                                     # also checks finger-extended count
TWO_HAND_FINGERTIP_DIST = 0.20
CURL_RATIO = 1.15
OK_SIGN_TIP_DIST = 0.07             # max thumb-tip-to-index-tip distance
                                     # to count as an "OK sign" circle
MISS_FRAMES_BEFORE_RESET = 12       # ~0.4s at 30fps: how long a hand can
                                     # briefly vanish before falling back
                                     # to the base cat
SHOW_DEBUG = True

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
    """3D distance (uses landmark z too). More rotation-invariant than a
    plain 2D/y-only comparison, which is what broke fist detection when
    the hand was rotated toward the camera."""
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def finger_states(lm):
    """lm: list of 21 landmarks for one hand. Returns [thumb, index, middle, ring, pinky].

    A finger counts as "extended" when its tip is farther from the wrist
    than its own middle knuckle (pip) is. A curled finger's tip folds back
    in toward the wrist, so this ratio holds regardless of how the whole
    hand is rotated in front of the camera — unlike comparing raw
    y-coordinates, which only works when the hand is upright.
    """
    wrist = lm[0]
    tip_ids = [4, 8, 12, 16, 20]
    pip_ids = [3, 6, 10, 14, 18]  # thumb uses its IP joint here
    states = []
    for tip_i, pip_i in zip(tip_ids, pip_ids):
        states.append(dist(lm[tip_i], wrist) > dist(lm[pip_i], wrist) * CURL_RATIO)
    return states


def classify_gesture(hands_data):
    """hands_data: list of (landmarks, finger_states) tuples, 0-2 entries."""
    if len(hands_data) == 2:
        (lm_a, fs_a), (lm_b, fs_b) = hands_data
        wrist_a, wrist_b = lm_a[0], lm_b[0]

        # Prayer: just "hands close together". No finger-pattern check,
        # since palms pressed together occlude each other and make
        # per-finger detection unreliable.
        if dist(wrist_a, wrist_b) < TWO_HAND_WRIST_DIST:
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
        thumb_index_tip_dist = dist(lm[4], lm[8])

        # OK/scuba sign: thumb+index tips touching (forms a circle),
        # other three fingers extended. Checked first since the
        # circle-forming thumb/index don't reliably read as "extended" or
        # "curled" by the wrist-distance heuristic.
        if middle and ring and pinky and thumb_index_tip_dist < OK_SIGN_TIP_DIST:
            return "scuba"

        if not any([index, middle, ring, pinky]):
            return "fist"

        # Finger gun: thumb + index out, everything else curled
        if thumb and index and not middle and not ring and not pinky:
            return "gun"

        if index and middle and not ring and not pinky:
            return "thinking"  # peace-sign shape: index + middle

        if index and not thumb and not any([middle, ring, pinky]):
            return "shush"  # index only, thumb tucked in

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
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    current_gesture = "base"
    candidate_gesture = "base"
    candidate_count = 0
    miss_count = 0
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

        if hands_data:
            miss_count = 0
            gesture = classify_gesture(hands_data)

            if gesture == candidate_gesture:
                candidate_count += 1
            else:
                candidate_gesture = gesture
                candidate_count = 1

            if candidate_count >= SMOOTHING_FRAMES:
                current_gesture = candidate_gesture
        else:
            # No hand seen this frame. Don't immediately reset to base —
            # a brief tracking dropout is normal. Only fall back after
            # MISS_FRAMES_BEFORE_RESET consecutive empty frames.
            miss_count += 1
            if miss_count >= MISS_FRAMES_BEFORE_RESET:
                current_gesture = "base"
                candidate_gesture = "base"
                candidate_count = 0

        if SHOW_DEBUG:
            y = 60
            for i, (lm, fs) in enumerate(hands_data):
                labels = ["thumb", "index", "middle", "ring", "pinky"]
                pattern = " ".join(f"{l}:{int(v)}" for l, v in zip(labels, fs))
                cv2.putText(frame, f"hand{i}: {pattern}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
                y += 22
                if len(hands_data) == 1:
                    ti_dist = dist(lm[4], lm[8])
                    cv2.putText(frame, f"thumb-index tip dist: {ti_dist:.3f} (scuba < {OK_SIGN_TIP_DIST})",
                                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
                    y += 22
            if len(hands_data) == 2:
                wrist_d = dist(hands_data[0][0][0], hands_data[1][0][0])
                tip_d = dist(hands_data[0][0][8], hands_data[1][0][8])
                cv2.putText(frame, f"wrist_dist: {wrist_d:.3f} (prayer < {TWO_HAND_WRIST_DIST})",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
                y += 22
                cv2.putText(frame, f"tip_dist: {tip_d:.3f} (shy < {TWO_HAND_FINGERTIP_DIST})",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
                y += 22
            if not hands_data:
                cv2.putText(frame, f"no hand (miss {miss_count}/{MISS_FRAMES_BEFORE_RESET})",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1)
                y += 22

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
