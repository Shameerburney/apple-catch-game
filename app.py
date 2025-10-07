import av
import cv2
import numpy as np
import mediapipe as mp
import time
import random
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# ---------------- CONFIG ----------------
CAM_WIDTH = 480   # lower res for speed
CAM_HEIGHT = 360
APPLE_RADIUS = 18
CUP_WIDTH = 140
CUP_HEIGHT = 28
GRAVITY = 0.6
SPAWN_INTERVAL = 1.0

mp_hands = mp.solutions.hands

# ----------------------------------------
def get_index_pos(hand_landmarks, w, h):
    return int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)


# ---------------- CLASS -----------------
class AppleGame(VideoProcessorBase):
    def __init__(self):
        self.apples = []
        self.last_spawn = time.time()
        self.score = 0
        self.cup_x = CAM_WIDTH // 2
        self.cup_y = CAM_HEIGHT - 60
        self.smooth_x = self.cup_x
        self.frame_count = 0
        self.prev_time = time.time()

        # ✅ Faster model setup
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3,
            model_complexity=0  # SPEED BOOST
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        # Only process every 3rd frame for speed
        self.frame_count += 1
        if self.frame_count % 3 == 0:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                ix, iy = get_index_pos(hand, w, h)
                target_x = ix
                # Smooth movement to reduce jitter
                self.smooth_x = int(self.smooth_x + (target_x - self.smooth_x) * 0.25)
                self.cup_x = self.smooth_x

        # Spawn new apples
        now = time.time()
        if now - self.last_spawn > SPAWN_INTERVAL:
            self.last_spawn = now
            self.apples.append({
                "x": random.randint(APPLE_RADIUS, CAM_WIDTH - APPLE_RADIUS),
                "y": -APPLE_RADIUS,
                "vy": random.uniform(2.0, 4.0)
            })

        # Update apples
        for a in self.apples:
            a["y"] += a["vy"]
            a["vy"] += GRAVITY * 0.03

        # Check catches or misses
        caught = []
        for i, a in enumerate(self.apples):
            if a["y"] > CAM_HEIGHT + APPLE_RADIUS:
                caught.append(i)
            elif a["y"] >= self.cup_y - APPLE_RADIUS and abs(a["x"] - self.cup_x) < CUP_WIDTH // 2:
                self.score += 1
                caught.append(i)

        for idx in sorted(caught, reverse=True):
            self.apples.pop(idx)

        # Draw everything
        overlay = img  # no copy for speed
        for a in self.apples:
            cv2.circle(overlay, (int(a["x"]), int(a["y"])), APPLE_RADIUS, (0, 0, 255), -1)
            cv2.circle(overlay, (int(a["x"]), int(a["y"] - APPLE_RADIUS - 6)), 4, (0, 255, 0), -1)

        # Draw cup
        cup_x1 = int(self.cup_x - CUP_WIDTH // 2)
        cup_x2 = int(self.cup_x + CUP_WIDTH // 2)
        cup_y1 = int(self.cup_y - CUP_HEIGHT // 2)
        cup_y2 = int(self.cup_y + CUP_HEIGHT // 2)
        cv2.rectangle(overlay, (cup_x1, cup_y1), (cup_x2, cup_y2), (200, 180, 50), -1)
        cv2.rectangle(overlay, (cup_x1, cup_y1), (cup_x2, cup_y2), (0, 0, 0), 2)

        # Score
        cv2.putText(overlay, f"Score: {self.score}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)

        # FPS counter
        now = time.time()
        fps = 1 / (now - self.prev_time)
        self.prev_time = now
        cv2.putText(overlay, f"FPS: {int(fps)}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Small delay to stabilize FPS
        time.sleep(0.03)
        return av.VideoFrame.from_ndarray(overlay, format="bgr24")


# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="🍎 Apple Catch Game", layout="centered")
st.title("🍎 Apple Catch Game")
st.markdown("**Move your index finger left and right to catch falling apples!**")

webrtc_streamer(
    key="apple-game",
    video_processor_factory=AppleGame,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)
