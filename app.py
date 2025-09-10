import cv2
import numpy as np
import mediapipe as mp
import time
import random
import streamlit as st

CAM_WIDTH = 640
CAM_HEIGHT = 480
APPLE_RADIUS = 18
CUP_WIDTH = 140
CUP_HEIGHT = 28
GRAVITY = 0.6
SPAWN_INTERVAL = 1.0

mp_hands = mp.solutions.hands

def get_index_pos(hand_landmarks, w, h):
    return int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)

def run_game():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    apples = []
    last_spawn = time.time()
    score = 0
    cup_x = CAM_WIDTH // 2
    cup_y = CAM_HEIGHT - 60
    smooth_x = cup_x

    frame_placeholder = st.empty()
    st.write("🎮 Press **Stop** to end the game.")

    with mp_hands.Hands(max_num_hands=1,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            h, w, _ = frame.shape

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                ix, iy = get_index_pos(hand, w, h)
                target_x = ix
                smooth_x = int(smooth_x + (target_x - smooth_x) * 0.25)
                cup_x = smooth_x

            now = time.time()
            if now - last_spawn > SPAWN_INTERVAL:
                last_spawn = now
                apples.append({
                    "x": random.randint(APPLE_RADIUS, CAM_WIDTH - APPLE_RADIUS),
                    "y": -APPLE_RADIUS,
                    "vy": random.uniform(2.0, 4.0)
                })

            for a in apples:
                a["y"] += a["vy"]
                a["vy"] += GRAVITY * 0.03

            caught = []
            for i, a in enumerate(apples):
                if a["y"] > CAM_HEIGHT + APPLE_RADIUS:
                    caught.append(i)
                else:
                    if a["y"] >= cup_y - APPLE_RADIUS and abs(a["x"] - cup_x) < CUP_WIDTH // 2:
                        score += 1
                        caught.append(i)

            for idx in sorted(caught, reverse=True):
                apples.pop(idx)

            overlay = frame.copy()
            for a in apples:
                cv2.circle(overlay, (int(a["x"]), int(a["y"])), APPLE_RADIUS, (0,0,255), -1)
                cv2.circle(overlay, (int(a["x"]), int(a["y"] - APPLE_RADIUS - 6)), 4, (0,255,0), -1)

            cup_x1 = int(cup_x - CUP_WIDTH // 2)
            cup_x2 = int(cup_x + CUP_WIDTH // 2)
            cup_y1 = int(cup_y - CUP_HEIGHT // 2)
            cup_y2 = int(cup_y + CUP_HEIGHT // 2)
            cv2.rectangle(overlay, (cup_x1, cup_y1), (cup_x2, cup_y2), (200,180,50), -1)
            cv2.rectangle(overlay, (cup_x1, cup_y1), (cup_x2, cup_y2), (0,0,0), 2)
            cv2.putText(overlay, f"Score: {score}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (50,50,50), 2)

            frame_placeholder.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

    cap.release()

st.title("🍎 Apple Catch Game")
if st.button("Start Game"):
    run_game()
