# ai_test_run.py
# ----------------
# Simple, hard-coded YOLO screen-capture demo.
# - Press 'q' or ESC to quit.
# - Window is resizable; the display scales to fit while preserving aspect ratio.
# EDIT ME: model_path (your trained weights) and monitor box (your capture area).

import os
import time
import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO

# ==== [ EDIT ME ] ============================================================
# Path to your trained weights (hard-coded). Keep this relative to the script if possible.
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "runs", "detect", "train2", "weights", "best.pt")

# Screen region to capture (hard-coded for your setup).
monitor = {
    "top": 125,      # Y offset
    "left": 0,       # X offset
    "width": 2560,   # capture width
    "height": 1400   # capture height
}

# Window settings
WINDOW_NAME = "AI Vision"
INITIAL_WINDOW_W = 1280  # initial window width
INITIAL_WINDOW_H = 720   # initial window height

# If you want the window to appear on a second monitor,
# set MOVE_WINDOW = True and adjust SECOND_MONITOR_ORIGIN to your second monitor's top-left.
MOVE_WINDOW = False
SECOND_MONITOR_ORIGIN = (1920, 0)  # (x, y) where to place the window if MOVE_WINDOW is True
# ============================================================================


def resize_with_aspect(img, target_w, target_h):
    """
    Resize img to fit inside (target_w x target_h) while preserving aspect ratio.
    Adds letterbox padding if needed to exactly match the window size.
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0 or target_w <= 0 or target_h <= 0:
        return img

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create canvas and center the resized image
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def draw_boxes(img_bgr, results, class_names):
    """
    Draw YOLO results on BGR image.
    """
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = class_names.get(cls_id, str(cls_id))
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img_bgr,
            f"{label} {conf:.2f}",
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return img_bgr


def main():
    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = YOLO(model_path)

    # Setup screen grabber
    sct = mss()

    # Create a resizable window
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, INITIAL_WINDOW_W, INITIAL_WINDOW_H)
    if MOVE_WINDOW:
        cv2.moveWindow(WINDOW_NAME, SECOND_MONITOR_ORIGIN[0], SECOND_MONITOR_ORIGIN[1])

    print("Press 'q' or ESC to quit.")
    # Small delay to let the window appear
    time.sleep(0.05)

    # Main loop
    while True:
        # Grab frame
        shot = sct.grab(monitor)
        frame_bgra = np.array(shot)                     # BGRA
        frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

        # Inference expects RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = model.predict(frame_rgb, verbose=False)[0]

        # Draw detections back on BGR
        frame_drawn = draw_boxes(frame_bgr, results, model.names)

        # Get current window size (OpenCV 4.5.4+). Fallback to initial size if unavailable.
        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(WINDOW_NAME)
            if win_w <= 0 or win_h <= 0:
                win_w, win_h = INITIAL_WINDOW_W, INITIAL_WINDOW_H
        except Exception:
            win_w, win_h = INITIAL_WINDOW_W, INITIAL_WINDOW_H

        # Resize to window with letterboxing
        display_img = resize_with_aspect(frame_drawn, win_w, win_h)

        # Show
        cv2.imshow(WINDOW_NAME, display_img)

        # Quit on 'q' or ESC
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # 27 = ESC
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
