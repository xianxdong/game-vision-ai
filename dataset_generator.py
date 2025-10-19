# dataset_generator.py
# --------------------
# EDIT ME: Set your dataset path and capture region for YOUR screen.

import time
import os
import keyboard
from pathlib import Path

import mss
import numpy as np
import cv2

# ==== [ EDIT ME: Where to save images ] ====
# Example (Windows absolute path). You can also use a relative path like: dataset_root = "dataset"
dataset_root = r"C:/Users/Xiang/Desktop/Projects/starve.io-ai/dataset"

# ==== [ EDIT ME: Screen capture region ] ====
# Use mss coordinates: top/left are the upper-left corner; width/height are the size to capture.
monitor = {
    "top": 125,     # Adjust for your game window
    "left": 0,
    "width": 2560,
    "height": 1400
}

# ==== Behavior settings (fine to keep as-is) ====
capture_key = "p"       # press to capture a frame
quit_key = "f2"         # press to quit
cooldown = 0.30         # seconds to wait after a capture
jpeg_quality = 95       # 1..100

# YOLO-style folder layout
train_dir = Path(dataset_root) / "images" / "train"
val_dir = Path(dataset_root) / "images" / "val"
train_dir.mkdir(parents=True, exist_ok=True)
val_dir.mkdir(parents=True, exist_ok=True)

def next_index(*folders: Path) -> int:
    """Return the next 0-based index so we don't overwrite existing frames."""
    max_found = -1
    for folder in folders:
        for name in os.listdir(folder):
            if name.lower().startswith("frame_") and name.lower().endswith(".jpg"):
                try:
                    n = int(name.split("_")[1].split(".")[0])
                    max_found = max(max_found, n)
                except Exception:
                    pass
    return max_found + 1  # next number

def main():
    # Validate monitor box a bit
    if monitor["width"] <= 0 or monitor["height"] <= 0:
        raise ValueError("Monitor width/height must be positive. Edit the monitor dict at the top of the file.")

    # Start numbering after any existing frames
    count = next_index(train_dir, val_dir)

    try:
        sct = mss.mss()
    except Exception as error:
        print("[ERROR] Could not initialize screen capture (mss).", error)
        print("Tip: Run this script with proper permissions, and make sure another app isn't blocking capture.")
        return

    print("Capturing to:")
    print(f"  {train_dir}")
    print(f"  {val_dir}")
    print(f"Hotkeys: capture='{capture_key}', quit='{quit_key}'")
    print("NOTE: If you clone this repo, edit dataset_root and monitor at the top for your own setup.")
    print("Ready. Press 'p' to capture, 'F2' to stop.")

    last_capture = 0.0
    while True:
        if keyboard.is_pressed(quit_key):
            print("Stopped screenshot capture.")
            break

        if keyboard.is_pressed(capture_key):
            now = time.time()
            # Simple debounce so holding P doesn’t rapid-fire hundreds of frames
            if now - last_capture >= cooldown:
                print("Grabbing screenshot...")
                shot = sct.grab(monitor)
                img = np.array(shot)# BGRA
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                # ~80/20 split: every 5th frame goes to val
                folder = val_dir if (count % 5 == 0) else train_dir
                out_path = folder / f"frame_{count:04}.jpg"

                ok = cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
                if ok:
                    print(f"Saved {out_path}")
                    count += 1
                else:
                    print(f"[WARN] Failed to save {out_path}")

                last_capture = now

        # Light sleep to avoid busy-waiting
        time.sleep(0.01)

if __name__ == "__main__":
    main()
