# tools/env_check.py
# Quick import/version check for all dependencies.

import importlib
import sys

pkgs = [
    "torch",
    "ultralytics",
    "cv2",
    "numpy",
    "mss",
    "keyboard",
    "pyautogui",
    "yaml",
]

def main():
    print("Python:", sys.version)
    for p in pkgs:
        try:
            m = importlib.import_module(p)
            ver = getattr(m, "__version__", "unknown")
            print(f"[OK] {p} - {ver}")
        except Exception as e:
            print(f"[FAIL] {p} - {e}")

    try:
        import torch
        print("CUDA available:", torch.cuda.is_available())
    except Exception:
        pass

if __name__ == "__main__":
    main()
