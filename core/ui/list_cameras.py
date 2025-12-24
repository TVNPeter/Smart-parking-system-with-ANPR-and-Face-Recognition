from __future__ import annotations

import argparse
import time
import cv2

BACKENDS = {
    "default": 0,  # let OpenCV choose
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
}


def try_open(index: int, backend: int, timeout_s: float = 1.0) -> bool:
    try:
        cap = cv2.VideoCapture(index, backend) if backend != 0 else cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            return False
        t0 = time.time()
        ok, _ = cap.read()
        while not ok and (time.time() - t0) < timeout_s:
            ok, _ = cap.read()
        cap.release()
        return bool(ok)
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe available camera indices on Windows")
    ap.add_argument("--max", type=int, default=10, help="Max index to probe (exclusive)")
    ap.add_argument("--backend", choices=list(BACKENDS.keys()), default="default", help="OpenCV backend to use")
    args = ap.parse_args()

    backend_flag = BACKENDS[args.backend]
    print(f"Probing indices 0..{args.max-1} with backend '{args.backend}'")
    found = []
    for i in range(args.max):
        ok = try_open(i, backend_flag)
        print(f"  index {i}: {'OK' if ok else 'unavailable'}")
        if ok:
            found.append(i)
    print("\nAvailable indices:")
    if found:
        print("  ", ", ".join(str(x) for x in found))
    else:
        print("  (none)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
