#!/usr/bin/env python3
"""Recompute NIQE after fixing the input scaling (0-255 -> 0-1).

Only the NIQE value is touched; FID/FVD/SSIM/... are kept as computed.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path.home() / "projects"))
import quality_metrics as qm  # noqa: E402
from benchmark_worker import refresh_report  # noqa: E402


def mouth_niqe(video_path, engine):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / 3.0)))
    grays = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            engine.detect(frame)
            face = qm.crop_face(frame, engine.last_bbox, 256, margin=1.0)
            grays.append(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY))
        idx += 1
    cap.release()
    return qm.compute_niqe(grays)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", action="append", required=True)
    ap.add_argument("--force", action="store_true",
                    help="recompute even when the stored value already looks sane")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    total = updated = skipped = failed = 0
    for outdir in args.outdir:
        folder = Path(outdir)
        print(f"=== {folder} ===", flush=True)
        for qjson in sorted(folder.glob("*/_quality/*.json")):
            total += 1
            if args.limit and total > args.limit:
                break
            try:
                data = json.loads(qjson.read_text())
            except Exception as exc:
                print(f"skip unreadable {qjson}: {exc}")
                skipped += 1
                continue
            old = (data.get("metrics") or {}).get("niqe")
            if (not args.force) and old is not None and float(old) < 50:
                skipped += 1
                continue
            video = data.get("output_video")
            if not video or not Path(video).exists():
                print(f"skip missing output for {qjson}")
                skipped += 1
                continue
            try:
                engine = qm.FaceEngine()
                value, detail = mouth_niqe(video, engine)
                data.setdefault("metrics", {})["niqe"] = qm.rnd(value, 4)
                data.setdefault("details", {})["niqe"] = detail
                data["niqe_recomputed"] = qm.now_iso()
                qm.atomic_write_json(qjson, data)
                updated += 1
                print(f"{qjson.parent.parent.name}/{qjson.stem}: "
                      f"{old} -> {data['metrics']['niqe']}", flush=True)
            except Exception as exc:
                failed += 1
                print(f"FAIL {qjson}: {type(exc).__name__}: {exc}", flush=True)

        try:
            refresh_report(folder)
        except Exception as exc:  # noqa: BLE001
            print(f"refresh_report failed: {exc}")

    print(f"DONE total={total} updated={updated} skipped={skipped} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
