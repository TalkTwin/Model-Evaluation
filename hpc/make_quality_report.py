#!/usr/bin/env python3
"""make_quality_report.py - build the separate quality-metrics HTML report.

Reads two benchmark output folders:
  * cross-audio / dubbing runs (real video + new voice mp3)
  * reconstruction runs (real video + its own audio)
and writes ONE self-contained HTML page with, per duration:
  input video + 4 model videos side by side + quality/system stats underneath.

Usage:
  python make_quality_report.py \
      --cross-dir ~/projects/outputs/bench-20261031-152743 \
      --recon-dir ~/projects/outputs/bench-20261101-001509 \
      --out ~/projects/outputs/bench-20261031-152743/quality_report.html
"""
import argparse
import html
import json
from pathlib import Path

MODELS = ["wav2lip", "heygem", "musetalk", "latentsync"]
MODEL_LABEL = {
    "wav2lip": "Wav2Lip",
    "heygem": "HeyGem",
    "musetalk": "MuseTalk",
    "latentsync": "LatentSync",
}
BADGE = {
    "wav2lip": "badge-wav2lip",
    "heygem": "badge-heygem",
    "musetalk": "badge-musetalk",
    "latentsync": "badge-latentsync",
}

QUALITY_ROWS = [
    ("lse_d", "LSE-D ↓", "lse"),
    ("lse_c", "LSE-C ↑", "lse"),
    ("csim", "CSIM ↑", "id"),
    ("fid", "FID ↓", "gen"),
    ("fvd", "FVD ↓", "gen"),
    ("ssim", "SSIM ↑", "rec"),
    ("psnr", "PSNR ↑", "rec"),
    ("lpips", "LPIPS ↓", "rec"),
    ("lmd_mouth", "LMD ↓", "rec"),
    ("niqe", "NIQE ↓", "nr"),
]

SYSTEM_ROWS = [
    ("wall_s", "Wall", "s"),
    ("rtf", "RTF", "x"),
    ("peak_vram_mb", "VRAM peak", "MB"),
    ("peak_rss_mb", "RAM peak", "MB"),
    ("out_size_mb", "Output size", "MB"),
    ("out_duration_s", "Output length", "s"),
]


def load_records(folder):
    folder = Path(folder)
    rp = folder / "report.json"
    records = {}
    if rp.exists():
        try:
            data = json.loads(rp.read_text())
            for rec in data.get("records") or []:
                records[(rec.get("model"), rec.get("tag"))] = rec
        except Exception:
            pass
    if not records:
        for js in sorted(folder.glob("*/*.json")):
            if js.parent.name.startswith("_"):
                continue
            try:
                rec = json.loads(js.read_text())
            except Exception:
                continue
            if rec.get("status") == "COMPLETED":
                records[(rec.get("model"), rec.get("tag"))] = rec
    return records


def attach_quality(folder, records):
    folder = Path(folder)
    for (model, tag), rec in records.items():
        qp = folder / str(model) / "_quality" / f"{tag}.json"
        if qp.exists():
            try:
                q = json.loads(qp.read_text())
                rec["quality"] = {"metrics": q.get("metrics") or {},
                                  "errors": q.get("errors") or {},
                                  "runtime_s": q.get("runtime_s"),
                                  "device": q.get("device")}
            except Exception:
                pass


def fmt(value, digits=2, suffix=""):
    if value is None or value == "":
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if abs(value) >= 100:
        text = f"{value:.0f}"
    elif abs(value) >= 10:
        text = f"{value:.1f}"
    else:
        text = f"{value:.{digits}f}"
    return text + suffix


def duration_value(rec):
    try:
        return float(((rec.get("input") or {}).get("audio") or {}).get("duration_s"))
    except (TypeError, ValueError):
        return None


def duration_label(seconds, mode):
    if seconds is None:
        return "unknown"
    near = round(seconds, 1)
    if near >= 60:
        return f"{near:.1f} s (1 min)"
    return f"{near:.0f} s"


def sort_key(rec):
    return duration_value(rec) or 0


def metric_value(rec, key):
    q = (rec.get("quality") or {}).get("metrics") or {}
    return q.get(key)


def best_values(recs, key):
    values = []
    for rec in recs:
        value = metric_value(rec, key)
        if value is not None:
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                pass
    if not values:
        return None
    lower_better = key in ("lse_d", "fid", "fvd", "lpips", "lmd_mouth", "niqe")
    return min(values) if lower_better else max(values)


RECON_ONLY = {"ssim", "psnr", "lpips", "lmd_mouth"}


def metric_valid(key, mode):
    return not (mode == "cross" and key in RECON_ONLY)


def metric_table(rec, best, mode):
    q = (rec.get("quality") or {}).get("metrics") or {}
    errors = (rec.get("quality") or {}).get("errors") or {}
    rows = ['<tr class="group-row"><th colspan="2">Quality</th></tr>']
    for key, label, group in QUALITY_ROWS:
        if not metric_valid(key, mode):
            rows.append(f"<tr><th>{label}</th><td class='na' "
                        f"title='not valid for cross-audio'>n/a</td></tr>")
            continue
        value = q.get(key)
        cls = ""
        if value is not None and best.get(key) is not None:
            try:
                if abs(float(value) - float(best[key])) < 1e-9:
                    cls = "best"
            except (TypeError, ValueError):
                pass
        note = ""
        if value is None and key in errors:
            note = f'<span class="err" title="{html.escape(errors[key])}">n/a</span>'
        rows.append(
            f"<tr><th>{label}</th><td class='{cls}'>{fmt(value)} {note}</td></tr>")
    rows.append(f"<tr><th>MOS (1–5)</th><td class='manual'>manual</td></tr>")
    return "".join(rows)


def system_table(rec):
    peaks = rec.get("peaks") or {}
    out = rec.get("output") or {}
    values = {
        "wall_s": rec.get("wall_s"),
        "rtf": rec.get("rtf"),
        "peak_vram_mb": peaks.get("vram_mb"),
        "peak_rss_mb": peaks.get("cpu_rss_mb"),
        "out_size_mb": out.get("size_mb"),
        "out_duration_s": out.get("duration_s"),
    }
    rows = ['<tr class="group-row"><th colspan="2">System</th></tr>']
    for key, label, unit in SYSTEM_ROWS:
        value = values.get(key)
        if key == "wall_s":
            text = fmt(value, suffix=" s")
        elif key == "rtf":
            text = fmt(value, suffix="x")
        elif key in ("peak_vram_mb", "peak_rss_mb"):
            text = fmt(value / 1024, digits=1, suffix=" GB") if value is not None else "—"
        elif key == "out_size_mb":
            text = fmt(value, suffix=" MB")
        elif key == "out_duration_s":
            text = fmt(value, suffix=" s")
        else:
            text = fmt(value)
        rows.append(f"<tr><th>{label}</th><td>{text}</td></tr>")
    return "".join(rows)


def card(rec, mode, best, video_src):
    model = rec.get("model")
    tag = rec.get("tag")
    label = MODEL_LABEL.get(model, model)
    badge = BADGE.get(model, "")
    status = rec.get("status")
    resolution = (rec.get("output") or {}).get("resolution") or "—"
    fps = (rec.get("output") or {}).get("fps")
    duration = (rec.get("output") or {}).get("duration_s")
    errors = (rec.get("quality") or {}).get("errors") or {}
    err_count = len([k for k in errors if k not in ("fatal",)])
    err_html = (f'<span class="warn" title="{html.escape(str(errors))}">'
                f'{err_count} metric warning(s)</span>' if err_count else "")
    return f"""
      <div class="model-card">
        <div class="model-head">
          <span class="badge {badge}">{label}</span>
          <span class="tag">{html.escape(str(tag))}</span>
          <span class="status {str(status).lower()}">{html.escape(str(status))}</span>
        </div>
        <video controls preload="metadata" src="{html.escape(video_src)}"></video>
        <div class="video-meta">{html.escape(str(resolution))} · {fmt(fps,1)} fps · {fmt(duration,2)} s</div>
        <table class="stats">
          <tbody>
            {metric_table(rec, best, mode)}
            {system_table(rec)}
          </tbody>
        </table>
        {err_html}
      </div>"""


GROUP_META = [
    ("lse", "Lip-sync", "group-lse"),
    ("id", "Identity", "group-id"),
    ("gen", "Generation", "group-gen"),
    ("rec", "Reconstruction", "group-rec"),
    ("nr", "No-reference", "group-nr"),
    ("sys", "System", "group-sys"),
]
SYSTEM_HEADERS = [("wall_s", "Wall"), ("rtf", "RTF"), ("vram", "VRAM")]


def comparison_table(recs, mode):
    counts = {}
    for _, _, group in QUALITY_ROWS:
        counts[group] = counts.get(group, 0) + 1
    best = {key: best_values(recs, key) for key, _, _ in QUALITY_ROWS}

    group_cells = []
    for key, label, cls in GROUP_META:
        span = counts.get(key) or (len(SYSTEM_HEADERS) if key == "sys" else 0)
        if span:
            group_cells.append(f'<th colspan="{span}" class="{cls}">{label}</th>')
    sub_cells = [f"<th>{label}</th>" for _, label, _ in QUALITY_ROWS]
    sub_cells += [f"<th>{label}</th>" for _, label in SYSTEM_HEADERS]

    rows = []
    for rec in recs:
        model = rec.get("model")
        q = (rec.get("quality") or {}).get("metrics") or {}
        cells = [f'<td class="model-cell"><span class="badge {BADGE.get(model,"")}">'
                 f'{MODEL_LABEL.get(model,model)}</span></td>']
        for key, _, _ in QUALITY_ROWS:
            if not metric_valid(key, mode):
                cells.append("<td class='na' title='not valid for cross-audio'>n/a</td>")
                continue
            value = q.get(key)
            cls = ""
            if value is not None and best.get(key) is not None:
                try:
                    if abs(float(value) - float(best[key])) < 1e-9:
                        cls = "best"
                except (TypeError, ValueError):
                    pass
            cells.append(f"<td class='{cls}'>{fmt(value)}</td>")
        cells.append(f"<td>{fmt(rec.get('wall_s'), suffix=' s')}</td>")
        cells.append(f"<td>{fmt(rec.get('rtf'), suffix='x')}</td>")
        vram = (rec.get("peaks") or {}).get("vram_mb")
        cells.append(f"<td>{fmt(vram / 1024, digits=1, suffix=' GB') if vram is not None else '—'}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")

    return ("<div class='table-wrap'><table class='compare'><thead>"
            "<tr class='group-row'><th rowspan='2' class='model-head'>Model</th>"
            + "".join(group_cells)
            + "</tr><tr>" + "".join(sub_cells) + "</tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></div>"
            + "<div class='table-legend'>Green pill = best value in this comparison. "
            "↓ lower is better · ↑ higher is better · n/a = not valid for this mode. "
            "Reconstruction metrics are only meaningful for reconstruction runs.</div>")


def anchor_for(mode, seconds):
    label = duration_label(seconds, mode)
    slug = str(label).replace(" ", "").replace("(", "").replace(")", "")
    return f"{mode}-{slug}"


def duration_section(mode, seconds, recs, input_video, video_prefix, input_label):
    label = duration_label(seconds, mode)
    anchor = anchor_for(mode, seconds)
    best = {key: best_values(recs, key) for key, _, _ in QUALITY_ROWS}
    audio = ((recs[0].get("input") or {}).get("audio") or {}) if recs else {}
    audio_name = Path(str(audio.get("path") or "")).name
    audio_dur = audio.get("duration_s")
    mode_note = ("reconstruction: the video's own audio was used; SSIM / PSNR / "
                 "LPIPS / LMD are valid." if mode == "reconstruction" else
                 "cross-audio / dubbing: a new voice track was used; only "
                 "no-reference metrics are valid (SSIM / PSNR / LPIPS / LMD = n/a).")
    cards = []
    for rec in sorted(recs, key=lambda r: MODELS.index(r.get("model")) if r.get("model") in MODELS else 99):
        model = rec.get("model")
        tag = rec.get("tag")
        src = f"{video_prefix}/{model}/{tag}.mp4"
        cards.append(card(rec, mode, best, src))
    return f"""
    <section id="{anchor}" class="duration-block">
      <div class="duration-head">
        <div>
          <span class="mode-pill {mode}">{'Reconstruction' if mode == 'reconstruction' else 'Cross-audio'}</span>
          <h2>{label}</h2>
        </div>
        <div class="dur-meta">{html.escape(audio_name)} · {fmt(audio_dur,2)} s audio</div>
      </div>
      <p class="mode-note">{mode_note}</p>
      <div class="input-row">
        <div class="input-card">
          <div class="input-title">📥 Input video <span>(same for all models)</span></div>
          <video controls preload="metadata" src="{html.escape(input_video)}"></video>
          <div class="video-meta">{html.escape(input_label)}</div>
        </div>
        <div class="audio-card">
          <div class="input-title">🎧 Driving audio</div>
          <div class="audio-name">{html.escape(audio_name)}</div>
          <div class="audio-dur">{fmt(audio_dur,2)} s</div>
          <p>All four models below used this exact audio file and the same input video.</p>
        </div>
      </div>
      <div class="cards">{''.join(cards)}</div>
      <h3>Comparison — {label}</h3>
      {comparison_table(recs, mode)}
    </section>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cross", action="append", nargs=2, metavar=("DIR", "PREFIX"),
                    default=[], help="cross-audio output folder + relative video prefix")
    ap.add_argument("--recon", action="append", nargs=2, metavar=("DIR", "PREFIX"),
                    default=[], help="reconstruction output folder + relative video prefix")
    ap.add_argument("--out", required=True)
    ap.add_argument("--input-video", default="inputs/marathi-demo-2/marathi-demo-2.mp4")
    ap.add_argument("--input-label",
                    default="marathi-demo-2.mp4 · 15.02 s · 1280×720 · 24 fps")
    args = ap.parse_args()

    sections = []          # (mode, seconds, records, video_prefix)
    total = 0

    def by_duration(records):
        groups = {}
        for rec in records.values():
            if rec.get("status") != "COMPLETED":
                continue
            sec = duration_value(rec)
            if sec is None:
                continue
            groups.setdefault(round(sec, 1), []).append(rec)
        return groups

    for mode, items in (("reconstruction", args.recon), ("cross", args.cross)):
        for folder, prefix in items:
            records = load_records(folder)
            attach_quality(folder, records)
            groups = by_duration(records)
            for sec in sorted(groups):
                recs = sorted(
                    groups[sec],
                    key=lambda r: MODELS.index(r.get("model"))
                    if r.get("model") in MODELS else 99)
                sections.append((mode, sec, recs, prefix))
                total += len(recs)

    nav = []
    for mode, sec, recs, prefix in sections:
        kind = "Recon" if mode == "reconstruction" else "Dubbing"
        nav.append(f'<a href="#{anchor_for(mode, sec)}">{kind} {duration_label(sec, mode)}</a>')

    body = [
        duration_section(mode, sec, recs, args.input_video, prefix, args.input_label)
        for mode, sec, recs, prefix in sections
    ]
    recon_sections = "".join(b for (s, b) in zip(sections, body)
                             if s[0] == "reconstruction")
    cross_sections = "".join(b for (s, b) in zip(sections, body)
                             if s[0] == "cross")

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Lip-Sync Quality Metrics — Video + Numbers</title>
<style>
  :root {{
    --bg:#f5f7fa; --card:#fff; --ink:#22303f; --muted:#65758b;
    --line:#dfe6ee; --blue:#2563eb; --green:#15803d; --amber:#b45309;
    --red:#b91c1c; --violet:#7c3aed;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
         font-family:"Segoe UI",system-ui,-apple-system,sans-serif; line-height:1.5; }}
  .wrap {{ max-width:1740px; margin:0 auto; padding:22px; }}
  header.hero {{ background:linear-gradient(135deg,#0f172a 0%,#1e3a8a 55%,#6d28d9 100%);
                 color:#fff; border-radius:16px; padding:32px 38px; margin-bottom:18px; }}
  header.hero h1 {{ margin:0 0 6px; font-size:clamp(1.5rem,3vw,2.2rem); }}
  header.hero p {{ margin:0; opacity:.9; }}
  .hero-links {{ margin-top:16px; display:flex; flex-wrap:wrap; gap:10px; }}
  .hero-links a {{ color:#fff; text-decoration:none; border:1px solid rgba(255,255,255,.35);
                   padding:7px 13px; border-radius:999px; font-size:.88rem; font-weight:600; }}
  .hero-links a:hover {{ background:rgba(255,255,255,.14); }}
  .meta-strip {{ display:flex; flex-wrap:wrap; gap:18px; background:#111827; color:#d1d5db;
                 border-radius:10px; padding:12px 18px; font-size:.86rem; margin-bottom:18px; }}
  .meta-strip b {{ color:#fff; }}
  nav.toc {{ position:sticky; top:0; z-index:10; background:rgba(255,255,255,.94);
             backdrop-filter:blur(8px); border:1px solid var(--line); border-radius:10px;
             padding:10px 14px; display:flex; flex-wrap:wrap; gap:8px; margin-bottom:22px; }}
  nav.toc a {{ color:var(--blue); text-decoration:none; font-weight:600; font-size:.85rem;
               padding:4px 9px; border-radius:6px; }}
  nav.toc a:hover {{ background:#e8efff; }}
  .mode-header {{ margin:26px 0 10px; display:flex; align-items:center; gap:12px; }}
  .mode-header h2 {{ margin:0; font-size:1.45rem; }}
  .mode-header p {{ margin:0; color:var(--muted); font-size:.9rem; }}
  .duration-block {{ background:var(--card); border:1px solid var(--line); border-radius:14px;
                     padding:22px; margin:18px 0 26px; box-shadow:0 2px 10px rgba(15,23,42,.04); }}
  .duration-head {{ display:flex; justify-content:space-between; align-items:flex-end;
                    gap:16px; border-bottom:2px solid #eef2f7; padding-bottom:12px; }}
  .duration-head h2 {{ margin:2px 0 0; font-size:1.45rem; }}
  .mode-pill {{ display:inline-block; font-size:.72rem; font-weight:800; letter-spacing:.06em;
                text-transform:uppercase; padding:3px 9px; border-radius:999px; color:#fff; }}
  .mode-pill.reconstruction {{ background:var(--violet); }}
  .mode-pill.cross {{ background:var(--blue); }}
  .dur-meta {{ color:var(--muted); font-size:.85rem; text-align:right; }}
  .mode-note {{ color:var(--muted); font-size:.86rem; margin:10px 0 16px; }}
  .input-row {{ display:grid; grid-template-columns:minmax(0,2fr) minmax(240px,1fr);
                gap:16px; margin-bottom:18px; }}
  .input-card,.audio-card {{ border:1px solid var(--line); border-radius:10px; padding:14px;
                             background:#fbfdff; }}
  .input-title {{ font-weight:700; font-size:.9rem; margin-bottom:9px; }}
  .input-title span {{ color:var(--muted); font-weight:500; }}
  .input-card video {{ width:100%; max-height:320px; border-radius:8px; background:#000; }}
  .video-meta {{ color:var(--muted); font-size:.77rem; margin-top:6px; }}
  .audio-name {{ font-family:ui-monospace,Consolas,monospace; font-size:.85rem; word-break:break-all; }}
  .audio-dur {{ font-size:1.8rem; font-weight:800; color:var(--blue); margin:8px 0; }}
  .audio-card p {{ color:var(--muted); font-size:.82rem; margin:0; }}
  .cards {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; }}
  @media (max-width:1250px) {{ .cards {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}
  @media (max-width:720px) {{ .cards {{ grid-template-columns:1fr; }}
                              .input-row {{ grid-template-columns:1fr; }} }}
  .model-card {{ border:1px solid var(--line); border-radius:10px; padding:12px;
                 display:flex; flex-direction:column; gap:8px; background:#fff; }}
  .model-head {{ display:flex; align-items:center; gap:7px; flex-wrap:wrap; }}
  .badge {{ display:inline-block; color:#fff; font-size:.7rem; font-weight:800;
            padding:3px 8px; border-radius:999px; letter-spacing:.03em; }}
  .badge-wav2lip {{ background:#15803d; }}
  .badge-heygem {{ background:#b91c1c; }}
  .badge-musetalk {{ background:#b45309; }}
  .badge-latentsync {{ background:#7c3aed; }}
  .tag {{ color:var(--muted); font-size:.72rem; font-family:ui-monospace,Consolas,monospace; }}
  .status {{ margin-left:auto; font-size:.7rem; font-weight:700; }}
  .status.completed {{ color:var(--green); }}
  .status.failed {{ color:var(--red); }}
  .model-card video {{ width:100%; aspect-ratio:16/9; border-radius:8px; background:#000; }}
  table.stats {{ width:100%; border-collapse:separate; border-spacing:0; font-size:.8rem;
                 border:1px solid #e6edf5; border-radius:8px; overflow:hidden; }}
  table.stats tr.group-row th {{ background:#f1f5f9; color:#334155; font-size:.66rem;
                 font-weight:800; letter-spacing:.09em; text-transform:uppercase;
                 padding:6px 8px; border-bottom:1px solid #e2e8f0; }}
  table.stats th {{ text-align:left; color:var(--muted); font-weight:600; padding:5px 8px;
                    border-bottom:1px solid #f1f5f9; white-space:nowrap; }}
  table.stats td {{ text-align:right; padding:5px 8px; border-bottom:1px solid #f1f5f9;
                    font-variant-numeric:tabular-nums; }}
  table.stats tbody tr:nth-child(even) td {{ background:#fbfdff; }}
  table.stats tbody tr:hover td {{ background:#eef6ff; }}
  table.stats tr:last-child th, table.stats tr:last-child td {{ border-bottom:none; }}
  table.stats td.best {{ color:#166534; font-weight:800; background:#ecfdf3;
                    box-shadow:inset 0 0 0 1px #bbf7d0; border-radius:6px; }}
  table.stats td.na {{ color:#94a3b8; font-style:italic; }}
  .err {{ color:var(--amber); font-size:.7rem; }}
  .manual {{ color:var(--muted); font-style:italic; }}
  .warn {{ color:var(--amber); font-size:.7rem; }}
  h3 {{ margin:20px 0 8px; font-size:1.05rem; }}
  .table-wrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:12px;
                 box-shadow:0 2px 10px rgba(15,23,42,.04); }}
  table.compare {{ width:100%; border-collapse:separate; border-spacing:0; font-size:.8rem;
                   min-width:1180px; }}
  table.compare thead th {{ background:#111827; color:#e5e7eb; padding:8px 9px; text-align:right;
                   white-space:nowrap; font-size:.7rem; text-transform:uppercase;
                   letter-spacing:.05em; border-bottom:1px solid #334155; }}
  table.compare thead tr.group-row th {{ background:#1e293b; color:#cbd5e1; font-size:.65rem;
                   letter-spacing:.08em; padding:6px 9px; border-bottom:1px solid #334155; }}
  table.compare thead th.model-head {{ text-align:left; vertical-align:bottom;
                   background:#0f172a; color:#fff; }}
  table.compare tbody td {{ padding:8px 9px; border-bottom:1px solid #eef2f7; text-align:right;
                   font-variant-numeric:tabular-nums; }}
  table.compare tbody td.model-cell {{ text-align:left; white-space:nowrap; font-weight:700; }}
  table.compare tbody tr:nth-child(even) td {{ background:#fbfdff; }}
  table.compare tbody tr:hover td {{ background:#eef6ff; }}
  table.compare tbody tr:last-child td {{ border-bottom:none; }}
  table.compare td.best {{ background:#ecfdf3; color:#166534; font-weight:800;
                   box-shadow:inset 0 0 0 1px #bbf7d0; }}
  table.compare td.na {{ color:#94a3b8; font-style:italic; }}
  table.compare th.model-head, table.compare td.model-cell {{ position:sticky; left:0; z-index:2; }}
  table.compare td.model-cell {{ background:#fff; }}
  table.compare tbody tr:nth-child(even) td.model-cell {{ background:#fbfdff; }}
  table.compare tbody tr:hover td.model-cell {{ background:#eef6ff; }}
  table.compare thead th.model-head {{ z-index:3; }}
  table.compare thead tr.group-row th.group-lse {{ box-shadow:inset 0 3px 0 #60a5fa; }}
  table.compare thead tr.group-row th.group-id {{ box-shadow:inset 0 3px 0 #34d399; }}
  table.compare thead tr.group-row th.group-gen {{ box-shadow:inset 0 3px 0 #fbbf24; }}
  table.compare thead tr.group-row th.group-rec {{ box-shadow:inset 0 3px 0 #a78bfa; }}
  table.compare thead tr.group-row th.group-nr {{ box-shadow:inset 0 3px 0 #f472b6; }}
  table.compare thead tr.group-row th.group-sys {{ box-shadow:inset 0 3px 0 #94a3b8; }}
  .table-legend {{ color:var(--muted); font-size:.75rem; margin:8px 2px 0; }}
  .note-box {{ background:#eef5ff; border-left:4px solid var(--blue); border-radius:0 10px 10px 0;
               padding:14px 16px; margin:18px 0; font-size:.86rem; color:#1e3a8a; }}
  .note-box b {{ color:#0f2f6b; }}
  .legend {{ display:flex; flex-wrap:wrap; gap:14px; font-size:.78rem; color:var(--muted); margin-top:8px; }}
  footer {{ text-align:center; color:var(--muted); font-size:.8rem; padding:22px 0 8px; }}
</style>
</head>
<body>
<div class="wrap">
  <header class="hero">
    <h1>🎯 Lip-Sync Quality Metrics — Videos + Numbers</h1>
    <p>Reconstruction and cross-audio (dubbing) tests · same input video, only the audio length changes</p>
    <div class="hero-links">
      <a href="benchmark_report.html">← System benchmark report</a>
      <a href="#recon-section">Reconstruction tests</a>
      <a href="#cross-section">Cross-audio / dubbing</a>
      <a href="#metric-guide">Metric guide</a>
    </div>
  </header>

  <div class="meta-strip">
    <span><b>Runs:</b> {total}</span>
    <span><b>Models:</b> Wav2Lip · HeyGem · MuseTalk · LatentSync</span>
    <span><b>GPU:</b> RTX PRO 6000 Blackwell MIG 1g.24gb</span>
    <span><b>Input:</b> marathi-demo-2.mp4 (15.02 s, 1280×720)</span>
    <span><b>Node:</b> aicoeserver05</span>
  </div>

  <nav class="toc">{''.join(nav)}<a href="#metric-guide">Metric guide</a></nav>

  <div id="metric-guide" class="note-box">
    <b>How to read this:</b> <b>LSE-D</b> lower is better (lip/audio distance), <b>LSE-C</b> higher is better
    (sync confidence), <b>CSIM</b> higher is better (ArcFace identity), <b>FID/FVD</b> lower is better,
    <b>SSIM/PSNR</b> higher and <b>LPIPS/LMD</b> lower are better. <b>NIQE</b> lower is better
    (median over 3 fps face-centred crops).
    <b>MOS</b> is a human 1–5 score and is left manual. SSIM / PSNR / LPIPS / LMD only make sense in
    reconstruction (the output is compared with the original frames). Cross-audio runs are scored with
    no-reference metrics + identity + lip-sync.
    <div class="legend"><span>Green = best value for that duration.</span>
      <span>— = not applicable / not computed.</span>
      <span>RTF = wall seconds ÷ audio seconds (lower is faster).</span></div>
  </div>

  <div id="recon-section" class="mode-header">
    <span class="mode-pill reconstruction">Reconstruction</span>
    <div><h2>Real video + its own audio</h2>
      <p>Frames can be compared with ground truth, so SSIM / PSNR / LPIPS / LMD are valid.</p></div>
  </div>
  {recon_sections or '<p>No reconstruction runs yet.</p>'}

  <div id="cross-section" class="mode-header">
    <span class="mode-pill cross">Cross-audio / dubbing</span>
    <div><h2>Same video + a new voice track</h2>
      <p>TalkTwin's production case. No ground-truth frames, so reconstruction metrics are n/a.</p></div>
  </div>
  {cross_sections or '<p>No cross-audio runs yet.</p>'}

  <footer>
    Generated from benchmark report data · videos are linked relatively next to this HTML file ·
    MOS remains manual by design.
  </footer>
</div>
</body>
</html>
"""
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} ({len(page)} chars, {total} runs)")


if __name__ == "__main__":
    main()
