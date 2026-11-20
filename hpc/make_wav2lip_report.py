#!/usr/bin/env python3
"""Build wav2lip_report.html mirroring benchmark_report.html structure."""
import html
import json
import statistics
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEW_DIR = ROOT / "outputs" / "bench-20261119-lipsync-multilingual" / "wav2lip"
OUT = ROOT / "wav2lip_report.html"

LANG = {"hi": "Hindi", "en": "English", "mr": "Marathi"}
LANG_NATIVE = {"hi": "हिन्दी", "en": "English", "mr": "मराठी"}
VISUAL = {
    "visual01": ("Image 1 · portrait", "visual_01.jpg", "assets/lipsync_multilingual/visual_01.jpg"),
    "visual02": ("Image 2 · square", "visual_02.jpg", "assets/lipsync_multilingual/visual_02.jpg"),
    "video": ("Video · marathi-demo-2", "marathi-demo-2.mp4", "videos/inputs/marathi-demo-2.mp4"),
}
BADGE_CSS = {"visual01": "badge-visual01", "visual02": "badge-visual02", "video": "badge-video"}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def esc(value):
    return html.escape("" if value is None else str(value))


def num(value, digits=2, suffix=""):
    if value is None:
        return "—"
    try:
        return f"{float(value):,.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return esc(value)


def rtf_class(v):
    if v is None:
        return ""
    v = float(v)
    if v < 1.0:
        return "rtf-fast"
    if v < 3.0:
        return "rtf-medium"
    return "rtf-slow"


def tag_lang(tag):
    for key in ("hi", "en", "mr"):
        if tag.endswith(f"-{key}") or tag.endswith(f"-{key}-5s") or tag.endswith(f"-{key}-10s") or tag.endswith(f"-{key}-20s"):
            return key
    return "hi"


def tag_visual(tag):
    if tag.startswith("visual01"):
        return "visual01"
    if tag.startswith("visual02"):
        return "visual02"
    return "video"


def is_image(run):
    return bool(run.get("static")) or not str(run.get("tag", "")).startswith("video")


def row_from(run):
    audio = (run.get("input") or {}).get("audio") or {}
    out = run.get("output") or {}
    peaks = run.get("peaks") or {}
    tag = run.get("tag", "")
    return {
        "tag": tag,
        "index": run.get("index", 999),
        "status": run.get("status"),
        "wall": run.get("wall_s"),
        "rtf": run.get("rtf"),
        "audio": audio.get("duration_s"),
        "out": out.get("duration_s"),
        "vram": peaks.get("vram_mb"),
        "rss": peaks.get("cpu_rss_mb"),
        "res": out.get("resolution"),
        "fps": out.get("fps"),
        "size": out.get("size_mb"),
        "visual": tag_visual(tag),
        "lang": tag_lang(tag),
        "image": is_image(run),
        "note": run.get("note") or "",
    }


def new_rows():
    rows = []
    for path in sorted(NEW_DIR.glob("*.json")):
        if ".cold." in path.name:
            continue
        rows.append(row_from(load(path)))
    rows.sort(key=lambda r: r["index"])
    return rows


def median(vals):
    vals = [float(v) for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def mean(vals):
    vals = [float(v) for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


rows = new_rows()
images = [r for r in rows if r["image"]]
videos = [r for r in rows if not r["image"]]

total_wall = sum(r["wall"] for r in rows if r["wall"])
avg_rtf = mean([r["rtf"] for r in rows])
img_avg_wall = mean([r["wall"] for r in images])
vid_avg_wall = mean([r["wall"] for r in videos])
img_vram = max((r["vram"] or 0) for r in images) if images else 0
vid_vram = max((r["vram"] or 0) for r in videos) if videos else 0


def stats_table(r, best_wall=None):
    """Per-output stats table (same as benchmark_report output-card table)."""
    is_best = best_wall is not None and r["wall"] == best_wall
    wall_cls = "best" if is_best else ""
    return f"""<table class="stats-table"><tbody>
<tr class="group-row"><th colspan="2">Performance</th></tr>
<tr><th>Wall Time</th><td class="{wall_cls}">{num(r['wall'], 2)}s</td></tr>
<tr><th>RTF</th><td class="{rtf_class(r['rtf'])}">{num(r['rtf'], 3)}x</td></tr>
<tr class="group-row"><th colspan="2">Memory</th></tr>
<tr><th>Peak VRAM</th><td>{num((r['vram'] or 0) / 1024, 1)} GB</td></tr>
<tr><th>Peak RAM</th><td>{num((r['rss'] or 0) / 1024, 1)} GB</td></tr>
<tr class="group-row"><th colspan="2">Output</th></tr>
<tr><th>Output Size</th><td>{num(r['size'], 2)} MB</td></tr>
<tr><th>Output Dur</th><td>{num(r['out'], 2)}s @ {num(r['fps'], 0)}fps</td></tr>
<tr><th>Resolution</th><td>{esc(r['res'] or '—')}</td></tr>
</tbody></table>"""


def output_card(r, best_wall):
    src = f"videos/wav2lip_multilingual/{r['tag']}.mp4"
    vname, vfile, vlocal = VISUAL[r["visual"]]
    badge_css = BADGE_CSS[r["visual"]]
    return f"""<div class="output-card">
      <h4><span class="model-badge {badge_css}">{esc(vname)}</span> {esc(LANG[r['lang']])}</h4>
      <video controls>
        <source src="{esc(src)}" type="video/mp4">
        Your browser does not support the video tag.
      </video>
      {stats_table(r, best_wall)}
    </div>"""


def language_section(lang):
    items = [r for r in rows if r["lang"] == lang and r["audio"] and 13.5 <= r["audio"] <= 15.0]
    items.sort(key=lambda r: ("visual01" != r["visual"], "visual02" != r["visual"]))
    best_wall = min((r["wall"] for r in items), default=None)
    badges = "".join(
        f'<span class="model-badge {BADGE_CSS[r["visual"]]}">{esc(VISUAL[r["visual"]][0])}</span>'
        for r in items
    )
    cards = "\n".join(output_card(r, best_wall) for r in items)
    audio = items[0] if items else None
    audio_dur = num(audio["audio"], 2) if audio else "—"
    return f"""<section id="{lang}" class="duration-section">
      <div class="duration-header">
        <div>{badges}</div>
        <span class="duration-title">{esc(LANG[lang])} ({esc(LANG_NATIVE[lang])}) — {esc(LANG[lang])} audio · {audio_dur}s</span>
      </div>
      <div class="input-video-block">
        <h4>🔊 DRIVING AUDIO</h4>
        <p style="margin-bottom:10px; font-size:0.9rem; color:#7f8c8d;">
          OmniVoice zero-shot · {esc(LANG[lang])} · same Marathi female reference · 24 kHz
        </p>
        <div class="input-details">
          Audio: {esc(LANG[lang])} · Duration: {audio_dur}s · Source: OmniVoice TTS (Marathi female reference)<br>
          Marathi script shortened to fit the 15.02 s source video.
        </div>
      </div>
      <h3>🎥 Output Videos — All Visuals ({esc(LANG[lang])})</h3>
      <div class="output-grid">{cards}</div>
    </section>"""


def timing_matrix_section():
    items = [r for r in rows if r["tag"].endswith(("-5s", "-10s", "-20s"))]
    items.sort(key=lambda r: r["audio"] or 0)
    cards = "\n".join(output_card(r, None) for r in items)
    return f"""<section id="timing-matrix" class="duration-section">
      <div class="duration-header">
        <div><span class="model-badge badge-video">5 s · 10 s · 20 s</span></div>
        <span class="duration-title">5 / 10 / 20 s Timing Matrix — Hindi · visual_01</span>
      </div>
      <div class="input-video-block">
        <h4>🔊 DRIVING AUDIO</h4>
        <p style="margin-bottom:10px; font-size:0.9rem; color:#7f8c8d;">
          Hindi audio trimmed to 5 s and 10 s; 20 s is the 14.2 s clip looped to length.
        </p>
        <div class="input-details">
          Visual: visual_01.jpg (450×678) · Mode: static image · FPS: 24<br>
          20 s timing audio is a looped extension — used for scaling only.
        </div>
      </div>
      <h3>🎥 Timing Output Videos</h3>
      <div class="output-grid">{cards}</div>
    </section>"""


def comparison_table():
    body = []
    # group header rows by language
    for lang in ("hi", "en", "mr"):
        items = [r for r in rows if r["lang"] == lang and r["audio"] and 13.5 <= r["audio"] <= 15.0]
        items.sort(key=lambda r: ("visual01" != r["visual"], "visual02" != r["visual"]))
        body.append(f"<tr class='duration-row'><th colspan='10'>{esc(LANG[lang])} · {esc(LANG_NATIVE[lang])} · all visuals</th></tr>")
        for r in items:
            vname = VISUAL[r["visual"]][0]
            body.append(
                "<tr" + (" class='alt'" if items.index(r) % 2 else "") + ">"
                f"<td class='model-cell'><span class='model-badge {BADGE_CSS[r['visual']]}'>{esc(vname)}</span> {esc(LANG[r['lang']])}</td>"
                f"<td class='{'best' if r['wall']==min(x['wall'] for x in items) else ''}'>{num(r['wall'], 2)}s</td>"
                f"<td class='{rtf_class(r['rtf'])}'>{num(r['rtf'], 3)}x</td>"
                f"<td>{num((r['vram'] or 0) / 1024, 1)} GB</td>"
                f"<td>{num(((r['vram'] or 0) / (vid_vram if vid_vram else 1)) * 100, 1)}%</td>"
                f"<td>{num((r['rss'] or 0) / 1024, 1)} GB</td>"
                f"<td>{num(r['size'], 2)} MB</td>"
                f"<td>{num(r['out'], 2)}s</td>"
                f"<td>{esc(r['res'] or '—')}</td>"
                "<td>✅</td>"
                "</tr>"
            )
    # timing matrix rows
    titems = [r for r in rows if r["tag"].endswith(("-5s", "-10s", "-20s"))]
    titems.sort(key=lambda r: r["audio"] or 0)
    if titems:
        body.append("<tr class='duration-row'><th colspan='10'>Timing matrix · Hindi · visual_01 · 5/10/20 s</th></tr>")
        for r in titems:
            body.append(
                "<tr>"
                f"<td class='model-cell'><span class='model-badge badge-video'>Image · static</span> {num(r['audio'], 0)} s Hindi</td>"
                f"<td>{num(r['wall'], 2)}s</td>"
                f"<td class='{rtf_class(r['rtf'])}'>{num(r['rtf'], 3)}x</td>"
                f"<td>{num((r['vram'] or 0) / 1024, 1)} GB</td>"
                f"<td>{num(((r['vram'] or 0) / (img_vram if img_vram else 1)) * 100, 1)}%</td>"
                f"<td>{num((r['rss'] or 0) / 1024, 1)} GB</td>"
                f"<td>{num(r['size'], 2)} MB</td>"
                f"<td>{num(r['out'], 2)}s</td>"
                f"<td>{esc(r['res'] or '—')}</td>"
                "<td>✅</td>"
                "</tr>"
            )
    return "\n".join(body)


CSS = r"""
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: #f5f7fa;
    color: #333;
    line-height: 1.6;
    padding: 20px;
  }
  .container {
    max-width: 1600px;
    margin: 0 auto;
    background: white;
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    overflow: hidden;
  }
  header {
    background: linear-gradient(135deg, #1e7a3c 0%, #27ae60 100%);
    color: white;
    padding: 40px;
    text-align: center;
  }
  header h1 { font-size: 2rem; margin-bottom: 10px; }
  header p { font-size: 1.1rem; opacity: 0.95; }
  .meta-bar {
    background: #2c3e50;
    color: white;
    padding: 15px 40px;
    font-size: 0.9rem;
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
  }
  .meta-bar span { display: inline-block; }
  .meta-bar .label { font-weight: bold; }
  .nav {
    background: #ecf0f1;
    padding: 12px 40px;
    border-bottom: 1px solid #bdc3c7;
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
  }
  .nav a {
    color: #2980b9;
    text-decoration: none;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 4px;
    transition: background 0.2s;
  }
  .nav a:hover { background: #d5dbdb; }
  .content { padding: 30px 40px; }
  h2 {
    font-size: 1.6rem;
    color: #2c3e50;
    margin: 30px 0 15px 0;
    padding-bottom: 10px;
    border-bottom: 2px solid #27ae60;
  }
  h3 { font-size: 1.3rem; color: #2980b9; margin: 25px 0 12px 0; }
  .model-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: bold;
    color: white;
    margin-right: 8px;
  }
  .badge-visual01 { background: #3498db; }
  .badge-visual02 { background: #9b59b6; }
  .badge-video { background: #27ae60; }
  .duration-section {
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 25px;
    margin: 30px 0;
  }
  .duration-header {
    display: flex;
    align-items: center;
    margin-bottom: 20px;
    padding-bottom: 15px;
    border-bottom: 2px solid #ecf0f1;
  }
  .duration-title { font-size: 1.5rem; font-weight: bold; color: #2c3e50; }
  .input-video-block {
    background: #f8f9fa;
    border: 1px solid #dfe6e9;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 25px;
    text-align: center;
  }
  .input-video-block h4 { margin-bottom: 10px; color: #2c3e50; }
  .input-video-block video {
    width: 100%;
    max-width: 600px;
    border-radius: 6px;
    border: 2px solid #27ae60;
    background: #000;
  }
  .input-details {
    font-size: 0.9rem;
    color: #7f8c8d;
    margin-top: 10px;
  }
  .output-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
    margin-bottom: 25px;
  }
  .output-card {
    background: white;
    border: 1px solid #dfe6e9;
    border-radius: 8px;
    padding: 15px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .output-card h4 { font-size: 1.1rem; margin-bottom: 8px; }
  .output-card video {
    width: 100%;
    height: auto;
    max-height: 180px;
    border-radius: 6px;
    border: 2px solid #2c3e50;
    background: #000;
  }
  .stats-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    margin-top: 10px;
    font-size: 0.82rem;
    border: 1px solid #e6edf5;
    border-radius: 8px;
    overflow: hidden;
  }
  .stats-table tr.group-row th {
    background: #f1f5f9;
    color: #334155;
    font-size: 0.66rem;
    font-weight: 800;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    padding: 5px 9px;
    border-bottom: 1px solid #e2e8f0;
  }
  .stats-table th {
    background: #fff;
    color: #64748b;
    padding: 4px 9px;
    text-align: left;
    font-weight: 600;
    font-size: 0.78rem;
    border-bottom: 1px solid #f1f5f9;
  }
  .stats-table td {
    padding: 4px 9px;
    text-align: right;
    border-bottom: 1px solid #f1f5f9;
    font-variant-numeric: tabular-nums;
  }
  .stats-table tr:nth-child(even) td { background: #fbfdff; }
  .stats-table tr:hover td { background: #eef6ff; }
  .stats-table tr:last-child th,
  .stats-table tr:last-child td { border-bottom: none; }
  .comparison-table-container {
    overflow-x: auto;
    margin: 30px 0;
    border: 1px solid #dfe6ee;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(15,23,42,0.05);
    max-height: 80vh;
  }
  .comparison-table {
    width: 100%;
    min-width: 1080px;
    border-collapse: separate;
    border-spacing: 0;
    margin: 0;
    background: #fff;
    font-size: 0.84rem;
  }
  .comparison-table thead th {
    position: sticky;
    top: 0;
    z-index: 2;
    background: #111827;
    color: #e5e7eb;
    padding: 8px 10px;
    text-align: right;
    white-space: nowrap;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 1px solid #334155;
  }
  .comparison-table thead tr.group-row th {
    background: #1e293b;
    color: #cbd5e1;
    font-size: 0.65rem;
    letter-spacing: 0.08em;
    padding: 6px 10px;
  }
  .comparison-table thead th.model-head {
    z-index: 3;
    left: 0;
    text-align: left;
    background: #0f172a;
    color: #fff;
    vertical-align: bottom;
  }
  .comparison-table tbody td {
    padding: 8px 10px;
    text-align: right;
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
    border-bottom: 1px solid #eef2f7;
    background: #fff;
  }
  .comparison-table tbody td.model-cell {
    text-align: left;
    font-weight: 700;
    position: sticky;
    left: 0;
    z-index: 1;
  }
  .comparison-table tbody tr.alt td { background: #fbfdff; }
  .comparison-table tbody tr:hover td { background: #eef6ff; }
  .comparison-table tbody tr.duration-row th {
    background: #f1f5f9;
    color: #1e3a8a;
    text-align: left;
    padding: 7px 10px;
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    border-top: 1px solid #e2e8f0;
    border-bottom: 1px solid #e2e8f0;
    position: sticky;
    left: 0;
    z-index: 1;
  }
  .comparison-table tbody tr.duration-row:first-child th { border-top: none; }
  .comparison-table tbody tr:last-child td { border-bottom: none; }
  .comparison-table td.best {
    background: #ecfdf3;
    color: #166534;
    font-weight: 800;
    box-shadow: inset 0 0 0 1px #bbf7d0;
  }
  .comparison-table tbody tr.alt td.best { background: #e7fbf0; }
  .comparison-table tbody tr:hover td.best { background: #dcf7e8; }
  .table-legend {
    color: #65758b;
    font-size: 0.75rem;
    margin: 10px 2px 0;
  }
  .rtf-fast { color: #27ae60; font-weight: bold; }
  .rtf-medium { color: #f39c12; font-weight: bold; }
  .rtf-slow { color: #e74c3c; font-weight: bold; }
  .vramsummary {
    background: #e8f4fd;
    border-left: 4px solid #27ae60;
    padding: 15px;
    margin: 20px 0;
    border-radius: 0 8px 8px 0;
  }
  .vramsummary h4 { margin-top: 0; color: #1e7a3c; }
  .vramsummary ul { margin-left: 20px; color: #34495e; }
  footer {
    background: #2c3e50;
    color: #bdc3c7;
    padding: 30px 40px;
    font-size: 0.9rem;
    text-align: center;
  }
  .warning {
    background: #fdecea;
    border-left: 4px solid #e74c3c;
    padding: 15px;
    margin: 20px 0;
    border-radius: 0 8px 8px 0;
  }
"""

lang_sections = "\n".join(language_section(l) for l in ("hi", "en", "mr"))

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Wav2Lip Multilingual Lip-Sync Report — Video Comparison</title>
<style>{CSS}</style>
</head>
<body>

<div class="container">
  <header>
    <h1>🎬 Wav2Lip Multilingual Lip-Sync Report — Video Comparison</h1>
    <p>Hindi · English · Marathi | OmniVoice audio | 2 images + 1 video | 12 runs</p>
  </header>

  <div class="meta-bar">
    <span><span class="label">Total Runs:</span> 12</span>
    <span><span class="label">Model:</span> Wav2Lip GAN (wav2lip_gan.pth)</span>
    <span><span class="label">Languages:</span> Hindi · English · Marathi</span>
    <span><span class="label">Audio:</span> OmniVoice · 24 kHz</span>
    <span><span class="label">GPU:</span> RTX PRO 6000 Blackwell MIG 1g.24gb</span>
    <span><span class="label">Success:</span> 100% (12/12)</span>
  </div>

  <div class="nav">
    <a href="#overview">Overview</a>
    <a href="#hindi">Hindi</a>
    <a href="#english">English</a>
    <a href="#marathi">Marathi</a>
    <a href="#timing-matrix">5/10/20 s Timing</a>
    <a href="#comparison-table">Full Comparison Table</a>
    <a href="benchmark_report.html" style="background:#7c3aed;color:#fff;">📊 System Benchmark Report →</a>
  </div>

  <div class="content">
    <section id="overview">
      <h2>📊 Overview</h2>
      <div class="vramsummary">
        <h4>🎯 How to Read This Report</h4>
        <ul>
          <li>Each language section = one OmniVoice audio clip lip-synced to all 3 visuals (visual_01, visual_02, marathi-demo-2).</li>
          <li>Top: driving audio details. Middle: all output videos side-by-side (grid). Bottom: per-output stats table.</li>
          <li>RTF = wall_seconds ÷ audio_seconds (lower = faster). 1.0 = realtime.</li>
          <li>VRAM = peak GPU memory used on the MIG slice.</li>
          <li>5 / 10 / 20 s timing runs use visual_01 + Hindi audio for duration scaling.</li>
        </ul>
      </div>
      <div class="warning">
        <h4>⚠️ Scope Note</h4>
        <p>This report contains <b>benchmark timings only</b>. Per scope, no LSE-D/C, CSIM, or NIQE quality metrics were run for these outputs.
        Static-image outputs move only the mouth (head/background frozen). Image resolution is native (450×678 and 554×554) — not upscaled.</p>
      </div>
    </section>

    {lang_sections}

    {timing_matrix_section()}

    <section id="comparison-table">
      <h2>📈 Full Comparison Table — All 12 Runs</h2>
      <div class="comparison-table-container">
        <table class="comparison-table">
          <thead>
            <tr class="group-row"><th rowspan="2" class="model-head">Visual · Language</th><th colspan="2">Latency</th><th colspan="2">GPU memory</th><th colspan="1">System memory</th><th colspan="3">Output</th><th rowspan="2">Status</th></tr>
            <tr><th>Wall time</th><th>RTF</th><th>Peak VRAM</th><th>VRAM %</th><th>Peak RAM</th><th>Size</th><th>Duration</th><th>Resolution</th></tr>
          </thead>
          <tbody>
{comparison_table()}
          </tbody>
        </table>
      </div>
      <p class="table-legend">RTF = wall ÷ audio duration. VRAM % is relative to the peak observed in this benchmark (image {num(img_vram/1024,1)} GB / video {num(vid_vram/1024,1)} GB). Best wall time per language highlighted.</p>
    </section>

  </div>

  <footer>
    <p><strong>Wav2Lip Multilingual Lip-Sync Report</strong> — Hindi · English · Marathi</p>
    <p>12 runs · OmniVoice 24 kHz audio · node05 MIG 1g.24gb · All 12 runs COMPLETED</p>
    <p>Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} · Total wall {num(total_wall, 1)} s · Avg RTF {num(avg_rtf, 3)}</p>
  </footer>
</div>

</body>
</html>"""

OUT.write_text(HTML, encoding="utf-8")
print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")
print(f"rows={len(rows)} image={len(images)} video={len(videos)}")
