"""Build a self-contained reports folder.

Creates:
    reports/benchmark_report.html   system metrics + videos (links rewritten)
    reports/quality_report.html     quality metrics + videos (links rewritten)
    reports/outputs_report.html     videos only: common input + 4 outputs per group
    reports/videos/                 inputs, cross, recon, long outputs
    reports/README.md

All source videos stay in outputs/.  Only copies are placed under reports/.
"""
import re
import shutil
from pathlib import Path

ROOT = Path(".")
OUT = ROOT / "outputs" / "bench-20261031-152743"
LONG = ROOT / "outputs" / "bench-20261031-143801"
DEST = ROOT / "reports"
VIDEOS = DEST / "videos"

MODELS = [
    ("wav2lip", "Wav2Lip", "badge-wav2lip"),
    ("heygem", "HeyGem", "badge-heygem"),
    ("musetalk", "MuseTalk", "badge-musetalk"),
    ("latentsync", "LatentSync", "badge-latentsync"),
]

GROUPS = [
    dict(mode="reconstruction", label="Reconstruction 5 s", anchor="recon-5s",
         tag="recon-5s", audio="own audio · 5 s",
         note="The video's own 5 s audio was used. Only outputs are shown here."),
    dict(mode="reconstruction", label="Reconstruction 10 s", anchor="recon-10s",
         tag="recon-10s", audio="own audio · 10 s",
         note="The video's own 10 s audio was used. Only outputs are shown here."),
    dict(mode="reconstruction", label="Reconstruction 15 s", anchor="recon-15s",
         tag="recon-15s", audio="own audio · 15 s",
         note="The video's own 15 s audio was used. Only outputs are shown here."),
    dict(mode="cross", label="Dubbing 5 s", anchor="dub-5s",
         tag="marathi-demo-2-5s", audio="marathi-demo-voice-2-5s.mp3 · 5 s",
         note="A separate 5 s voice track was dubbed onto the common input video."),
    dict(mode="cross", label="Dubbing 10 s", anchor="dub-10s",
         tag="marathi-demo-2-10s", audio="marathi-demo-voice-2-10s.mp3 · 10 s",
         note="A separate 10 s voice track was dubbed onto the common input video."),
    dict(mode="cross", label="Dubbing 20 s", anchor="dub-20s",
         tag="marathi-demo-2-20s", audio="marathi-demo-voice-2-20s.mp3 · 20 s",
         note="A separate 20 s voice track was dubbed onto the common input video."),
    dict(mode="cross", label="Dubbing 64.5 s (1 min)", anchor="dub-64s",
         tag="marathi-demo-2", audio="marathi-demo-voice-2.mp3 · 64.5 s",
         note="The full 64.5 s voice track was used; outputs are extended to the audio length.",
         long=True),
]


def copy(src, dst):
    if not src.exists():
        raise SystemExit(f"missing source: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def rewrite(src_html, dst_html):
    s = src_html.read_text(encoding="utf-8")
    s = re.sub(r'src="inputs/[^"]+"',
               'src="videos/inputs/marathi-demo-2.mp4"', s)
    s = re.sub(r'src="recon/(wav2lip|heygem|musetalk|latentsync)/([^"]+)"',
               r'src="videos/recon/\1/\2"', s)
    s = re.sub(r'src="\./(wav2lip|heygem|musetalk|latentsync)/([^"]+)"',
               r'src="videos/cross/\1/\2"', s)
    s = re.sub(r'src="(wav2lip|heygem|musetalk|latentsync)/([^"]+)"',
               r'src="videos/cross/\1/\2"', s)
    s = re.sub(r'src="\.\./bench-20261031-143801/(wav2lip|heygem|musetalk|latentsync)/([^"]+)"',
               r'src="videos/long/\1/\2"', s)
    dst_html.write_text(s, encoding="utf-8")
    return s


def output_src(group, model):
    if group["mode"] == "reconstruction":
        return f"videos/recon/{model}/{group['tag']}.mp4"
    if group.get("long"):
        return f"videos/long/{model}/marathi-demo-2.mp4"
    return f"videos/cross/{model}/{group['tag']}.mp4"


def model_card(group, model, label, badge):
    src = output_src(group, model)
    return f"""
      <div class="model-card">
        <div class="model-head">
          <span class="badge {badge}">{label}</span>
          <span class="tag">{group['tag']}</span>
        </div>
        <video controls preload="metadata" src="{src}"></video>
        <div class="video-meta">{label} · {group['audio'].split('·')[-1].strip()}</div>
      </div>"""


def build_outputs_html():
    nav = "".join(
        f'<a href="#{g["anchor"]}">{g["label"]}</a>' for g in GROUPS)

    sections = []
    for g in GROUPS:
        cards = "".join(model_card(g, m, label, badge) for m, label, badge in MODELS)
        mode_label = "Reconstruction" if g["mode"] == "reconstruction" else "Cross-audio"
        mode_class = "reconstruction" if g["mode"] == "reconstruction" else "cross"
        sections.append(f"""
  <section id="{g['anchor']}" class="duration-block">
    <div class="duration-head">
      <div>
        <span class="mode-pill {mode_class}">{mode_label}</span>
        <h2>{g['label']}</h2>
      </div>
      <div class="dur-meta">{g['audio']}</div>
    </div>
    <p class="mode-note">{g['note']}</p>

    <div class="input-card">
      <div class="input-title">📥 Input video <span>(common source for every model)</span></div>
      <video controls preload="metadata" src="videos/inputs/marathi-demo-2.mp4"></video>
      <div class="video-meta">marathi-demo-2.mp4 · 15.02 s · 1280×720 · 24 fps</div>
    </div>

    <div class="cards">{cards}
    </div>
  </section>""")

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Lip-Sync Outputs — Marathi Demo (videos only)</title>
<style>
  :root {{
    --bg:#f5f7fa; --card:#fff; --ink:#22303f; --muted:#65758b;
    --line:#dfe6ee; --blue:#2563eb; --green:#15803d; --amber:#b45309;
    --red:#b91c1c; --violet:#7c3aed;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
         font-family:"Segoe UI",system-ui,-apple-system,sans-serif; line-height:1.5; }}
  .wrap {{ max-width:1600px; margin:0 auto; padding:22px; }}
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
  .input-card {{ border:1px solid var(--line); border-radius:10px; padding:14px;
                 background:#fbfdff; margin-bottom:18px; }}
  .input-title {{ font-weight:700; font-size:.9rem; margin-bottom:9px; }}
  .input-title span {{ color:var(--muted); font-weight:500; }}
  .input-card video {{ width:100%; max-height:360px; border-radius:8px; background:#000; }}
  .video-meta {{ color:var(--muted); font-size:.77rem; margin-top:6px; }}
  .cards {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; }}
  @media (max-width:1250px) {{ .cards {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}
  @media (max-width:720px)  {{ .cards {{ grid-template-columns:1fr; }} }}
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
  .model-card video {{ width:100%; aspect-ratio:16/9; border-radius:8px; background:#000; }}
  footer {{ text-align:center; color:var(--muted); font-size:.8rem; padding:22px 0 8px; }}
</style>
</head>
<body>
<div class="wrap">
  <header class="hero">
    <h1>🎬 Lip-Sync Outputs — Marathi Demo</h1>
    <p>All inputs and all four model outputs in one place · videos only, no metrics</p>
    <div class="hero-links">
      <a href="benchmark_report.html">← System benchmark report</a>
      <a href="quality_report.html">Quality metrics report</a>
      <a href="#recon-5s">Reconstruction</a>
      <a href="#dub-5s">Dubbing</a>
    </div>
  </header>

  <div class="meta-strip">
    <span><b>Input:</b> marathi-demo-2.mp4 · 15.02 s · 1280×720 · 24 fps</span>
    <span><b>Models:</b> Wav2Lip · HeyGem · MuseTalk · LatentSync</span>
    <span><b>Outputs:</b> 28 videos (7 test groups × 4 models)</span>
    <span><b>GPU:</b> RTX PRO 6000 Blackwell MIG 1g.24gb</span>
  </div>

  <nav class="toc">{nav}</nav>

  <div class="note-box" style="background:#eef5ff;border-left:4px solid #2563eb;
       border-radius:0 10px 10px 0;padding:14px 16px;margin:18px 0;font-size:.86rem;color:#1e3a8a;">
    The same <b>marathi-demo-2.mp4</b> source video is used everywhere; only the driving
    audio length changes. Reconstruction uses the video's own audio, dubbing uses a separate
    voice track. Quality numbers live in <a href="quality_report.html"><b>quality_report.html</b></a>.
  </div>
{''.join(sections)}
</div>
<footer>Lip-Sync model benchmark · aicoeserver05 · RTX PRO 6000 Blackwell MIG 1g.24gb</footer>
</body>
</html>
"""
    (DEST / "outputs_report.html").write_text(page, encoding="utf-8")
    return page


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    VIDEOS.mkdir(parents=True, exist_ok=True)

    # one common input video (all duration folders hold the same source)
    copy(OUT / "inputs" / "marathi-demo-2" / "marathi-demo-2.mp4",
         VIDEOS / "inputs" / "marathi-demo-2.mp4")

    for model, _, _ in MODELS:
        for tag in ("marathi-demo-2-5s", "marathi-demo-2-10s", "marathi-demo-2-20s"):
            copy(OUT / model / f"{tag}.mp4", VIDEOS / "cross" / model / f"{tag}.mp4")
        for tag in ("recon-5s", "recon-10s", "recon-15s"):
            copy(OUT / "recon" / model / f"{tag}.mp4", VIDEOS / "recon" / model / f"{tag}.mp4")
        copy(LONG / model / "marathi-demo-2.mp4",
             VIDEOS / "long" / model / "marathi-demo-2.mp4")

    rewrite(OUT / "benchmark_report.html", DEST / "benchmark_report.html")
    rewrite(OUT / "quality_report.html", DEST / "quality_report.html")

    readme = """# Lip-Sync benchmark reports — Marathi demo

Open **benchmark_report.html** for system metrics (wall time, RTF, VRAM) and video comparison.
Open **quality_report.html** for quality metrics (LSE-D/C, FID, FVD, SSIM, PSNR, LPIPS, CSIM, LMD, NIQE).

The two reports link directly to each other — no separate landing page needed.

Videos are self-contained under `videos/`:

- `videos/inputs/marathi-demo-2.mp4` — common 15.02 s source video
- `videos/recon/<model>/recon-{5,10,15}s.mp4` — reconstruction (video's own audio)
- `videos/cross/<model>/marathi-demo-2-{5,10,20}s.mp4` — dubbing
- `videos/long/<model>/marathi-demo-2.mp4` — 64.5 s dubbing

Models: Wav2Lip, HeyGem, MuseTalk, LatentSync.
MOS is a manual 1–5 rating and is intentionally left blank.
"""
    (DEST / "README.md").write_text(readme, encoding="utf-8")

    print("built", DEST)
    htmls = list(DEST.glob("*.html"))
    print("html:", [p.name for p in htmls])
    total = sum(p.stat().st_size for p in VIDEOS.rglob("*.mp4"))
    print(f"videos: {len(list(VIDEOS.rglob('*.mp4')))} files, {total/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
