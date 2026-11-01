"""Restyle the static system benchmark report tables.

- scopes the old global table CSS to .comparison-table (it was leaking into
  the per-model .stats-table)
- upgrades .stats-table with group rows, zebra rows and hover
- rebuilds the full 16-run comparison table grouped by duration, with best
  values per duration highlighted

Run from the project root:  python hpc/restyle_benchmark_report.py
"""
import re
from pathlib import Path

PATH = Path("outputs/bench-20261031-152743/benchmark_report.html")

MODEL_ORDER = ["wav2lip", "heygem", "musetalk", "latentsync"]
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
DURATION_LABEL = {
    "5s": "5 s",
    "10s": "10 s",
    "20s": "20 s",
    "64.5s": "64.5 s (full track)",
}

OLD_STATS_CSS = """  .stats-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 10px;
    font-size: 0.85rem;
  }
  .stats-table th {
    background: #ecf0f1;
    color: #34495e;
    padding: 6px 8px;
    text-align: left;
    font-weight: 600;
    font-size: 0.8rem;
  }
  .stats-table td {
    padding: 6px 8px;
    border-bottom: 1px solid #f0f0f0;
  }
"""

NEW_STATS_CSS = """  .stats-table {
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
"""

OLD_GLOBAL_CSS = """  table {
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
    background: white;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    border-radius: 8px;
    overflow: hidden;
  }
  th {
    background: #34495e;
    color: white;
    padding: 14px;
    text-align: left;
    font-weight: 600;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  td {
    padding: 12px 14px;
    border-bottom: 1px solid #e0e0e0;
    font-size: 0.9rem;
  }
  tr:nth-child(even) { background: #f9fafb; }
  tr:hover { background: #e8f4fd; }
"""

NEW_COMPARE_CSS = """  .comparison-table-container {
    overflow: auto;
    margin: 26px 0;
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
  .comparison-table thead th.g-latency { box-shadow: inset 0 3px 0 #60a5fa; }
  .comparison-table thead th.g-gpu { box-shadow: inset 0 3px 0 #34d399; }
  .comparison-table thead th.g-ram { box-shadow: inset 0 3px 0 #fbbf24; }
  .comparison-table thead th.g-output { box-shadow: inset 0 3px 0 #a78bfa; }
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
  .comparison-table tbody tr.duration-row th .dur-label {
    position: sticky;
    left: 12px;
    display: inline-block;
  }
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
"""

ROW_RE = re.compile(
    r"<tr><td>([^<]+)</td><td><span class=\"model-badge ([^\"]+)\">([^<]+)</span></td>"
    r"(.*?)</tr>",
    re.S,
)
CELL_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.S)


def parse_rows(s):
    rows = []
    for m in ROW_RE.finditer(s):
        duration, badge, label, rest = m.groups()
        cells = CELL_RE.findall(rest)
        if len(cells) != 9:
            raise SystemExit(f"row {label} {duration}: expected 9 cells, got {len(cells)}")
        rows.append(dict(duration=duration, badge=badge, label=label,
                         wall=cells[0], rtf=cells[1], vram=cells[2],
                         vram_pct=cells[3], ram=cells[4], size=cells[5],
                         outdur=cells[6], resolution=cells[7], status=cells[8]))
    return rows


def number(text):
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(m.group(0)) if m else None


def fmt_value(value):
    return value


def best_index(rows, key):
    vals = [(number(r[key]), i) for i, r in enumerate(rows)]
    vals = [(v, i) for v, i in vals if v is not None]
    if not vals:
        return None
    return min(vals)[1]


def build_full_table(rows):
    by_dur = {}
    for r in rows:
        by_dur.setdefault(r["duration"], {})[r["label"]] = r

    head = (
        "<thead><tr class=\"group-row\">"
        "<th rowspan=\"2\" class=\"model-head\">Model</th>"
        "<th colspan=\"2\" class=\"g-latency\">Latency</th>"
        "<th colspan=\"2\" class=\"g-gpu\">GPU memory</th>"
        "<th colspan=\"1\" class=\"g-ram\">System memory</th>"
        "<th colspan=\"3\" class=\"g-output\">Output</th>"
        "<th rowspan=\"2\">Status</th>"
        "</tr><tr>"
        "<th>Wall time</th><th>RTF</th><th>Peak VRAM</th><th>VRAM %</th>"
        "<th>Peak RAM</th><th>Size</th><th>Duration</th><th>Resolution</th>"
        "</tr></thead>"
    )

    body = []
    for duration in ("5s", "10s", "20s", "64.5s"):
        group = by_dur[duration]
        ordered = [group[MODEL_LABEL[m]] for m in MODEL_ORDER if MODEL_LABEL[m] in group]
        # only models that are in this group, in canonical order
        best = {
            key: best_index(ordered, key)
            for key in ("wall", "rtf", "vram", "ram")
        }
        body.append(
            f"<tr class=\"duration-row\"><th colspan=\"10\">"
            f"<span class=\"dur-label\">{DURATION_LABEL[duration]} · cross-audio</span>"
            f"</th></tr>"
        )
        for i, r in enumerate(ordered):
            model_key = next(k for k, v in MODEL_LABEL.items() if v == r["label"])

            def cell(key, cls=""):
                classes = [c for c in (cls, "best" if best.get(key) == i else "") if c]
                attr = f" class=\"{' '.join(classes)}\"" if classes else ""
                return f"<td{attr}>{r[key]}</td>"

            alt = " class=\"alt\"" if i % 2 else ""
            body.append(
                f"<tr{alt}>"
                f"<td class=\"model-cell\"><span class=\"model-badge {BADGE[model_key]}\">"
                f"{r['label']}</span></td>"
                + cell("wall")
                + cell("rtf", "rtf-" + ("fast" if number(r["rtf"]) < 4 else
                                        "medium" if number(r["rtf"]) < 12 else "slow"))
                + cell("vram")
                + f"<td>{r['vram_pct']}</td>"
                + cell("ram")
                + f"<td>{r['size']}</td><td>{r['outdur']}</td>"
                + f"<td>{r['resolution']}</td><td>{r['status']}</td>"
                "</tr>"
            )

    return (
        "<table class=\"comparison-table\">"
        + head
        + "<tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def main():
    s = PATH.read_text(encoding="utf-8")

    if OLD_STATS_CSS not in s:
        raise SystemExit("stats-table CSS block not found (already patched?)")
    s = s.replace(OLD_STATS_CSS, NEW_STATS_CSS)

    if OLD_GLOBAL_CSS not in s:
        raise SystemExit("global table CSS block not found (already patched?)")
    s = s.replace(OLD_GLOBAL_CSS, NEW_COMPARE_CSS)

    # stats tables: add group rows (same labels in all 16 cards)
    s = s.replace('<table class="stats-table">',
                  '<table class="stats-table"><tbody>')
    parts = s.split('<table class="stats-table"><tbody>')
    fixed = [parts[0]]
    for part in parts[1:]:
        end = part.find("</table>")
        table, rest = part[:end], part[end:]
        table = table.replace(
            '<tr><th>Wall Time</th>',
            '<tr class="group-row"><th colspan="2">Performance</th></tr>'
            '<tr><th>Wall Time</th>')
        table = table.replace(
            '<tr><th>Peak VRAM</th>',
            '<tr class="group-row"><th colspan="2">Memory</th></tr>'
            '<tr><th>Peak VRAM</th>')
        table = table.replace(
            '<tr><th>Output Size</th>',
            '<tr class="group-row"><th colspan="2">Output</th></tr>'
            '<tr><th>Output Size</th>')
        table = table.replace("</tbody>", "")  # avoid double tbody
        table += "</tbody>"
        fixed.append('<table class="stats-table"><tbody>' + table + rest)
    s = "".join(fixed)

    # rebuild the full comparison table
    start = s.find('<section id="comparison-table">')
    if start < 0:
        raise SystemExit("comparison section not found")
    cstart = s.find('<div class="comparison-table-container">', start)
    tstart = s.find("<table>", cstart)
    tend = s.find("</table>", tstart)
    cend = s.find("</div>", tend)
    if min(cstart, tstart, tend, cend) < 0:
        raise SystemExit("comparison table container not found")
    rows = parse_rows(s[tstart:tend])
    if len(rows) != 16:
        raise SystemExit(f"expected 16 rows, parsed {len(rows)}")
    new_table = build_full_table(rows)
    legend = ('<div class="table-legend">Green pill = best value for that duration. '
              'RTF = real-time factor (wall time ÷ output duration; lower is faster). '
              'All 16 runs completed.</div>')
    s = (s[:cstart] + '<div class="comparison-table-container">' + new_table
         + '</div>' + legend + s[cend + len("</div>"):])

    PATH.write_text(s, encoding="utf-8")
    print(f"restyled {PATH} ({len(rows)} comparison rows)")


if __name__ == "__main__":
    main()
