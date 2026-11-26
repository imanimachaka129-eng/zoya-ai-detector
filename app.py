import io
import re
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime
from xml.sax.saxutils import escape

from flask import Flask, jsonify, request, render_template_string, send_file

from detector import analyze
from humanizer import humanize

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB uploads

W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def read_docx(data):
    """Extract paragraph text from a .docx without extra dependencies."""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    paras = []
    for p in root.iter(W_NS + "p"):
        txt = "".join(t.text or "" for t in p.iter(W_NS + "t"))
        if txt.strip():
            paras.append(txt)
    return "\n\n".join(paras)


def read_upload(f):
    name = (f.filename or "").lower()
    data = f.read()
    if name.endswith(".docx"):
        return read_docx(data)
    if name.endswith(".txt"):
        for enc in ("utf-8", "utf-16", "latin-1"):
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                continue
    raise ValueError("Aina ya faili haikubaliki. Tumia .txt au .docx.")

PAGE = """<!doctype html>
<html lang="sw">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Detector &amp; Humanizer</title>
<style>
  :root { --bg:#0f1420; --card:#171e2e; --ink:#e8ecf4; --mute:#8a94a8; --acc:#4f8cff; --ok:#33c481; --warn:#f2b441; --bad:#ef5b5b; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:system-ui,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--ink); }
  main { max-width:980px; margin:0 auto; padding:24px 16px 60px; }
  h1 { font-size:1.5rem; margin:0 0 4px; }
  p.sub { color:var(--mute); margin:0 0 20px; }
  .tabs { display:flex; gap:8px; margin-bottom:14px; }
  .tab { padding:8px 16px; border-radius:8px; background:var(--card); color:var(--mute); border:0; cursor:pointer; font-size:.95rem; }
  .tab.on { background:var(--acc); color:#fff; }
  textarea { width:100%; min-height:220px; padding:12px; border-radius:10px; border:1px solid #2a3350; background:var(--card); color:var(--ink); font-size:1rem; resize:vertical; }
  .row { display:flex; gap:10px; align-items:center; margin:12px 0; flex-wrap:wrap; }
  button.go { background:var(--acc); color:#fff; border:0; padding:10px 20px; border-radius:8px; font-size:1rem; cursor:pointer; }
  select { background:var(--card); color:var(--ink); border:1px solid #2a3350; padding:9px; border-radius:8px; }
  .card { background:var(--card); border-radius:12px; padding:16px; margin-top:14px; }
  .big { font-size:2.4rem; font-weight:700; }
  .bar { height:8px; border-radius:6px; background:#26304a; overflow:hidden; margin:4px 0 10px; }
  .bar > i { display:block; height:100%; background:var(--acc); }
  .sig { display:grid; grid-template-columns:150px 1fr 40px; gap:8px; align-items:center; font-size:.9rem; color:var(--mute); }
  .sent { padding:6px 8px; border-radius:6px; margin:4px 0; line-height:1.5; }
  .hidden { display:none; }
  small { color:var(--mute); }
  #hout { white-space:pre-wrap; line-height:1.6; }
</style>
</head>
<body>
<main>
  <h1>AI Detector &amp; Humanizer</h1>
  <p class="sub">Chambua maandishi (English / Kiswahili) na uyaboreshe yasomeke kiasili zaidi.</p>

  <div class="tabs">
    <button class="tab on" data-t="detect">Detector</button>
    <button class="tab" data-t="human">Humanizer</button>
  </div>

  <div class="row">
    <label>Pakia faili (.txt / .docx): <input type="file" id="file" accept=".txt,.docx"></label>
    <small id="fmsg"></small>
  </div>
  <textarea id="txt" placeholder="Bandika maandishi hapa (angalau maneno 40)..."></textarea>

  <div id="detect" class="pane">
    <div class="row">
      <button class="go" id="bd">Chunguza</button>
      <button class="go" style="background:#2a3350" onclick="exportDoc('detect','pdf')">Hifadhi PDF</button>
      <button class="go" style="background:#2a3350" onclick="exportDoc('detect','docx')">Hifadhi DOCX</button>
    </div>
    <div id="dres"></div>
  </div>

  <div id="human" class="pane hidden">
    <div class="row">
      <label>Kiwango:
        <select id="lvl"><option>light</option><option selected>medium</option><option>strong</option></select>
      </label>
      <button class="go" id="bh">Humanize</button>
      <button class="go" id="bc" style="background:#2a3350">Copy</button>
      <button class="go" style="background:#2a3350" onclick="exportDoc('humanize','pdf')">Hifadhi PDF</button>
      <button class="go" style="background:#2a3350" onclick="exportDoc('humanize','docx')">Hifadhi DOCX</button>
    </div>
    <div class="card hidden" id="hbox"><div id="hout"></div><small id="hmeta"></small></div>
    <div id="hcheck"></div>
  </div>
</main>

<script>
const $ = s => document.querySelector(s);
document.querySelectorAll('.tab').forEach(b => b.onclick = () => {
  document.querySelectorAll('.tab').forEach(x => x.classList.toggle('on', x === b));
  document.querySelectorAll('.pane').forEach(p => p.classList.toggle('hidden', p.id !== b.dataset.t));
});
const col = v => v >= 70 ? 'var(--bad)' : v >= 50 ? 'var(--warn)' : 'var(--ok)';
const esc = s => s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

function render(r) {
  if (r.error) return `<div class="card">${esc(r.error)}</div>`;
  const sigs = Object.entries(r.signals).map(([k,v]) =>
    `<div class="sig"><span>${k}</span><div class="bar"><i style="width:${v}%;background:${col(v)}"></i></div><span>${v}</span></div>`).join('');
  const sents = r.sentences.map(s =>
    `<div class="sent" style="background:rgba(${s.ai_score>=55?'239,91,91':s.ai_score>=40?'242,180,65':'51,196,129'},.16)">${esc(s.text)} <small>(${s.ai_score})</small></div>`).join('');
  const det = Object.entries(r.details).map(([k,v]) =>
    `<small>• ${k}: ${Array.isArray(v) ? (v.join(', ') || 'none') : esc(String(v))}</small>`).join('<br>');
  return `<div class="card">
    <div class="big" style="color:${col(r.ai_probability)}">${r.ai_probability}%</div>
    <div>${r.label}</div>
    <small>${r.word_count} words · ${r.sentence_count} sentences · lang: ${r.language} · perplexity: ${r.used_perplexity ? 'on' : 'off'}</small>
    <h4>Ishara (signals)</h4>${sigs}<p>${det}</p>
    <h4>Sentensi kwa sentensi</h4>${sents}
    <p><small>${r.disclaimer}</small></p></div>`;
}

async function post(url, body) {
  const r = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  return r.json();
}

$('#bd').onclick = async () => {
  $('#dres').innerHTML = '<div class="card">Inachambua...</div>';
  $('#dres').innerHTML = render(await post('/api/detect', {text: $('#txt').value}));
};
$('#bh').onclick = async () => {
  const r = await post('/api/humanize', {text: $('#txt').value, level: $('#lvl').value});
  $('#hbox').classList.remove('hidden');
  $('#hout').textContent = r.text || r.error;
  lastH = r.text ? {text: r.text, level: r.meta.level, language: r.meta.language, before: r.before, after: r.after} : null;
  $('#hmeta').textContent = r.meta ? `level: ${r.meta.level} · lang: ${r.meta.language}` : '';
  $('#hcheck').innerHTML = r.after ? `<div class="card"><b>Kabla:</b> ${r.before}% → <b>Baada:</b> ${r.after}%<br><small>Soma tena maandishi na uhakikishe bado yanamaanisha unachokusudia.</small></div>` : '';
};
$('#bc').onclick = () => navigator.clipboard.writeText($('#hout').textContent);

let lastH = null;
async function exportDoc(kind, fmt) {
  const body = {kind: kind, format: fmt};
  if (kind === 'humanize') {
    if (!lastH) { alert('Bofya Humanize kwanza.'); return; }
    Object.assign(body, lastH);
  } else {
    body.text = $('#txt').value;
  }
  const res = await fetch('/api/export', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  if (!res.ok) {
    let m = 'Imeshindwa kuhifadhi.';
    try { m = (await res.json()).error || m; } catch (e) {}
    alert(m);
    return;
  }
  const blob = await res.blob();
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = (kind === 'humanize' ? 'humanized' : 'ai-report') + '.' + fmt;
  document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(a.href);
}
$('#file').onchange = async e => {
  const f = e.target.files[0];
  if (!f) return;
  const fd = new FormData(); fd.append('file', f);
  $('#fmsg').textContent = 'Inasoma...';
  try {
    const r = await (await fetch('/api/extract', {method:'POST', body: fd})).json();
    if (r.error) { $('#fmsg').textContent = r.error; return; }
    $('#txt').value = r.text;
    $('#fmsg').textContent = `${f.name}: maneno ${r.text.split(/\\s+/).filter(Boolean).length}`;
  } catch (err) { $('#fmsg').textContent = 'Imeshindwa kusoma faili.'; }
};
</script>
</body>
</html>"""


@app.get("/")
def index():
    return render_template_string(PAGE)


# ----------------------------------------------------------------- export
# A report is a list of tuples: (kind, value[, extra]) rendered to PDF or DOCX.
def _detect_model(text):
    r = analyze(text)
    if "error" in r:
        raise ValueError(r["error"])
    m = [
        ("title", "Ripoti ya AI Detector"),
        ("meta", datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("kv", [("Uwezekano wa AI", f"{r['ai_probability']}%"),
                ("Hitimisho", r["label"]),
                ("Lugha", "Kiswahili" if r["language"] == "sw" else "Kiingereza"),
                ("Maneno", r["word_count"]),
                ("Sentensi", r["sentence_count"])]),
        ("h", "Ishara (0-100, juu = inafanana zaidi na AI)"),
        ("table", [["Ishara", "Alama"]] + [[k, v] for k, v in r["signals"].items()], [11, 5]),
    ]
    det = []
    for k, v in r["details"].items():
        det.append(f"{k}: " + (", ".join(v) if isinstance(v, list) and v else
                               ("hakuna" if isinstance(v, list) else str(v))))
    if det:
        m += [("h", "Maelezo"), ("p", "\n\n".join(det))]
    m += [("h", "Sentensi kwa sentensi"),
          ("table", [["#", "Sentensi", "Alama"]] +
           [[i + 1, s["text"], s["ai_score"]] for i, s in enumerate(r["sentences"])], [1.2, 12.3, 2.5]),
          ("note", r["disclaimer"])]
    return m


def _humanize_model(text, d):
    kv = [("Kiwango", d.get("level", "")), ("Lugha", d.get("language", ""))]
    if d.get("before") is not None and d.get("after") is not None:
        kv += [("Alama ya AI kabla", f"{d['before']}%"), ("Alama ya AI baada", f"{d['after']}%")]
    return [
        ("title", "Maandishi Yaliyoboreshwa (Humanized)"),
        ("meta", datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("kv", kv),
        ("h", "Maandishi"),
        ("p", text),
        ("note", "Soma tena maandishi na uhakikishe yanamaanisha unachokusudia. "
                 "Alama za detector ni makadirio, si ushahidi."),
    ]


def render_docx(model):
    from docx import Document
    doc = Document()
    for item in model:
        kind, val = item[0], item[1]
        if kind == "title":
            doc.add_heading(val, 0)
        elif kind == "meta":
            doc.add_paragraph(val)
        elif kind == "h":
            doc.add_heading(val, 1)
        elif kind == "p":
            for chunk in val.split("\n\n"):
                doc.add_paragraph(chunk)
        elif kind == "kv":
            t = doc.add_table(rows=0, cols=2)
            t.style = "Table Grid"
            for k, v in val:
                cells = t.add_row().cells
                cells[0].text, cells[1].text = str(k), str(v)
                for run in cells[0].paragraphs[0].runs:
                    run.bold = True
        elif kind == "table":
            t = doc.add_table(rows=len(val), cols=len(val[0]))
            t.style = "Table Grid"
            for i, row in enumerate(val):
                for j, c in enumerate(row):
                    t.cell(i, j).text = str(c)
            for c in t.rows[0].cells:
                for run in c.paragraphs[0].runs:
                    run.bold = True
        elif kind == "note":
            run = doc.add_paragraph().add_run(val)
            run.italic = True
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf


def render_pdf(model):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    ss = getSampleStyleSheet()
    body = ss["BodyText"]

    def para(t, style=body):
        return Paragraph(escape(str(t)).replace("\n", "<br/>"), style)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm, title="AI Detector")
    story = []
    for item in model:
        kind, val = item[0], item[1]
        widths = item[2] if len(item) > 2 else None
        if kind == "title":
            story.append(Paragraph(escape(val), ss["Title"]))
        elif kind == "meta":
            story += [para(val), Spacer(1, 10)]
        elif kind == "h":
            story += [Spacer(1, 8), Paragraph(escape(val), ss["Heading2"])]
        elif kind == "p":
            for chunk in val.split("\n\n"):
                story += [para(chunk), Spacer(1, 6)]
        elif kind in ("kv", "table"):
            rows = [[para(c) for c in r] for r in val]
            if kind == "kv":
                widths = [5, 11]
            t = Table(rows, colWidths=[w * cm for w in widths] if widths else None, repeatRows=1 if kind == "table" else 0)
            style = [("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
                     ("VALIGN", (0, 0), (-1, -1), "TOP")]
            if kind == "table":
                style.append(("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke))
            t.setStyle(TableStyle(style))
            story += [t, Spacer(1, 8)]
        elif kind == "note":
            story += [Spacer(1, 10), Paragraph("<i>" + escape(val) + "</i>", body)]
    doc.build(story)
    buf.seek(0)
    return buf


@app.post("/api/export")
def api_export():
    d = request.get_json(force=True, silent=True) or {}
    fmt, kind = d.get("format"), d.get("kind")
    text = (d.get("text") or "").strip()
    try:
        if kind == "humanize":
            if not text:
                raise ValueError("Hakuna maandishi ya kuhifadhi. Bofya Humanize kwanza.")
            model = _humanize_model(text, d)
        else:
            model = _detect_model(text)
        if fmt == "docx":
            buf = render_docx(model)
            mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        elif fmt == "pdf":
            buf = render_pdf(model)
            mime = "application/pdf"
        else:
            raise ValueError("Format haikubaliki (tumia pdf au docx).")
    except ImportError:
        return jsonify({"error": "Sakinisha maktaba: pip install reportlab python-docx"}), 500
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    name = ("humanized" if kind == "humanize" else "ai-report") + datetime.now().strftime("-%Y%m%d-%H%M") + "." + fmt
    return send_file(buf, mimetype=mime, as_attachment=True, download_name=name)


@app.post("/api/extract")
def api_extract():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "Hakuna faili lililopakiwa."}), 400
    try:
        text = read_upload(f)
    except (ValueError, KeyError, zipfile.BadZipFile, ET.ParseError) as e:
        msg = str(e) if isinstance(e, ValueError) else "Faili haliwezi kusomwa (huenda limeharibika)."
        return jsonify({"error": msg}), 400
    return jsonify({"text": text})


@app.errorhandler(413)
def too_big(_e):
    return jsonify({"error": "Faili ni kubwa mno (kikomo 5 MB)."}), 413


@app.post("/api/detect")
def api_detect():
    data = request.get_json(force=True, silent=True) or {}
    return jsonify(analyze(data.get("text", "")))


@app.post("/api/humanize")
def api_humanize():
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or "").strip()
    if len(text.split()) < 10:
        return jsonify({"error": "Weka maandishi marefu zaidi."}), 400
    out, meta = humanize(text, level=data.get("level", "medium"))
    before = analyze(text)
    after = analyze(out)
    return jsonify({
        "text": out,
        "meta": meta,
        "before": before.get("ai_probability"),
        "after": after.get("ai_probability"),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
