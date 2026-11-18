"""Generate the revised two-column 1200 x 900 mm ASCOMP poster.

All geometry is vector; only the original, unchanged paper QR is a bitmap.
Prior revisions are retained under versions/.
"""
import hashlib
import json
from pathlib import Path
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph
from reportlab.graphics.barcode.qr import QrCodeWidget
from reportlab.graphics.shapes import Drawing
from reportlab.graphics import renderPDF

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/poster'
PDF = ROOT / 'output/pdf/ASCOMP2026_MinPI_Poster_120x90cm.pdf'
W, H = 1200 * mm, 900 * mm
M, GAP = 100, 104
CW = (W - 2 * M - GAP) / 2
X1, X2 = M, M + CW + GAP
NAVY, INK, TEAL = '#142F46', '#243F50', '#087E83'
LIGHT, PALE, MUTED = '#EDF6F5', '#F2F5F7', '#4D6472'
RULE, CORAL, WARM, WHITE = '#C9D6DC', '#A64E37', '#FAF1ED', '#FFFFFF'
ROWS = [
    {'dataset': 'Madelon', 'compression': 43.60, 'sd': 31.96, 'du': .00469, 'du_sd': .00497},
    {'dataset': 'Abalone', 'compression': 27.50, 'sd': 25.62, 'du': -.00565, 'du_sd': .01077},
    {'dataset': 'Credit-Card-Fraud', 'compression': 2.76, 'sd': 2.89, 'du': -.00078, 'du_sd': .00335},
    {'dataset': 'Spambase', 'compression': .72, 'sd': .99, 'du': -.00003, 'du_sd': .00007},
    {'dataset': 'Gisette', 'compression': .60, 'sd': .89, 'du': -.00011, 'du_sd': .00016},
    {'dataset': 'ISOLET', 'compression': 0., 'sd': 0., 'du': 0., 'du_sd': 0.},
]

for name, filename in [('Arial', 'arial.ttf'), ('Arial-Bold', 'arialbd.ttf'),
                       ('Arial-Italic', 'ariali.ttf'), ('Arial-BoldItalic', 'arialbi.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(Path('C:/Windows/Fonts') / filename)))
pdfmetrics.registerFontFamily('Arial', normal='Arial', bold='Arial-Bold',
                             italic='Arial-Italic', boldItalic='Arial-BoldItalic')
pdfmetrics.registerFont(TTFont('PosterMath', 'C:/Windows/Fonts/cambria.ttc', subfontIndex=1))
manifest = {
    'page': {'width_pt': W, 'height_pt': H, 'width_mm': 1200, 'height_mm': 900},
    'layout': {'columns': 2, 'revision': 'spacious-algorithm-dual-qr-v5', 'column_width_pt': CW},
    'text': [], 'fit_checks': [], 'data': ROWS,
    'glyph_audit': {'checked_characters': 0, 'missing': []},
    'provenance': {
        'paper': 'output/pdf/MinPI_arXiv_final.pdf',
        'compression_data': 'Table 2: five-seed means and sample SD',
        'aggregate_removal': 'Section 4.1: all 30 dataset-seed units',
        'nsga_ii': 'Section 4.3: held-out hypervolume, not Figure 3 validation curves',
        'random_comparison': 'Section 4.1: Madelon and Abalone mean held-out delta utility, matched deletion counts',
        'algorithm': 'Section 2, Section 3, Appendix A: game, A/B/hybrid, all-or-none gate, optional NSGA-II',
        'reference_layout': 'User-supplied poster.pdf used for section/flow organization only',
        'graphics': 'Vector redraw from reported paper values'},
    'source_hashes': {}}


def fill(color):
    c.setFillColor(colors.HexColor(color))


def rect(x, y, w, h, color, radius=0):
    fill(color)
    if radius:
        c.roundRect(x, H - y - h, w, h, radius, fill=1, stroke=0)
    else:
        c.rect(x, H - y - h, w, h, fill=1, stroke=0)


def line(x1, y1, x2, y2, color=RULE, width=2):
    c.setStrokeColor(colors.HexColor(color))
    c.setLineWidth(width)
    c.line(x1, H - y1, x2, H - y2)


def para(text, x, y, w, size=40, leading=None, color=INK, bold=False,
         align=TA_LEFT, max_h=None, label=None):
    for symbol in '∪⊆∖':
        text = text.replace(symbol, f'<font name="PosterMath">{symbol}</font>')
    style = ParagraphStyle('poster', fontName='Arial-Bold' if bold else 'Arial',
                           fontSize=size, leading=leading or size * 1.23,
                           textColor=colors.HexColor(color), alignment=align,
                           spaceBefore=0, spaceAfter=0)
    p = Paragraph(text, style)
    for frag in p.frags:
        for char in getattr(frag, 'text', ''):
            if char.isspace():
                continue
            manifest['glyph_audit']['checked_characters'] += 1
            if not pdfmetrics.getFont(frag.fontName).face.charToGlyph.get(ord(char), 0):
                manifest['glyph_audit']['missing'].append({'char': char, 'font': frag.fontName})
                raise ValueError(f'Missing glyph U+{ord(char):04X} in {frag.fontName}')
    used_w, used_h = p.wrap(w, H)
    if max_h is not None:
        manifest['fit_checks'].append({'label': label or text[:60],
                                      'used_height': used_h, 'available_height': max_h})
        if used_h > max_h + .1:
            raise ValueError(f'Text overflows {label or text[:60]}: {used_h} > {max_h}')
    p.drawOn(c, x, H - y - used_h)
    manifest['text'].append({'text': text, 'x': x, 'y': y, 'width': used_w,
                             'height': used_h, 'font_size': size,
                             'alignment': {TA_LEFT: 'left', TA_CENTER: 'center', TA_RIGHT: 'right'}[align]})
    return used_h


def header():
    rect(0, 0, W, 16, NAVY)
    para('ASCOMP 2026 POSTER', M, 58, W - 2 * M, 96, leading=116,
         bold=True, color=TEAL, align=TA_CENTER, max_h=120)
    para('Min-Marginal Power Indices', M, 207, W - 2 * M, 124, leading=138,
         bold=True, color=NAVY, align=TA_CENTER, max_h=142)
    para('for Multi-Objective Feature Selection', M, 352, W - 2 * M, 124, leading=138,
         bold=True, color=NAVY, align=TA_CENTER, max_h=142)
    para('Wenjie Shi  &amp;  Chi Zhao*', X1, 535, CW, 76,
         bold=True, color=NAVY, align=TA_LEFT, max_h=96)
    para('Saint Petersburg State University', X1, 637, CW, 52,
         align=TA_LEFT, max_h=67)
    para('7/9 Universitetskaya nab., Saint Petersburg, 199034, Russia',
         X2, 535, CW, 42, align=TA_RIGHT, max_h=55)
    para('st147487@student.spbu.ru', X2, 597, CW, 40,
         align=TA_RIGHT, color=MUTED, max_h=52)
    para('* Corresponding author: chi.zhao@spbu.ru', X2, 654, CW, 40,
         align=TA_RIGHT, color=MUTED, max_h=52)

    # The first image already includes SCAN ME. No visible URL or link annotation.
    qw, qy, qgap = 90 * mm, 36, 36
    qh = qw * 1320 / 1100
    qx = W - M - 2 * qw - qgap
    c.drawImage(str(ROOT / 'output/QRcode.png'), qx, H - qy - qh, qw, qh,
                preserveAspectRatio=True, mask='auto')
    tx = qx + qw + qgap
    widget = QrCodeWidget('https://t.me/+79516564253', barBorder=4, barLevel='M')
    x0, y0, x1, y1 = widget.getBounds()
    drawing = Drawing(qw, qw, transform=[qw / (x1-x0), 0, 0, qw / (y1-y0), 0, 0])
    # renderPDF otherwise initializes Times-Roman even for a text-free drawing.
    drawing.initialFontName = 'Arial'
    drawing.add(widget)
    rect(tx, qy, qw, qw, WHITE)
    renderPDF.draw(drawing, c, tx, H - qy - qw)
    para('Contact author', tx - 12, 304, qw + 24, 31, bold=True, color=NAVY,
         align=TA_CENTER, max_h=40)
    para('Telegram', tx - 12, 344, qw + 24, 30, color=TEAL,
         align=TA_CENTER, max_h=39)
    manifest['qrs'] = [
        {'x_pt': qx, 'y_top_pt': qy, 'width_mm': 90, 'height_mm': 108,
         'effective_dpi': 1100 / (90 / 25.4), 'format': 'original raster',
         'payload': 'https://hlnks.co/63671101'},
        {'x_pt': tx, 'y_top_pt': qy, 'width_mm': 90, 'height_mm': 90,
         'format': 'vector', 'quiet_zone_modules': 4, 'error_correction': 'M',
         'payload': 'https://t.me/+79516564253'}]
    line(M, 738, W - M, 738, RULE, 2)


def section(x, text):
    para(text, x, 775, CW, 54, bold=True, color=NAVY, max_h=69)
    line(x, 855, x + CW, 855, RULE, 2)


def flow_card(number, title, y, height):
    rect(X1, y, CW, height, PALE, radius=18)
    fill(TEAL)
    c.circle(X1 + 48, H - y - 42, 29, fill=1, stroke=0)
    para(str(number), X1 + 19, y + 22, 58, 30, leading=37, bold=True,
         color=WHITE, align=TA_CENTER, max_h=38)
    para(title, X1 + 95, y + 16, CW - 125, 42, bold=True, color=NAVY, max_h=55)


def flow_arrow(y1, y2):
    xx = X1 + 48
    line(xx, y1 + 6, xx, y2 - 6, TEAL, 3)
    line(xx - 9, y2 - 17, xx, y2 - 6, TEAL, 3)
    line(xx + 9, y2 - 17, xx, y2 - 6, TEAL, 3)


def method():
    section(X1, '01  CORE IDEA &amp; ALGORITHM')
    para('<b>From average importance to worst-case evidence.</b><br/>'
         'A harmful coalition witness: adding a feature turns v = 1 into 0.',
         X1, 885, CW, 42, leading=55, max_h=114)
    rect(X1, 1030, CW, 250, LIGHT, radius=18)
    para('v<sub>α</sub>(S) = 1[U<sub>val</sub>(S) ≥ α U<sub>val</sub>(N)]',
         X1 + 32, 1058, CW - 64, 50, leading=65, color=NAVY, max_h=69)
    para('MinPI<sub>i</sub> = min [v<sub>α</sub>(S ∪ {i}) - v<sub>α</sub>(S)]',
         X1 + 32, 1143, CW - 64, 50, leading=65, bold=True, color=TEAL, max_h=69)
    para('over S ⊆ N ∖ {i}; N = analyzed features', X1 + 32, 1218, CW - 64,
         35, color=MUTED, max_h=45)
    para('ALGORITHM FLOW', X1, 1308, CW, 36, bold=True, color=TEAL, max_h=46)

    flow_card(1, 'Prepare the game', 1376, 180)
    para('Split 64% / 16% / 20% into train / validation / test.<br/>'
         'Fit coalition models on train; evaluate the game on validation.',
         X1 + 32, 1458, CW - 64, 36, leading=45, max_h=92)
    flow_arrow(1556, 1590)

    flow_card(2, 'Estimate MinPI &amp; collect witnesses', 1590, 364)
    para('<b>A:</b> sample coalition size, then a subset, uniformly.<br/>'
         '<b>B:</b> perturb a training-only MI-correlation seed<br/>'
         '(drop each member with p = 0.3; add 0-5 features).<br/>'
         'Take the minimum observed marginal; H = features with estimate -1.',
         X1 + 32, 1672, CW - 64, 35, leading=44, max_h=178)
    para('n ≤ 20: exact  |  20 &lt; n ≤ 75: B*  |  n &gt; 75: min(A, B)',
         X1 + 32, 1867, CW - 64, 32, bold=True, color=TEAL, max_h=41)
    para('*Use A if correlation neighbors are unavailable; n = |N|.',
         X1 + 32, 1912, CW - 64, 31, color=MUTED, max_h=39)
    flow_arrow(1954, 1988)

    flow_card(3, 'Validate joint removal', 1988, 250)
    para('U<sub>val</sub>(N ∖ H) ≥ U<sub>val</sub>(N) - ε',
         X1 + 32, 2070, CW - 64, 50, leading=63, bold=True, color=TEAL, max_h=66)
    para('ε = 0.005 (classification); 0.01 (regression).',
         X1 + 32, 2139, CW - 64, 36, max_h=46)
    para('<b>PASS:</b> remove all H.  <b>FAIL:</b> keep all features.',
         X1 + 32, 2190, CW - 64, 35, max_h=44)
    flow_arrow(2238, 2272)

    flow_card(4, 'Evaluate (optional NSGA-II search)', 2272, 180)
    para('Select on validation; optionally run MinPI-prefiltered NSGA-II.<br/>'
         'Finally report held-out utility and feature compression.',
         X1 + 32, 2354, CW - 64, 36, leading=45, max_h=92)


def results():
    section(X2, '02  RESULTS &amp; EVIDENCE')
    para('6 datasets × 5 seeds  |  XGBoost', X2, 885, CW, 46,
         bold=True, color=NAVY, max_h=59)
    para('α = 0.95; training-only preprocessing; ≤100 analyzed features.<br/>'
         'Utilities: ROC AUC, OVR AUC or R².',
         X2, 959, CW, 36, leading=46, max_h=95)
    rect(X2, 1090, CW, 164, NAVY, radius=18)
    para('12.53%', X2 + 30, 1104, 660, 72, leading=82,
         bold=True, color='#81D9D1', align=TA_CENTER)
    para('mean feature removal', X2 + 30, 1198, 660, 35, color=WHITE, align=TA_CENTER)
    line(X2 + CW / 2, 1115, X2 + CW / 2, 1229, '#4A6578', 2)
    para('-0.000313', X2 + 800, 1104, CW - 830, 72, leading=82,
         bold=True, color=WHITE, align=TA_CENTER)
    para('mean held-out ΔU', X2 + 800, 1198, CW - 830, 35, color=WHITE, align=TA_CENTER)
    para('Five-seed mean ± SD; removal of analyzed features.<br/>'
         'ΔU = held-out utility of retained subset - full-model utility.',
         X2, 1288, CW, 34, leading=43, color=MUTED, max_h=88)
    para('DATASET', X2, 1405, 430, 33, bold=True, color=TEAL, max_h=43)
    para('REMOVED (%)', X2 + 470, 1405, 590, 33, bold=True,
         color=TEAL, align=TA_CENTER, max_h=45)
    para('HELD-OUT ΔU', X2 + 1120, 1405, CW - 1120, 33,
         bold=True, color=TEAL, align=TA_RIGHT, max_h=45)
    line(X2, 1455, X2 + CW, 1455, RULE)
    chartx, chartw = X2 + 470, 590
    for i, row in enumerate(ROWS):
        y = 1473 + i * 93
        para(row['dataset'], X2, y + 9, 445, 39, bold=True, color=NAVY, max_h=51)
        para(f"{row['compression']:.2f} ± {row['sd']:.2f}", chartx, y, chartw,
             38, align=TA_CENTER, max_h=50)
        value = f"{row['du']:+.5f}" if row['du'] else '0.00000'
        para(value, X2 + 1120, y - 2, CW - 1120, 41, bold=True,
             color=TEAL if row['du'] > 0 else INK, align=TA_RIGHT, max_h=53)
        para(f"± {row['du_sd']:.5f}", X2 + 1120, y + 48, CW - 1120,
             34, color=MUTED, align=TA_RIGHT, max_h=44)
        yy = y + 63
        rect(chartx, yy, chartw, 16, PALE, radius=5)
        meanpx = chartw * row['compression'] / 80
        if meanpx:
            rect(chartx, yy, max(meanpx, 2), 16, TEAL, radius=min(5, meanpx / 2))
        lo = max(0, row['compression'] - row['sd']) / 80 * chartw
        hi = (row['compression'] + row['sd']) / 80 * chartw
        if hi:
            line(chartx + lo, yy + 8, chartx + hi, yy + 8, NAVY, 2.5)
            line(chartx + lo, yy - 3, chartx + lo, yy + 19, NAVY, 2.5)
            line(chartx + hi, yy - 3, chartx + hi, yy + 19, NAVY, 2.5)
            fill(TEAL)
            c.circle(chartx + meanpx, H - yy - 8, 8, fill=1, stroke=0)
    for tick in [0, 40, 80]:
        xx = chartx + chartw * tick / 80
        para(str(tick), xx - 22, 2045, 70, 34, color=MUTED)
    rect(X2, 2120, CW, 160, LIGHT, radius=16)
    para('MATCHED-COUNT REMOVAL: MinPI vs random (ΔU)',
         X2 + 25, 2133, CW - 50, 31, bold=True, color=TEAL, max_h=40)
    para('Madelon: +0.0047 vs -0.0732', X2 + 25, 2182, CW - 50,
         37, bold=True, color=NAVY, max_h=47)
    para('Abalone: -0.0056 vs -0.0370', X2 + 25, 2231, CW - 50,
         37, bold=True, color=NAVY, max_h=47)
    para('NSGA-II: feature count vs predictive utility', X2, 2320, CW, 39,
         bold=True, color=NAVY, max_h=50)
    para('800 evaluations/variant; mean held-out ΔHV (hypervolume) +0.01584.<br/>'
         'Madelon: +0.09515; overall Holm p = 0.4375.',
         X2, 2390, CW, 36, leading=46, max_h=95)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    for relative in ['output/pdf/MinPI_arXiv_final.pdf', 'output/QRcode.png',
                     'MinPI_arXiv_source(1).zip', 'tmp/pdfs/tele_code.png']:
        manifest['source_hashes'][relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    with Image.open(ROOT / 'output/QRcode.png') as qr:
        assert qr.size == (1100, 1320)
    c = canvas.Canvas(str(PDF), pagesize=(W, H), pageCompression=1,
                      initialFontName='Arial', initialFontSize=40, initialLeading=49)
    c.setTitle('ASCOMP 2026 Poster - Min-Marginal Power Indices for Multi-Objective Feature Selection')
    c.setAuthor('Wenjie Shi; Chi Zhao')
    c.setSubject('120 x 90 cm; two columns; four-step algorithm; two upper-right QR codes')
    rect(0, 0, W, H, WHITE)
    header()
    line(W / 2, 780, W / 2, 2485, RULE, 2)
    method()
    results()
    c.showPage()
    c.save()
    (OUT / 'layout_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Created: {PDF.name}; 1200 x 900 mm; two columns; {len(manifest["text"])} text elements')
