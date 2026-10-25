"""Generate the revised two-column 1200 x 900 mm ASCOMP poster.

All geometry is vector; only the original, unchanged QR is a bitmap.
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
    'layout': {'columns': 2, 'revision': 'split-identity-objective-banner', 'column_width_pt': CW},
    'text': [], 'fit_checks': [], 'data': ROWS,
    'glyph_audit': {'checked_characters': 0, 'missing': []},
    'provenance': {
        'paper': 'output/pdf/MinPI_arXiv_final.pdf',
        'compression_data': 'Table 2: five-seed means and sample SD',
        'aggregate_removal': 'Section 4.1: all 30 dataset-seed units',
        'nsga_ii': 'Section 4.3: held-out hypervolume, not Figure 3 validation curves',
        'random_comparison': 'Section 4.1: Madelon mean held-out delta utility, matched deletion counts',
        'objective_banner': 'Section 3.1: minimize normalized feature count and prediction loss in NSGA-II',
        'diagram': 'Conceptual illustration of an evaluated -1 marginal; not additional data',
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
    para('Min-Marginal Power Indices', M, 215, W - 2 * M, 124, leading=138,
         bold=True, color=NAVY, align=TA_CENTER, max_h=142)
    para('for Multi-Objective Feature Selection', M, 352, W - 2 * M, 124, leading=138,
         bold=True, color=NAVY, align=TA_CENTER, max_h=142)
    para('Wenjie Shi  &amp;  Chi Zhao*', X1, 541, CW, 76,
         bold=True, color=NAVY, align=TA_LEFT, max_h=96)
    para('Saint Petersburg State University', X1, 646, CW, 52,
         align=TA_LEFT, max_h=67)
    para('7/9 Universitetskaya nab., Saint Petersburg, 199034, Russia',
         X2, 542, CW, 42, align=TA_RIGHT, max_h=55)
    para('st147487@student.spbu.ru', X2, 606, CW, 40,
         align=TA_RIGHT, color=MUTED, max_h=52)
    para('* Corresponding author: chi.zhao@spbu.ru', X2, 666, CW, 40,
         align=TA_RIGHT, color=MUTED, max_h=52)

    # Reuse the space released by the two-sided author block for the actual
    # optimization objectives, rather than repeating the title or result values.
    rect(M, 755, W - 2 * M, 106, NAVY, radius=16)
    para('OPTIMIZATION<br/>OBJECTIVES', M + 30, 770, 490, 34, leading=39,
         bold=True, color='#81D9D1', max_h=80)
    line(M + 550, 773, M + 550, 843, '#4A6578', 2)
    para('Minimize feature count  +  Maximize predictive utility',
         M + 600, 776, W - 2 * M - 650, 52, leading=64,
         bold=True, color=WHITE, align=TA_CENTER, max_h=67)


def section(x, text):
    para(text, x, 923, CW, 54, bold=True, color=NAVY, max_h=69)
    line(x, 1006, x + CW, 1006, RULE, 2)


def method():
    section(X1, '01  CORE IDEA &amp; METHOD')
    para('<b>From average importance to worst-case evidence.</b><br/>MinPI detects feature additions that push utility below the quota.',
         X1, 1040, CW, 43, leading=55, max_h=114)
    rect(X1, 1190, CW, 229, PALE, radius=18)
    bw = 430
    rect(X1 + 28, 1220, bw, 165, LIGHT, radius=12)
    rect(X1 + CW - bw - 28, 1220, bw, 165, WARM, radius=12)
    para('Acceptable', X1 + 28, 1240, bw, 42, bold=True, color=TEAL, align=TA_CENTER)
    para('v(S) = 1', X1 + 28, 1305, bw, 48, color=TEAL, align=TA_CENTER)
    para('Unacceptable', X1 + CW - bw - 28, 1240, bw, 42, bold=True,
         color=CORAL, align=TA_CENTER)
    para('v(S ∪ {i}) = 0', X1 + CW - bw - 28, 1309, bw, 43, color=CORAL, align=TA_CENTER)
    ax1, ax2, ay = X1 + 494, X1 + CW - 494, 1307
    line(ax1, ay, ax2, ay, NAVY, 5)
    line(ax2 - 20, ay - 14, ax2, ay, NAVY, 5)
    line(ax2 - 20, ay + 14, ax2, ay, NAVY, 5)
    para('+ feature i', X1 + 475, 1228, CW - 950, 42, align=TA_CENTER)
    para('marginal = -1', X1 + 475, 1333, CW - 950, 39, bold=True,
         color=CORAL, align=TA_CENTER)

    rect(X1, 1460, CW, 303, LIGHT, radius=18)
    para('THE WORST-CASE MARGINAL', X1 + 30, 1480, CW - 60, 34,
         bold=True, color=TEAL, max_h=44)
    para('v<sub>α</sub>(S) = 1[U(M<sub>S</sub>) ≥ α U<sub>full</sub>]',
         X1 + 30, 1541, CW - 60, 53, leading=68, color=NAVY, max_h=72)
    para('MinPI<sub>i</sub> = min [v<sub>α</sub>(S ∪ {i}) - v<sub>α</sub>(S)]',
         X1 + 30, 1630, CW - 60, 51, leading=66, bold=True, color=TEAL, max_h=71)
    para('over S ⊆ N ∖ {i}; N = analyzed features', X1 + 30, 1711, CW - 60,
         36, color=MUTED, max_h=46)

    para('Validation-gated feature removal', X1, 1801, CW, 48, bold=True, color=NAVY, max_h=61)
    para('H: features with an evaluated harmful coalition witness (-1).',
         X1, 1871, CW, 38, max_h=50)
    para('U<sub>val</sub>(N ∖ H) ≥ U<sub>val</sub>(N) - ε',
         X1, 1940, CW, 52, leading=66, color=TEAL, bold=True, max_h=70)
    para('ε = 0.005 (classification); 0.01 (regression).', X1, 2022, CW,
         37, max_h=48)
    para('Pass: remove H. Fail: remove nothing. Models fit on training data.',
         X1, 2081, CW, 37, max_h=48)

    qx, qy, qw = X1 + 8, 2178, 90 * mm
    qh = qw * 1320 / 1100
    c.drawImage(str(ROOT / 'output/QRcode.png'), qx, H - qy - qh, qw, qh,
                preserveAspectRatio=True, mask='auto')
    c.linkURL('https://hlnks.co/63671101', (qx, H - qy - qh, qx + qw, H - qy), relative=0)
    para('CORE INSIGHT', X1 + 316, 2190, CW - 316, 35,
         bold=True, color=TEAL, max_h=45)
    para('From harmful-coalition evidence<br/>to validation-gated feature removal.',
         X1 + 316, 2252, CW - 316, 43, leading=55, bold=True, color=NAVY, max_h=114)
    para('Scan to read the full paper', X1 + 316, 2399, CW - 316, 36, max_h=46)
    para('hlnks.co/63671101', X1 + 316, 2450, CW - 316, 33, color=TEAL, max_h=43)
    manifest['qr'] = {'x_pt': qx, 'y_top_pt': qy, 'width_mm': 90, 'height_mm': 108,
                       'effective_dpi': 1100 / (90 / 25.4), 'payload': 'https://hlnks.co/63671101'}


def results():
    section(X2, '02  EXPERIMENTS &amp; RESULTS')
    para('6 datasets × 5 seeds  |  XGBoost  |  α = 0.95<br/>64/16/20% train/validation/test; ≤100 analyzed features.',
         X2, 1040, CW, 41, leading=55, max_h=114)
    para('Five-seed mean ± SD; removal of analyzed features.<br/>ΔU = held-out utility - full-model utility.',
         X2, 1184, CW, 36, leading=47, color=MUTED, max_h=98)
    para('DATASET', X2, 1309, 430, 35, bold=True, color=TEAL, max_h=45)
    para('REMOVED (%)', X2 + 470, 1309, 590, 35, bold=True,
         color=TEAL, align=TA_CENTER, max_h=45)
    para('HELD-OUT ΔU', X2 + 1120, 1309, CW - 1120, 35,
         bold=True, color=TEAL, align=TA_RIGHT, max_h=45)
    line(X2, 1368, X2 + CW, 1368, RULE)
    chartx, chartw = X2 + 470, 590
    for i, row in enumerate(ROWS):
        y = 1390 + i * 100
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
        para(str(tick), xx - 22, 1992, 70, 34, color=MUTED)

    rect(X2, 2068, CW, 164, NAVY, radius=18)
    para('12.53%', X2 + 30, 2082, 660, 68, leading=78,
         bold=True, color='#81D9D1', align=TA_CENTER)
    para('mean feature removal', X2 + 30, 2176, 660, 35, color=WHITE, align=TA_CENTER)
    line(X2 + CW / 2, 2093, X2 + CW / 2, 2207, '#4A6578', 2)
    para('-0.000313', X2 + 800, 2082, CW - 830, 68, leading=78,
         bold=True, color=WHITE, align=TA_CENTER)
    para('mean held-out ΔU', X2 + 800, 2176, CW - 830, 35, color=WHITE, align=TA_CENTER)
    para('<b>NSGA-II:</b> mean held-out ΔHV +0.01584; 800 evaluations/variant.<br/>'
         'Madelon: +0.09515. Overall Holm p = 0.4375.',
         X2, 2271, CW, 37, leading=48, max_h=100)
    rect(X2, 2390, CW, 116, LIGHT, radius=16)
    para('MADELON: MATCHED-COUNT REMOVAL', X2 + 25, 2400, CW - 50,
         33, bold=True, color=TEAL, max_h=43)
    para('MinPI ΔU +0.0047  |  Random ΔU -0.0732', X2 + 25, 2450, CW - 50,
         40, bold=True, color=NAVY, max_h=51)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    for relative in ['output/pdf/MinPI_arXiv_final.pdf', 'output/QRcode.png', 'MinPI_arXiv_source(1).zip']:
        manifest['source_hashes'][relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    with Image.open(ROOT / 'output/QRcode.png') as qr:
        assert qr.size == (1100, 1320)
    c = canvas.Canvas(str(PDF), pagesize=(W, H), pageCompression=1,
                      initialFontName='Arial', initialFontSize=40, initialLeading=49)
    c.setTitle('ASCOMP 2026 Poster - Min-Marginal Power Indices for Multi-Objective Feature Selection')
    c.setAuthor('Wenjie Shi; Chi Zhao')
    c.setSubject('120 x 90 cm; two-column poster; centered titles and split author/contact block')
    rect(0, 0, W, H, WHITE)
    header()
    line(W / 2, 930, W / 2, 2490, RULE, 2)
    method()
    results()
    c.showPage()
    c.save()
    (OUT / 'layout_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Created: {PDF.name}; 1200 x 900 mm; two columns; {len(manifest["text"])} text elements')
