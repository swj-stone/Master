"""Create a 1200 x 900 mm vector poster from verified final-paper results.

Run with the bundled Python. All coordinates use points, measured from top left.
The original article, final ZIP and QR are read-only inputs.
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
M, GAP = 96, 50
CW = (W - 2 * M - 2 * GAP) / 3
X1, X2, X3 = M, M + CW + GAP, M + 2 * (CW + GAP)

NAVY = '#142F46'
INK = '#243F50'
TEAL = '#087E83'
LIGHT = '#EDF6F5'
PALE = '#F2F5F7'
MUTED = '#4D6472'
RULE = '#C9D6DC'
CORAL = '#A64E37'
WARM = '#FAF1ED'
WHITE = '#FFFFFF'

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

manifest = {'page': {'width_pt': W, 'height_pt': H, 'width_mm': 1200, 'height_mm': 900},
            'text': [], 'fit_checks': [], 'data': ROWS,
            'glyph_audit': {'checked_characters': 0, 'missing': []},
            'provenance': {
                'paper': 'output/pdf/MinPI_arXiv_final.pdf',
                'compression_data': 'Table 2: MinPI feature removal summary, five-seed means and sample SD',
                'aggregate_removal': 'Section 4.1: all 30 dataset-seed units',
                'nsga_ii': 'Section 4.3: held-out hypervolume, not the validation curves of Figure 3',
                'diagram': 'Conceptual illustration of an evaluated -1 marginal; not simulated data',
                'graphics': 'Original vector redraw from paper values; no rasterized paper panels'},
            'source_hashes': {}}


def fill(color):
    c.setFillColor(colors.HexColor(color))


def rect(x, y, w, h, color, radius=0, stroke=None, line_width=1):
    fill(color)
    c.setLineWidth(line_width)
    if stroke:
        c.setStrokeColor(colors.HexColor(stroke))
    if radius:
        c.roundRect(x, H - y - h, w, h, radius, fill=1, stroke=bool(stroke))
    else:
        c.rect(x, H - y - h, w, h, fill=1, stroke=bool(stroke))


def line(x1, y1, x2, y2, color=RULE, width=2):
    c.setStrokeColor(colors.HexColor(color))
    c.setLineWidth(width)
    c.line(x1, H - y1, x2, H - y2)


def para(text, x, y, w, size=38, leading=None, color=INK, bold=False,
         align=TA_LEFT, max_h=None, label=None):
    # Arial lacks these set operators. Use embedded Cambria Math only for those glyphs.
    for symbol in '∪⊆∖':
        text = text.replace(symbol, f'<font name="PosterMath">{symbol}</font>')
    style = ParagraphStyle('poster', fontName='Arial-Bold' if bold else 'Arial',
                           fontSize=size, leading=leading or size * 1.23,
                           textColor=colors.HexColor(color), alignment=align,
                           spaceBefore=0, spaceAfter=0, allowWidows=0, allowOrphans=0)
    p = Paragraph(text, style)
    for frag in p.frags:
        for char in getattr(frag, 'text', ''):
            if char.isspace():
                continue
            manifest['glyph_audit']['checked_characters'] += 1
            font = pdfmetrics.getFont(frag.fontName)
            if not font.face.charToGlyph.get(ord(char), 0):
                manifest['glyph_audit']['missing'].append({'char': char, 'font': frag.fontName})
                raise ValueError(f'Missing glyph U+{ord(char):04X} in {frag.fontName}')
    used_w, used_h = p.wrap(w, H)
    if max_h is not None:
        manifest['fit_checks'].append({'label': label or text[:50],
                                       'used_height': used_h, 'available_height': max_h})
        if used_h > max_h + .1:
            raise ValueError(f'Text overflows {label or text[:50]}: {used_h:.1f} > {max_h:.1f}')
    p.drawOn(c, x, H - y - used_h)
    manifest['text'].append({'text': text, 'x': x, 'y': y, 'width': used_w,
                             'height': used_h, 'font_size': size})
    return used_h


def arrow_down(cx, y1, y2):
    line(cx, y1, cx, y2 - 11, TEAL, 5)
    p = c.beginPath()
    p.moveTo(cx - 12, H - y2 + 16)
    p.lineTo(cx, H - y2)
    p.lineTo(cx + 12, H - y2 + 16)
    c.setStrokeColor(colors.HexColor(TEAL))
    c.drawPath(p, fill=0, stroke=1)


def section(x, number, label):
    para(number, x, 826, 106, 50, bold=True, color=TEAL, max_h=65)
    para(label, x + 115, 826, CW - 115, 50, bold=True, color=NAVY, max_h=65)
    line(x, 904, x + CW, 904, RULE, 3)


def header():
    rect(0, 0, W, 14, NAVY)
    para('ASCOMP 2026 POSTER', M, 62, 1700, 42, bold=True, color=TEAL)
    para('COOPERATIVE GAMES  /  FEATURE SELECTION', W - M - 1600, 68, 1600,
         31, color=MUTED, align=TA_RIGHT)
    para('Min-Marginal Power Indices', M, 139, W - 2 * M, 112,
         leading=122, bold=True, color=NAVY, max_h=125)
    para('for Multi-Objective Feature Selection', M, 265, W - 2 * M, 112,
         leading=122, bold=True, color=NAVY, max_h=125)
    para('Wenjie Shi  &amp;  Chi Zhao*', M, 411, W - 2 * M, 51, bold=True,
         max_h=66)
    para('Saint Petersburg State University', M, 485, 1550, 37, max_h=48)
    para('7/9 Universitetskaya nab., Saint Petersburg, 199034, Russia', M, 536,
         1900, 33, max_h=44)
    para('st147487@student.spbu.ru', W - M - 1250, 486, 1250, 33,
         align=TA_RIGHT, max_h=44)
    para('* Corresponding author: chi.zhao@spbu.ru', W - M - 1400, 536, 1400,
         33, align=TA_RIGHT, max_h=44)

    rect(M, 619, W - 2 * M, 158, NAVY, radius=18)
    para('Find when a feature can hurt.<br/>Verify removal before filtering.',
         M + 38, 651, 1480, 43, leading=51, color=WHITE, bold=True, max_h=108)
    line(1700, 645, 1700, 752, '#4A6578', 2)
    para('12.53%', 1750, 637, 640, 72, leading=78, bold=True, color='#81D9D1')
    para('mean analyzed-feature removal', 1750, 725, 670, 30, color=WHITE, max_h=38)
    para('-0.000313', 2480, 637, 775, 72, leading=78, bold=True, color=WHITE)
    para('mean held-out utility change', 2480, 725, 775, 30, color=WHITE, max_h=38)


def problem():
    section(X1, '01', 'THE PROBLEM')
    para('Averages can hide<br/>a harmful context.', X1, 939, CW, 47, leading=55,
         bold=True, color=NAVY, max_h=112)
    para('Adding a feature can turn an acceptable model into an unacceptable one '
         'in a particular coalition.', X1, 1075, CW, 39, leading=49, max_h=149)

    rect(X1, 1250, CW, 268, PALE, radius=18)
    para('ONE COALITION, ONE HARMFUL WITNESS', X1 + 30, 1272, CW - 60,
         31, color=MUTED, bold=True, max_h=40)
    bw = 325
    rect(X1 + 30, 1343, bw, 125, LIGHT, radius=12)
    rect(X1 + CW - bw - 30, 1343, bw, 125, WARM, radius=12)
    para('Acceptable', X1 + 30, 1355, bw, 34, color=TEAL, bold=True, align=TA_CENTER)
    para('v(S) = 1', X1 + 30, 1404, bw, 40, color=TEAL, align=TA_CENTER)
    para('Unacceptable', X1 + CW - bw - 30, 1355, bw, 34, color=CORAL,
         bold=True, align=TA_CENTER)
    para('v(S ∪ {i}) = 0', X1 + CW - bw - 30, 1404, bw, 35, color=CORAL, align=TA_CENTER)
    ax1, ax2, ay = X1 + 377, X1 + CW - 378, 1418
    line(ax1, ay, ax2, ay, NAVY, 4)
    line(ax2 - 15, ay - 11, ax2, ay, NAVY, 4)
    line(ax2 - 15, ay + 11, ax2, ay, NAVY, 4)
    para('+ feature i', X1 + 356, 1357, CW - 712, 32, color=NAVY, align=TA_CENTER)
    para('Δ = -1', X1 + 356, 1440, CW - 712, 32, color=CORAL, bold=True, align=TA_CENTER)

    para('A worst-case marginal', X1, 1558, CW, 43, bold=True, color=NAVY, max_h=56)
    para('v<sub>α</sub>(S) = 1[U(M<sub>S</sub>) ≥ α U<sub>full</sub>]',
         X1 + 15, 1640, CW - 30, 48, leading=63, color=NAVY, max_h=66)
    para('Δ<sub>i</sub>(S) = v<sub>α</sub>(S ∪ {i}) - v<sub>α</sub>(S)',
         X1 + 15, 1728, CW - 30, 45, leading=59, color=NAVY, max_h=63)
    para('MinPI<sub>i</sub> = min Δ<sub>i</sub>(S)',
         X1 + 15, 1817, CW - 30, 55, leading=69, color=TEAL, bold=True, max_h=73)
    para('over all S ⊆ N ∖ {i}; N is the analyzed feature set',
         X1 + 15, 1893, CW - 30, 31, color=MUTED, max_h=41)

    rect(X1, 1964, CW, 62, LIGHT, radius=10)
    for j, (label, color) in enumerate([('-1 harmful', CORAL), ('0 neutral*', NAVY), ('+1 decisive', TEAL)]):
        para(label, X1 + j * CW / 3, 1976, CW / 3, 34, bold=True,
             color=color, align=TA_CENTER, max_h=44)
    para('*Exact values. A sampled 0 or +1 is not a certificate; sampling can miss witnesses.',
         X1, 2054, CW, 32, leading=40, color=MUTED, max_h=85)

    qx, qy, qw = X1 + 12, 2180, 90 * mm
    qh = qw * 1320 / 1100
    c.drawImage(str(ROOT / 'output/QRcode.png'), qx, H - qy - qh, qw, qh,
                preserveAspectRatio=True, mask='auto')
    c.linkURL('https://hlnks.co/63671101', (qx, H - qy - qh, qx + qw, H - qy), relative=0)
    para('Scan to read<br/>the full paper', X1 + 310, 2203, CW - 310, 45,
         leading=54, bold=True, color=NAVY, max_h=112)
    para('Theory, experiments<br/>and complete results', X1 + 310, 2336, CW - 310,
         34, leading=43, color=MUTED, max_h=90)
    para('hlnks.co/63671101', X1 + 310, 2440, CW - 310, 30, color=TEAL, max_h=40)
    manifest['qr'] = {'x_pt': qx, 'y_top_pt': qy, 'width_mm': 90,
                       'height_mm': 108, 'effective_dpi': 1100 / (90 / 25.4),
                       'payload': 'https://hlnks.co/63671101'}


def method():
    section(X2, '02', 'THE METHOD')
    rect(X2, 939, CW, 281, PALE, radius=18)
    para('1  Discover candidates', X2 + 30, 961, CW - 60, 44,
         bold=True, color=NAVY, max_h=56)
    para('Fit coalition models on training data.<br/>Evaluate threshold crossings on validation data.',
         X2 + 30, 1034, CW - 60, 38, leading=47, max_h=146)
    para('Exact enumeration or sampled search', X2 + 30, 1164, CW - 60,
         32, color=TEAL, bold=True, max_h=43)

    arrow_down(X2 + CW / 2, 1235, 1271)
    rect(X2, 1288, CW, 391, LIGHT, radius=18)
    para('2  Gate proposed removals', X2 + 30, 1310, CW - 60, 44,
         bold=True, color=NAVY, max_h=56)
    para('Let H be the candidate harmful set.<br/>Accept its removal only if:',
         X2 + 30, 1383, CW - 60, 38, leading=47, max_h=98)
    para('U<sub>val</sub>(N ∖ H) ≥ U<sub>val</sub>(N) - ε',
         X2 + 30, 1505, CW - 60, 44, leading=58, color=TEAL, bold=True, max_h=61)
    para('ε = 0.005 (classification); 0.01 (regression)',
         X2 + 30, 1577, CW - 60, 33, color=INK, max_h=46)
    para('PASS: remove H  |  FAIL: remove nothing', X2 + 30, 1630, CW - 60,
         32, color=TEAL, bold=True, max_h=41)

    arrow_down(X2 + CW / 2, 1694, 1730)
    rect(X2, 1747, CW, 244, PALE, radius=18)
    para('3  Evaluate &amp; optimize', X2 + 30, 1769, CW - 60, 44,
         bold=True, color=NAVY, max_h=56)
    para('Use held-out test data for final reporting.<br/>Accepted subsets may prefilter NSGA-II.',
         X2 + 30, 1844, CW - 60, 38, leading=47, max_h=143)

    para('EXPERIMENTAL PROTOCOL', X2, 2041, CW, 35, bold=True, color=TEAL, max_h=46)
    para('6 datasets × 5 seeds', X2, 2106, CW, 49, bold=True, color=NAVY, max_h=63)
    para('64% train  /  16% validation  /  20% test', X2, 2181, CW, 36,
         color=INK, max_h=48)
    para('XGBoost; α = 0.95; at most 100 analyzed features after training-only preprocessing.',
         X2, 2253, CW, 37, leading=46, max_h=141)
    para('ROC AUC, OVR AUC or R².<br/>Controls delete the same accepted count.',
         X2, 2400, CW, 32, leading=40, color=MUTED, max_h=85)


def results():
    section(X3, '03', 'THE EVIDENCE')
    para('Compression is<br/>dataset-dependent.', X3, 939, CW, 47,
         leading=55, bold=True, color=NAVY, max_h=112)
    para('Five-seed mean ± SD. Bars: analyzed features removed.<br/>ΔU: held-out utility minus the full model.',
         X3, 1075, CW, 32, leading=40, color=MUTED, max_h=85)
    para('COMPRESSION (%)', X3, 1191, 650, 30, bold=True, color=TEAL, max_h=39)
    para('HELD-OUT ΔU', X3 + 730, 1191, CW - 730, 30,
         bold=True, color=TEAL, align=TA_RIGHT, max_h=39)
    line(X3, 1244, X3 + CW, 1244, RULE, 2)

    chartw = 658
    # A shared linear scale; SD whiskers extend above the mean and are truncated at zero below.
    for i, row in enumerate(ROWS):
        y = 1263 + i * 102
        para(row['dataset'], X3, y, 415, 33, bold=True, color=NAVY, max_h=43)
        para(f"{row['compression']:.2f} ± {row['sd']:.2f}", X3 + 414, y, 280,
             31, color=INK, align=TA_RIGHT, max_h=40)
        value = f"{row['du']:+.5f}" if row['du'] else '0.00000'
        para(value, X3 + 738, y - 1, CW - 738, 36, bold=True,
             color=TEAL if row['du'] > 0 else INK, align=TA_RIGHT, max_h=46)
        para(f"± {row['du_sd']:.5f}", X3 + 738, y + 43, CW - 738,
             30, color=MUTED, align=TA_RIGHT, max_h=39)
        yy = y + 62
        rect(X3, yy, chartw, 14, PALE, radius=5)
        meanpx = chartw * row['compression'] / 80
        if meanpx:
            rect(X3, yy, max(meanpx, 2), 14, TEAL, radius=min(5, meanpx / 2))
        lo = max(0, row['compression'] - row['sd']) / 80 * chartw
        hi = (row['compression'] + row['sd']) / 80 * chartw
        if hi:
            line(X3 + lo, yy + 7, X3 + hi, yy + 7, NAVY, 2.2)
            line(X3 + lo, yy - 3, X3 + lo, yy + 17, NAVY, 2.2)
            line(X3 + hi, yy - 3, X3 + hi, yy + 17, NAVY, 2.2)
            fill(TEAL)
            c.circle(X3 + meanpx, H - yy - 7, 7, fill=1, stroke=0)

    # Shared scale kept outside the last row.
    for tick in [0, 40, 80]:
        xx = X3 + chartw * tick / 80
        para(str(tick), xx - (25 if tick else 0), 1879, 75, 30, color=MUTED)
    para('14/30 units had nonzero removal; 3/17 nonempty candidate sets were vetoed by the gate.',
         X3, 1936, CW, 32, leading=40, color=MUTED, max_h=86)

    rect(X3, 2054, CW, 247, LIGHT, radius=18)
    para('NSGA-II: equal search budgets', X3 + 28, 2074, CW - 56, 38,
         bold=True, color=NAVY, max_h=50)
    para('+0.01584', X3 + 28, 2140, 470, 62, leading=70, color=TEAL, bold=True, max_h=74)
    para('mean held-out ΔHV<br/>800 evaluations per variant',
         X3 + 492, 2141, CW - 520, 31, leading=39, max_h=82)
    para('Holm p = 0.4375; not significant. Madelon drives most of the gain.',
         X3 + 28, 2236, CW - 56, 31, leading=39, color=INK, max_h=82)

    para('WHAT THIS DOES NOT ESTABLISH', X3, 2337, CW, 32, bold=True, color=CORAL, max_h=43)
    para('No overall superiority or speedup. The validation gate does not guarantee held-out non-inferiority.',
         X3, 2397, CW, 35, leading=44, max_h=134)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    PDF.parent.mkdir(parents=True, exist_ok=True)
    for rel in ['output/pdf/MinPI_arXiv_final.pdf', 'output/QRcode.png', 'MinPI_arXiv_source(1).zip']:
        manifest['source_hashes'][rel] = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
    with Image.open(ROOT / 'output/QRcode.png') as qr:
        assert qr.size == (1100, 1320)
    c = canvas.Canvas(str(PDF), pagesize=(W, H), pageCompression=1,
                      initialFontName='Arial', initialFontSize=38, initialLeading=47)
    c.setTitle('ASCOMP 2026 Poster - Min-Marginal Power Indices for Multi-Objective Feature Selection')
    c.setAuthor('Wenjie Shi; Chi Zhao')
    c.setSubject('120 x 90 cm scientific poster; vector graphics; original paper QR')
    rect(0, 0, W, H, WHITE)
    header()
    problem()
    method()
    results()
    c.showPage()
    c.save()
    (OUT / 'layout_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Created: {PDF.name}')
    print(f'Page: {W:.6f} x {H:.6f} pt (1200 x 900 mm); text elements: {len(manifest["text"])}')
