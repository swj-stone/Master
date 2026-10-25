"""Generate the algorithm-complete three-column 1200 x 900 mm ASCOMP poster.

All geometry is vector; only the original, unchanged paper QR is a bitmap.
Prior revisions are retained under conference-paper/archive/.
"""
import hashlib
import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
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
OUT = ROOT / 'conference-paper/poster'
PDF = OUT / 'ASCOMP_MinPPI_Poster_Wenjie_SHI.pdf'
TEX_DIR = OUT / 'latex'
BUILD_DIR = ROOT / 'tmp/pdfs/ascomp-poster/latex-build'
BACKGROUND = TEX_DIR / 'poster-background.pdf'
tex_nodes = []
W, H = 1200 * mm, 900 * mm
M, GAP = 100, 104
CW = (W - 2 * M - GAP) / 2
X1, X2 = M, M + CW + GAP
BODY_GAP = 64
BW = (W - 2 * M - 2 * BODY_GAP) / 3
BX = [M + i * (BW + BODY_GAP) for i in range(3)]
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
    'layout': {'columns': 3, 'revision': 'latex-final-audit-v7', 'column_width_pt': BW},
    'typesetting': {'engine': 'XeLaTeX', 'math_font': 'Latin Modern Math', 'text_font': 'Arial'},
    'text': [], 'fit_checks': [], 'data': ROWS,
    'glyph_audit': {'checked_characters': 0, 'missing': []},
    'provenance': {
        'paper': 'conference-paper/paper/icomp2026_conference.pdf',
        'compression_data': 'Table 2: five-seed means and sample SD',
        'aggregate_removal': 'Section 4.1: all 30 dataset-seed units',
        'nsga_ii': 'Section 4.3: held-out hypervolume, not Figure 3 validation curves',
        'random_comparison': 'Section 4.1: Madelon and Abalone mean held-out delta utility, matched deletion counts',
        'algorithm': 'Section 2, Section 3, Appendix A: game, A/B/hybrid, all-or-none gate, optional NSGA-II',
        'estimator_F1': 'Section 4.4: Abalone five-seed mean harmful-set F1 at K=2,5,10,25',
        'tools': 'Sections 2-3 and Appendix A: MI, maximum-weight absolute-correlation tree, XGBoost CUDA, TreeSHAP, NSGA-II',
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


def tex_escape(value):
    tokens = re.split(r'(ΔHV|ΔU|R²|F1|[αΔε≥≤→←×±∪⊆∖]|\b[NnSiKkHm]\b)', value)
    maths = {'ΔHV': r'\Delta\mathrm{HV}', 'ΔU': r'\Delta U', 'R²': r'R^2',
             'F1': r'F_1', 'α': r'\alpha', 'Δ': r'\Delta', 'ε': r'\varepsilon',
             '≥': r'\geq', '≤': r'\leq', '→': r'\to', '←': r'\gets',
             '×': r'\times', '±': r'\pm', '∪': r'\cup', '⊆': r'\subseteq', '∖': r'\setminus'}
    escape = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
              '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
              '^': r'\textasciicircum{}', '<': r'\textless{}', '>': r'\textgreater{}'}
    return ''.join(r'\(' + maths.get(t, t) + r'\)' if t in maths or re.fullmatch(r'[NnSiKkHm]', t)
                   else ''.join(escape.get(ch, ch) for ch in t) for t in tokens)


class HtmlToTex(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == 'br':
            self.parts.append(r'\\')
        else:
            self.parts.append({'b': r'\textbf{', 'i': r'\textit{',
                               'sub': r'\textsubscript{', 'super': r'\textsuperscript{',
                               'font': '{'}.get(tag, '{'))

    def handle_endtag(self, tag):
        if tag != 'br':
            self.parts.append('}')

    def handle_data(self, data):
        self.parts.append(tex_escape(data))


def para(text, x, y, w, size=40, leading=None, color=INK, bold=False,
         align=TA_LEFT, max_h=None, label=None, latex=None):
    original_text = text
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
        # Actual dimensions are checked from XeLaTeX box measurements below.
    manifest['text'].append({'text': text, 'x': x, 'y': y, 'width': used_w,
                             'height': used_h, 'font_size': size,
                             'alignment': {TA_LEFT: 'left', TA_CENTER: 'center', TA_RIGHT: 'right'}[align],
                             'max_height': max_h, 'latex': latex})
    if latex is None:
        parser = HtmlToTex()
        parser.feed(original_text)
        latex = ''.join(parser.parts)
    idx = len(manifest['text']) - 1
    alignment = {TA_LEFT: r'\raggedright', TA_CENTER: r'\centering', TA_RIGHT: r'\raggedleft'}[align]
    weight = r'\bfseries' if bold else r'\mdseries'
    tex_nodes.append(
        rf'\sbox{{\posterbox}}{{\parbox[t]{{{w:.5f}bp}}{{'
        rf'\normalfont\fontsize{{{size}bp}}{{{leading or size * 1.23}bp}}\selectfont{weight}'
        rf'\color[HTML]{{{color[1:]}}}{alignment}\hyphenpenalty=10000\exhyphenpenalty=10000 '
        + latex + r'\par}}' + '\n'
        + rf'\typeout{{POSTERBOX:{idx}:\the\dimexpr\ht\posterbox+\dp\posterbox\relax}}' + '\n'
        + rf'\begin{{tikzpicture}}[remember picture,overlay]\node[anchor=north west,inner sep=0pt,outer sep=0pt] at '
        rf'([xshift={x:.5f}bp,yshift=-{y:.5f}bp]current page.north west){{\usebox{{\posterbox}}}};\end{{tikzpicture}}%')
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
    c.drawImage(str(OUT / 'assets/QRcode.png'), qx, H - qy - qh, qw, qh,
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
    para(text, x, 775, BW, 46, bold=True, color=NAVY, max_h=59)
    line(x, 845, x + BW, 845, RULE, 2)


def box_title(x, y, title, height, color=PALE):
    rect(x, y, BW, height, color, radius=16)
    para(title, x + 24, y + 15, BW - 48, 37, bold=True, color=NAVY, max_h=48)


def foundations():
    x = BX[0]
    section(x, '01  PRINCIPLE &amp; TOOLKIT')
    para('A coalition witness reveals harmful additions:<br/>acceptable utility becomes unacceptable.',
         x, 875, BW, 39, leading=49, bold=True, color=NAVY, max_h=100)
    rect(x, 1000, BW, 470, LIGHT, radius=18)
    para('N: analyzed features; n = |N|; Uval(N) > 0; 0 < α ≤ 1', x + 24, 1020, BW - 48,
         32, max_h=42, latex=r'\(N\): analyzed features; \(n=|N|,\quad\Uval(N)>0,\quad 0<\alpha\leq1\).')
    para('v<sub>α</sub>(S) = 1[U<sub>val</sub>(S) ≥ α U<sub>val</sub>(N)]',
         x + 24, 1074, BW - 48, 42, leading=54, color=NAVY, max_h=64,
         latex=r'\(\displaystyle v_\alpha(S)=\mathbf{1}\!\left[\Uval(S)\geq\alpha\Uval(N)\right]\)')
    para('Δ<sub>i</sub>(S) = v<sub>α</sub>(S ∪ {i}) - v<sub>α</sub>(S)',
         x + 24, 1144, BW - 48, 41, leading=53, color=NAVY, max_h=60,
         latex=r'\(\displaystyle\Delta_i(S)=v_\alpha(S\cup\{i\})-v_\alpha(S)\)')
    para('MinPI<sub>i</sub> = MinSS<sub>i</sub> = MinBz<sub>i</sub> = min Δ<sub>i</sub>(S)',
         x + 24, 1214, BW - 48, 38, leading=50, color=TEAL, max_h=86,
         latex=r'\(\displaystyle\MinPI_i=\MinSS_i=\MinBz_i=\min_{S\subseteq N\setminus\{i\}}\Delta_i(S)\)')
    para('Exact MinPI labels:', x + 24, 1308, BW - 48, 31, max_h=38)
    para('<b>-1 harmful:</b> some coalition flips 1 → 0.<br/>'
         '<b>0 neutral:</b> all marginals ≥0; at least one is 0.<br/>'
         '<b>+1 decisive:</b> every coalition flips 0 → 1.',
         x + 24, 1352, BW - 48, 32, leading=35, max_h=111,
         latex=r'\textbf{\(-1\) harmful:} some coalition flips \(1\to0\).\\'
               r'\textbf{\(0\) neutral:} all marginals \(\geq0\); at least one is \(0\).\\'
               r'\textbf{\(+1\) decisive:} every coalition flips \(0\to1\).')
    para('MinSS / MinBz: minimum Shapley-Shubik / Banzhaf.<br/>'
         'Both minima cover all coalitions; at most 2<super>n</super> utilities.',
         x, 1490, BW, 32, leading=40, color=MUTED, max_h=86,
         latex=r'MinSS / MinBz: minimum Shapley-Shubik / Banzhaf.\\'
               r'Both minima cover all coalitions; exact uses at most \(2^n\) utilities.')

    para('CORE TOOLS', x, 1592, BW, 40, bold=True, color=TEAL, max_h=51)
    para('<b>MI (mutual information)</b> ranks feature-target dependence.<br/>'
         'A <b>maximum-weight absolute-correlation tree</b><br/>'
         'represents pairwise association and guides Algorithm B.',
         x, 1658, BW, 34, leading=43, max_h=132)
    para('<b>XGBoost:</b> 100 rounds, depth 4, learning rate 0.1;<br/>'
         'CUDA trees. Fit coalition models on training data.',
         x, 1812, BW, 34, leading=43, max_h=89)
    para('<b>Training-only preprocessing:</b> variance filtering →<br/>'
         'MI selection (≤100 features) → standardization.',
         x, 1923, BW, 34, leading=43, max_h=89)
    para('<b>Utility:</b> ROC AUC / one-vs-rest AUC / R².<br/>'
         'Uval(S): validation utility of train-fitted subset S.<br/>'
         'Empty coalition: training class priors / response mean.',
         x, 2027, BW, 33, leading=39, max_h=121,
         latex=r'\textbf{Utility:} ROC AUC / one-vs-rest AUC / \(R^2\).\\'
               r'\(\Uval(S)\): validation utility of train-fitted subset \(S\).\\'
               r'Empty coalition: training class priors / response mean.')

    para('MATCHED-COUNT CONTROLS', x, 2180, BW, 38, bold=True, color=TEAL, max_h=49)
    para('All controls delete the accepted MinPI count k.<br/>'
         '<b>SHAP-bottom:</b> lowest mean |TreeSHAP| on train.<br/>'
         '<b>MinShap / Banzhaf-bottom:</b> lowest minimum / mean<br/>'
         'continuous utility marginal from shared coalitions.<br/>'
         '<b>Random:</b> 30 matched-count trials per seed.',
         x, 2247, BW, 32, leading=39, max_h=200)
    para('Quota sweep (seed 42): α in {0.85, 0.90, 0.95, 0.99}.',
         x, 2465, BW, 31, color=MUTED, max_h=40,
         latex=r'Quota sweep (seed 42): \(\alpha\in\{0.85,0.90,0.95,0.99\}\).')


def algorithm():
    x = BX[1]
    section(x, '02  ALGORITHM FLOW')
    para('<b>Train 64% → validation 16% → test 20%</b><br/>'
         'Fit on train; compute the threshold game on validation.',
         x, 875, BW, 33, leading=43, max_h=89)

    box_title(x, 1000, 'Algorithm A: size-uniform sampling', 180)
    para('Draw s uniformly from {0, ..., n-1}; then draw<br/>'
         'an s-feature subset uniformly from N ∖ {i}.',
         x + 24, 1065, BW - 48, 33, leading=42, max_h=91,
         latex=r'Draw \(s\) uniformly from \(\{0,\ldots,n-1\}\); then draw\\'
               r'an \(s\)-feature subset uniformly from \(N\setminus\{i\}\).')

    box_title(x, 1200, 'Algorithm B: MI-correlation guidance', 300)
    para('B<sub>i</sub>: top ≤10 tree neighbors by absolute correlation;<br/>'
         'pad from the top-MI half of N ∖ {i} toward min(20, floor(n/3)).<br/>'
         'Each draw: reset to B<sub>i</sub>; drop each independently (p = 0.3).<br/>'
         'Draw a uniformly from 0, ..., min(5, n-1);<br/>'
         'add up to a absent features from N ∖ {i} in random order.',
         x + 24, 1265, BW - 48, 33, leading=42, max_h=225,
         latex=r'\(B_i\): up to 10 tree neighbors ranked by absolute correlation;\\'
               r'pad from the top-MI half of \(N\setminus\{i\}\) toward \(\min\{20,\lfloor n/3\rfloor\}\).\\'
               r'Each draw: reset \(S\gets B_i\); drop each independently (\(p=0.3\)).\\'
               r'Draw \(a\) uniformly from \(\{0,\ldots,\min(5,n-1)\}\);\\'
               r'add \(\min\{a,n-1-|S|\}\) absent features in random order.')
    para('Both: m ← min(m, Δ<sub>i</sub>(S)), initialized at +1.<br/>'
         '<b>Stop at -1</b> with a witness; else return m after K draws.<br/>'
         '<b>K in {2, 5, 10, 25}</b> per feature per estimator.<br/>'
         'ISOLET uses {2,5,10}; removal uses the largest K.',
         x, 1525, BW, 33, leading=42, max_h=176,
         latex=r'Initialize \(m\gets+1\); update \(m\gets\min\{m,\Delta_i(S)\}\).\\'
               r'\textbf{Stop at \(-1\)} with a witness; else return \(m\) after \(K\) draws.\\'
               r'\(K\in\{2,5,10,25\}\) per feature per estimator.\\'
               r'ISOLET: \(K\in\{2,5,10\}\); removal uses the largest \(K\).')

    rect(x, 1720, BW, 140, LIGHT, radius=16)
    para('HYBRID DISPATCH', x + 24, 1732, BW - 48, 32,
         bold=True, color=TEAL, max_h=41)
    para('n ≤ 20: exact  |  20 &lt; n ≤ 75: B*  |  n &gt; 75: min(A, B)',
         x + 24, 1780, BW - 48, 31, max_h=42,
         latex=r'\(n\leq20\): exact\quad|\quad\(20<n\leq75\): B*\quad|\quad\(n>75\): \(\min(A,B)\)')
    para('*Use A if no correlation tree is available; min is featurewise.',
         x + 24, 1822, BW - 48, 30, color=MUTED, max_h=38)

    box_title(x, 1880, 'Validate joint removal', 282)
    para('H = {i: estimated MinPI<sub>i</sub> = -1}', x + 24, 1940, BW - 48,
         34, max_h=52,
         latex=r'\(\displaystyle H=\bigl\{i\in N:\widehat{\MinPI}_i=-1\bigr\}\)')
    para('U<sub>val</sub>(N ∖ H) ≥ U<sub>val</sub>(N) - ε',
         x + 24, 1994, BW - 48, 42, leading=54, color=TEAL, max_h=58,
         latex=r'\(\displaystyle\Uval(N\setminus H)\geq\Uval(N)-\varepsilon\)')
    para('ε = 0.005 (classification); 0.01 (regression).<br/>'
         '<b>PASS:</b> remove all H.  <b>FAIL:</b> keep all N.<br/>'
         'Test retained features; controls delete the removed count.',
         x + 24, 2055, BW - 48, 31, leading=34, max_h=107,
         latex=r'\(\varepsilon=0.005\) (classification); \(0.01\) (regression).\\'
               r'Remove \(R=H\) on \textbf{PASS}; \(R=\varnothing\) on \textbf{FAIL}.\\'
               r'Test \(S_{\mathrm{ret}}=N\setminus R\) once; controls delete \(k=|R|\).')

    box_title(x, 2175, 'Optional MinPI-prefiltered NSGA-II', 330, LIGHT)
    para('Non-dominated sorting genetic algorithm II; minimize:', x + 24, 2232, BW - 48,
         30, max_h=41)
    para('(|S|/n, 1 - clip(U<sub>val</sub>(S), 0, 1))',
         x + 24, 2272, BW - 48, 38, leading=51, color=TEAL, max_h=94,
         latex=r'\(\displaystyle\left(\frac{|S|}{n},\;1-\operatorname{clip}\!\left(\Uval(S),0,1\right)\right)\)')
    para('n is shared: analyzed dimension before prefiltering.<br/>'
         'population 40; 800 evaluations per variant.<br/>'
         'HV reference (1.1, 1.1); select fronts on validation.<br/>'
         '<b>Evaluate</b> the selected fronts on held-out test data.',
         x + 24, 2380, BW - 48, 31, leading=32, max_h=125)


def results():
    x = BX[2]
    section(x, '03  EXPERIMENTAL EVIDENCE')
    para('6 datasets × 5 seeds', x, 875, BW, 44,
         bold=True, color=NAVY, max_h=56)
    para('Seeds: 42, 123, 456, 789, 1024  |  default α = 0.95',
         x, 936, BW, 31, color=MUTED, max_h=40)
    rect(x, 996, BW, 148, NAVY, radius=18)
    para('12.53%', x + 16, 1009, BW / 2 - 24, 60, leading=70,
         bold=True, color='#81D9D1', align=TA_CENTER, max_h=73)
    para('mean feature removal', x + 16, 1091, BW / 2 - 24, 30,
         color=WHITE, align=TA_CENTER, max_h=38)
    line(x + BW / 2, 1018, x + BW / 2, 1121, '#4A6578', 2)
    para('-0.000313', x + BW / 2 + 8, 1009, BW / 2 - 24, 60, leading=70,
         bold=True, color=WHITE, align=TA_CENTER, max_h=73)
    para('mean held-out ΔU', x + BW / 2 + 8, 1091, BW / 2 - 24, 30,
         color=WHITE, align=TA_CENTER, max_h=38)
    para('Five-seed mean ± SD. ΔU = retained - full utility.<br/>'
         'Analyzed-feature removal; bars 0-80%, whiskers SD.',
         x, 1170, BW, 31, leading=39, color=MUTED, max_h=86,
         latex=r'\(\Delta U=\Utest(S_{\mathrm{ret}})-\Utest(N)\); five-seed mean \(\pm\) SD.\\'
               r'Analyzed-feature removal; bars: 0-80\%, whiskers: SD.')
    para('DATASET', x, 1270, 310, 30, bold=True, color=TEAL, max_h=38)
    para('REMOVED (%)', x + 330, 1270, 340, 30, bold=True,
         color=TEAL, align=TA_CENTER, max_h=38)
    para('HELD-OUT ΔU', x + 710, 1270, BW - 710, 30, bold=True,
         color=TEAL, align=TA_RIGHT, max_h=38)
    line(x, 1314, x + BW, 1314, RULE)
    chartx, chartw = x + 330, 340
    for i, row in enumerate(ROWS):
        y = 1330 + i * 87
        para(row['dataset'], x, y + 4, 310, 31, bold=True, color=NAVY, max_h=40)
        para(f"{row['compression']:.2f} ± {row['sd']:.2f}", chartx, y, chartw,
             32, align=TA_CENTER, max_h=41)
        value = f"{row['du']:+.5f}" if row['du'] else '0.00000'
        para(value, x + 710, y - 2, BW - 710, 34, bold=True,
             color=TEAL if row['du'] > 0 else INK, align=TA_RIGHT, max_h=44)
        para(f"± {row['du_sd']:.5f}", x + 710, y + 40, BW - 710,
             30, color=MUTED, align=TA_RIGHT, max_h=38)
        yy = y + 55
        rect(chartx, yy, chartw, 14, PALE, radius=5)
        meanpx = chartw * row['compression'] / 80
        if meanpx:
            rect(chartx, yy, max(meanpx, 2), 14, TEAL, radius=min(5, meanpx / 2))
        lo = max(0, row['compression'] - row['sd']) / 80 * chartw
        hi = (row['compression'] + row['sd']) / 80 * chartw
        if hi:
            line(chartx + lo, yy + 7, chartx + hi, yy + 7, NAVY, 2)
            line(chartx + lo, yy - 3, chartx + lo, yy + 17, NAVY, 2)
            line(chartx + hi, yy - 3, chartx + hi, yy + 17, NAVY, 2)
            fill(TEAL)
            c.circle(chartx + meanpx, H - yy - 7, 6, fill=1, stroke=0)

    rect(x, 1870, BW, 142, LIGHT, radius=16)
    para('MATCHED-COUNT REMOVAL: MinPI vs random', x + 20, 1883, BW - 40,
         30, bold=True, color=TEAL, max_h=38)
    para('Madelon: ΔU +0.0047 vs -0.0732<br/>'
         'Abalone: ΔU -0.0056 vs -0.0370',
         x + 20, 1928, BW - 40, 33, leading=39, bold=True, color=NAVY, max_h=81)

    para('NSGA-II: held-out hypervolume', x, 2040, BW, 37,
         bold=True, color=NAVY, max_h=48)
    para('ΔHV = MinPI-prefiltered - full-space search.<br/>'
         '<b>Mean +0.01584</b> (Holm p = 0.4375); Madelon +0.09515.<br/>'
         'Positive dataset-averaged change in all five seeds.',
         x, 2100, BW, 31, leading=39, max_h=120,
         latex=r'\(\Delta\mathrm{HV}=\mathrm{HV}_{\mathrm{prefiltered}}-\mathrm{HV}_{\mathrm{full}}\).\\'
               r'\textbf{Mean +0.01584} (Holm \(p=0.4375\)); Madelon +0.09515.\\'
               r'Positive dataset-averaged change in all five seeds.')

    para('ESTIMATOR ACCURACY', x, 2250, BW, 37, bold=True, color=TEAL, max_h=48)
    para('Abalone (8 features): harmful-set F1 vs exact MinPI.<br/>'
         'Hybrid uses exact enumeration: F1 = 1 at every K.',
         x, 2303, BW, 30, leading=36, color=MUTED, max_h=80,
         latex=r'Abalone: mean harmful-set \(F_1\) vs exact; 8 features, 5 seeds.\\'
               r'Hybrid uses exact enumeration: \(F_1=1\) at every \(K\).')
    # Compact numeric panel keeps all four budgets and both estimator outcomes.
    columns = [(x, 330), (x + 350, 150), (x + 520, 150), (x + 690, 150), (x + 860, 164)]
    for j, value in enumerate(['K', '2', '5', '10', '25']):
        xx, width = columns[j]
        para(value, xx, 2390, width, 30, bold=True, color=TEAL,
             align=TA_LEFT if j == 0 else TA_CENTER, max_h=38)
    for yy, row in [(2428, ['A', '0.100', '0.150', '0.230', '0.443']),
                    (2466, ['B', '0.000', '0.080', '0.133', '0.313'])]:
        for j, value in enumerate(row):
            xx, width = columns[j]
            para(value, xx, yy, width, 30, bold=j == 0, color=INK,
                 align=TA_LEFT if j == 0 else TA_CENTER, max_h=38)
    manifest['estimator_f1'] = {'dataset': 'Abalone', 'seeds': 5,
        'budgets': [2, 5, 10, 25], 'A': [.100, .150, .230, .443],
        'B': [.000, .080, .133, .313], 'hybrid': [1., 1., 1., 1.]}

if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    for relative in ['conference-paper/paper/icomp2026_conference.pdf',
                     'conference-paper/poster/assets/QRcode.png',
                     '../MinPI_arXiv_source(1).zip', 'conference-paper/poster/assets/tele_code.png']:
        manifest['source_hashes'][relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    with Image.open(OUT / 'assets/QRcode.png') as qr:
        assert qr.size == (1100, 1320)
    c = canvas.Canvas(str(BACKGROUND), pagesize=(W, H), pageCompression=1,
                      initialFontName='Arial', initialFontSize=40, initialLeading=49)
    c.setTitle('ASCOMP 2026 Poster - Min-Marginal Power Indices for Multi-Objective Feature Selection')
    c.setAuthor('Wenjie Shi; Chi Zhao')
    c.setSubject('120 x 90 cm; three columns; full core algorithm and toolkit; two upper-right QR codes')
    rect(0, 0, W, H, WHITE)
    header()
    for boundary in [BX[1] - BODY_GAP / 2, BX[2] - BODY_GAP / 2]:
        line(boundary, 780, boundary, 2505, RULE, 2)
    foundations()
    algorithm()
    results()
    c.showPage()
    c.save()
    (TEX_DIR / 'poster-content.tex').write_text(
        '% Generated by build_poster.py. Edit content/layout in the generator.\n' + '\n\n'.join(tex_nodes),
        encoding='utf-8')
    for _ in range(2):
        build = subprocess.run(['xelatex', '-no-shell-escape', '-interaction=nonstopmode',
                                '-halt-on-error', '-output-directory', str(BUILD_DIR), 'main.tex'], cwd=TEX_DIR,
                               capture_output=True, text=True, encoding='utf-8', errors='replace')
        if build.returncode:
            raise RuntimeError(build.stdout[-7000:] + build.stderr[-2000:])
    log = (BUILD_DIR / 'main.log').read_text(encoding='utf-8', errors='replace')
    if re.search(r'Overfull \\[hv]box|Missing character:', log):
        raise RuntimeError(f'LaTeX overflow or missing glyph: inspect {BUILD_DIR / "main.log"}')
    actual_boxes = {int(i): float(height) * 72 / 72.27
                    for i, height in re.findall(r'POSTERBOX:(\d+):([\d.]+)pt', log)}
    assert len(actual_boxes) == len(manifest['text']), 'Missing LaTeX box measurements'
    manifest['fit_checks'] = []
    for idx, item in enumerate(manifest['text']):
        item['height'] = actual_boxes[idx]
        if item['max_height'] is not None:
            manifest['fit_checks'].append({'label': item['text'][:70],
                                          'used_height': item['height'],
                                          'available_height': item['max_height']})
            if item['height'] > item['max_height'] + .1:
                raise ValueError(f'LaTeX text overflows node {idx}: {item["height"]:.2f} > '
                                 f'{item["max_height"]}: {item["text"][:100]}')
    shutil.copy2(BUILD_DIR / 'main.pdf', PDF)
    manifest['typesetting']['compile_success'] = True
    manifest['typesetting']['overfull_boxes'] = 0
    manifest['typesetting']['missing_characters'] = 0
    (OUT / 'layout_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Created: {PDF.name}; 1200 x 900 mm; three columns; {len(manifest["text"])} text elements')
