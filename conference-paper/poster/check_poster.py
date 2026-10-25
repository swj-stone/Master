"""Read-only preflight for the ASCOMP poster and its protected inputs."""
import hashlib
import json
import re
import sys
import unicodedata
import unittest
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'conference-paper/poster'
PDF = OUT / 'ASCOMP_MinPPI_Poster_Wenjie_SHI.pdf'
MANIFEST = OUT / 'layout_manifest.json'
TEX_DIR = OUT / 'latex'
BUILD_DIR = ROOT / 'tmp/pdfs/ascomp-poster/latex-build'
HASHES = {
    'conference-paper/paper/icomp2026_conference.pdf': '2c7cc05471252f1e464230d3a01cd5d1e8d4fb8852fb46f37162a6ca5fc26424',
    'conference-paper/poster/assets/QRcode.png': '3b007243b94c94bcc09a92cc86a5c0900ace9d4a4c103e565a84a07e2245435d',
    '../MinPI_arXiv_source(1).zip': '5e09469bd8731d4158b16921b737e5c99b56a46e0e95163c2ea93e03d09e4cf6',
    'conference-paper/poster/assets/tele_code.png': 'f9ac461d4908cecd45fc29d684eb90c68798e8ab771661c5b0a60bcf70538d63',
}


def normalized_text(value, compact=False):
    value = unicodedata.normalize('NFKC', value).replace('\N{MINUS SIGN}', '-')
    return ''.join(value.split()) if compact else ' '.join(value.split())


def poster_text(compact=False):
    return normalized_text(PdfReader(PDF).pages[0].extract_text(), compact=compact)


def resource_fonts(resources, seen=None):
    """Include fonts inside nested PDF forms, not only page-level resources."""
    seen = set() if seen is None else seen
    resources = resources.get_object()
    if id(resources) in seen:
        return
    seen.add(id(resources))
    for reference in resources.get('/Font', {}).get_object().values() if '/Font' in resources else ():
        yield reference.get_object()
    for reference in resources.get('/XObject', {}).get_object().values() if '/XObject' in resources else ():
        child = reference.get_object().get('/Resources')
        if child is not None:
            yield from resource_fonts(child, seen)


def font_descriptors(font):
    if font.get('/Subtype') == '/Type0':
        for reference in font.get('/DescendantFonts', []):
            yield from font_descriptors(reference.get_object())
    else:
        descriptor = font.get('/FontDescriptor')
        yield descriptor.get_object() if descriptor is not None else None


class PosterPreflight(unittest.TestCase):
    def test_protected_sources_unchanged(self):
        for relative, expected in HASHES.items():
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), expected, relative)

    def test_deliverable_exists(self):
        self.assertTrue(PDF.is_file(), 'Poster has not been generated yet')

    def test_page_size_orientation_and_count(self):
        reader = PdfReader(PDF)
        self.assertEqual(len(reader.pages), 1)
        box = reader.pages[0].mediabox
        self.assertAlmostEqual(float(box.width) * 25.4 / 72, 1200, places=2)
        self.assertAlmostEqual(float(box.height) * 25.4 / 72, 900, places=2)

    def test_required_copy_and_statistics(self):
        text = poster_text(compact=True)
        for expected in ['ASCOMP 2026 POSTER', 'Min-Marginal Power Indices',
                         'Multi-Objective Feature Selection', 'Wenjie Shi', 'Chi Zhao',
                         'Saint Petersburg State University', 'Universitetskaya',
                         '199034', 'Russia', 'Contact author', 'Telegram', '12.53%', '-0.000313',
                         'Madelon', 'Abalone', 'Credit-Card-Fraud', 'Spambase',
                         'Gisette', 'ISOLET', '0.4375', '800']:
            self.assertIn(normalized_text(expected, compact=True), text)
        self.assertNotIn('MinPPI', text)
        self.assertNotIn('ICOMP', text)

    def test_embedded_used_fonts(self):
        page = PdfReader(PDF).pages[0]
        fonts = list(resource_fonts(page['/Resources']))
        self.assertTrue(fonts, 'No PDF fonts found')
        for font in fonts:
            descriptors = list(font_descriptors(font))
            self.assertTrue(descriptors, str(font.get('/BaseFont')))
            for descriptor in descriptors:
                self.assertIsNotNone(descriptor, str(font.get('/BaseFont')))
                streams = [descriptor[key].get_object() for key in
                           ['/FontFile', '/FontFile2', '/FontFile3'] if key in descriptor]
                self.assertTrue(streams, str(font.get('/BaseFont')))
                self.assertTrue(all(stream.get_data() for stream in streams),
                                str(font.get('/BaseFont')))

    def test_layout_bounds_and_sizes(self):
        data = json.loads(MANIFEST.read_text(encoding='utf-8'))
        self.assertEqual(len(data['qrs']), 2)
        for qr in data['qrs']:
            self.assertLess(qr['y_top_pt'], data['page']['height_pt'] * .15)
            self.assertGreater(qr['x_pt'], data['page']['width_pt'] * .75)
        paper, telegram = data['qrs']
        self.assertGreaterEqual(paper['effective_dpi'], 300)
        self.assertEqual(paper['payload'], 'https://hlnks.co/63671101')
        self.assertEqual(telegram['payload'], 'https://t.me/+79516564253')
        self.assertEqual(telegram['format'], 'vector')
        self.assertEqual(telegram['quiet_zone_modules'], 4)
        self.assertGreaterEqual(min(t['font_size'] for t in data['text']), 30)
        for item in data['text']:
            self.assertGreaterEqual(item['x'], 0, item['text'])
            self.assertGreaterEqual(item['y'], 0, item['text'])
            self.assertLessEqual(item['x'] + item['width'], data['page']['width_pt'], item['text'])
            self.assertLessEqual(item['y'] + item['height'], data['page']['height_pt'], item['text'])
        for item in data['fit_checks']:
            self.assertLessEqual(item['used_height'], item['available_height'], item['label'])

    def test_all_visible_glyphs_have_font_outlines(self):
        data = json.loads(MANIFEST.read_text(encoding='utf-8'))
        self.assertIn('glyph_audit', data)
        self.assertGreater(data['glyph_audit']['checked_characters'], 1000)
        self.assertEqual(data['glyph_audit']['missing'], [])

    def test_three_columns_and_centered_titles(self):
        data = json.loads(MANIFEST.read_text(encoding='utf-8'))
        self.assertEqual(data.get('layout', {}).get('columns'), 3)
        expectations = [('ASCOMP 2026 POSTER', 96), ('Min-Marginal Power Indices', 112),
                        ('for Multi-Objective Feature Selection', 112)]
        for text, minimum_size in expectations:
            matches = [t for t in data['text'] if text in t['text']]
            self.assertEqual(len(matches), 1, text)
            item = matches[0]
            self.assertEqual(item.get('alignment'), 'center', text)
            self.assertAlmostEqual(item['x'] + item['width'] / 2, data['page']['width_pt'] / 2, places=3)
            self.assertGreaterEqual(item['font_size'], minimum_size, text)

    def test_split_identity_and_reused_header_space(self):
        data = json.loads(MANIFEST.read_text(encoding='utf-8'))
        midpoint = data['page']['width_pt'] / 2
        for text, minimum_size in [('Wenjie Shi', 76), ('Saint Petersburg State University', 52)]:
            item = next(t for t in data['text'] if text in t['text'])
            self.assertEqual(item['alignment'], 'left', text)
            self.assertLess(item['x'] + item['width'], midpoint, text)
            self.assertGreaterEqual(item['font_size'], minimum_size, text)
        for text, minimum_size in [('Universitetskaya', 42), ('st147487@student.spbu.ru', 40),
                                   ('Corresponding author:', 40)]:
            item = next(t for t in data['text'] if text in t['text'])
            self.assertEqual(item['alignment'], 'right', text)
            self.assertGreater(item['x'], midpoint, text)
            self.assertGreaterEqual(item['font_size'], minimum_size, text)

    def test_body_density_is_bounded(self):
        new_words = len(poster_text().split())
        # The latest request explicitly asks for denser, fuller three-column copy.
        self.assertLess(new_words, 850)

    def test_contribution_focused_copy(self):
        text = poster_text(compact=True).lower()
        for phrase in ['does not guarantee', 'not significant', 'no overall superiority',
                       'sampling can still miss', 'non-inferiority']:
            self.assertNotIn(normalized_text(phrase, compact=True), text, phrase)
        for phrase in ['coalition witness', 'algorithm flow', '0.09515', '0.4375',
                       'matched-count removal', '+0.0047', '-0.0732']:
            self.assertIn(normalized_text(phrase, compact=True), text, phrase)

    def test_algorithm_and_qr_copy(self):
        text = poster_text(compact=True)
        for phrase in ['Algorithm A', 'Algorithm B', 'Validate joint removal',
                       'Evaluate', 'MI-correlation', 'exact', 'PASS', 'FAIL', 'NSGA-II']:
            self.assertIn(normalized_text(phrase, compact=True), text)
        self.assertNotIn(normalized_text('Scan to read the full paper', compact=True), text)
        self.assertNotIn('hlnks.co/', text)
        self.assertFalse(PdfReader(PDF).pages[0].get('/Annots'))

    def test_full_algorithm_and_core_tools(self):
        text = poster_text(compact=True)
        for phrase in ['MinSS', 'MinBz', 'Shapley-Shubik', 'Banzhaf',
                       'mutual information', 'maximum-weight', 'correlation tree',
                       'top-MI half', '0.3', 'Stop at', 'K ∈ {2, 5, 10, 25}', 'CUDA',
                       'TreeSHAP', 'MinShap', 'variance filtering',
                       'population 40', 'clip', '(1.1, 1.1)',
                       '0.443', '0.313', 'F1', 'class priors']:
            self.assertIn(normalized_text(phrase, compact=True), text)
        for phrase in ['reset S ← Bi', 'MinPI-prefiltered',
                       'ΔHV = HVprefiltered - HVfull',
                       'Non-dominated sorting genetic algorithm II',
                       'Hybrid uses exact enumeration: F1 = 1 at every K',
                       'Abalone: mean harmful-set F1 vs exact; 8 features, 5 seeds']:
            self.assertIn(normalized_text(phrase, compact=True), text)
        comparison = json.loads(MANIFEST.read_text(encoding='utf-8'))['estimator_f1']
        self.assertEqual(comparison['budgets'], [2, 5, 10, 25])
        self.assertEqual(comparison['A'], [.100, .150, .230, .443])
        self.assertEqual(comparison['B'], [.000, .080, .133, .313])
        self.assertEqual(comparison['hybrid'], [1., 1., 1., 1.])

    def test_xelatex_compilation_has_no_overflow_or_missing_characters(self):
        data = json.loads(MANIFEST.read_text(encoding='utf-8'))
        typesetting = data['typesetting']
        self.assertEqual(typesetting['engine'], 'XeLaTeX')
        self.assertIs(typesetting['compile_success'], True)
        self.assertEqual(typesetting['overfull_boxes'], 0)
        self.assertEqual(typesetting['missing_characters'], 0)
        log = (BUILD_DIR / 'main.log').read_text(encoding='utf-8')
        self.assertIn('This is XeTeX', log)
        self.assertRegex(log.replace('\n', ''), r'Output written on .*main\.pdf \(1 page')
        self.assertNotRegex(log, r'Overfull \\[hv]box|Missing character:')
        self.assertEqual(hashlib.sha256(PDF.read_bytes()).hexdigest(),
                         hashlib.sha256((BUILD_DIR / 'main.pdf').read_bytes()).hexdigest())

    def test_latin_modern_math_is_embedded(self):
        page = PdfReader(PDF).pages[0]
        math_fonts = [font for font in resource_fonts(page['/Resources'])
                      if 'LatinModernMath' in str(font.get('/BaseFont', ''))]
        self.assertTrue(math_fonts, 'Compiled PDF has no Latin Modern Math font')
        for font in math_fonts:
            self.assertEqual(font['/Subtype'], '/Type0')
            self.assertIn('/ToUnicode', font)
            descriptors = list(font_descriptors(font))
            self.assertTrue(descriptors)
            for descriptor in descriptors:
                self.assertIsNotNone(descriptor)
                self.assertIn('LatinModernMath', str(descriptor.get('/FontName', '')))
                self.assertTrue(any(key in descriptor for key in
                                    ['/FontFile', '/FontFile2', '/FontFile3']))

    def test_core_mathematics_is_generated_and_compiled_as_latex(self):
        content = (TEX_DIR / 'poster-content.tex').read_text(encoding='utf-8')
        for expression in [
                r'v_\alpha(S)=\mathbf{1}\!\left[\Uval(S)\geq\alpha\Uval(N)\right]',
                r'\Delta_i(S)=v_\alpha(S\cup\{i\})-v_\alpha(S)',
                r'\MinPI_i=\MinSS_i=\MinBz_i=\min_{S\subseteq N\setminus\{i\}}\Delta_i(S)',
                r'H=\bigl\{i\in N:\widehat{\MinPI}_i=-1\bigr\}',
                r'\Uval(N\setminus H)\geq\Uval(N)-\varepsilon',
                r'\frac{|S|}{n}', r'\operatorname{clip}',
                r'\min\{20,\lfloor n/3\rfloor\}',
                r'K\in\{2,5,10,25\}', r'K\in\{2,5,10\}',
                r'm\gets\min\{m,\Delta_i(S)\}',
                r'k=|R|', r'S_{\mathrm{ret}}=N\setminus R']:
            with self.subTest(expression=expression):
                self.assertIn(expression, content)
        text = poster_text(compact=True)
        for expression in [
                'vα(S)=1[Uval(S)≥αUval(N)]',
                'Δi(S)=vα(S∪{i})-vα(S)',
                'MinPIi=MinSSi=MinBzi=minS⊆N∖{i}Δi(S)',
                'H={i∈N∶\u0302MinPIi=-1}',
                'Uval(N∖H)≥Uval(N)-ε',
                '(|S|n,1-clip(Uval(S),0,1))',
                'min{20,⌊n/3⌋}', 'K∈{2,5,10,25}', 'K∈{2,5,10}',
                'm←min{m,Δi(S)}', 'k=|R|', 'Sret=N∖R']:
            with self.subTest(expression=expression):
                self.assertIn(normalized_text(expression, compact=True), text)

    def test_actual_latex_body_boxes_are_in_bounds_and_do_not_overlap(self):
        data = json.loads(MANIFEST.read_text(encoding='utf-8'))
        log = (BUILD_DIR / 'main.log').read_text(encoding='utf-8')
        measurements = re.findall(r'POSTERBOX:(\d+):([\d.]+)pt', log)
        measured = {int(index): float(height) * 72 / 72.27
                    for index, height in measurements}
        self.assertEqual(len(measurements), len(data['text']))
        self.assertEqual(set(measured), set(range(len(data['text']))))
        headings = [item for item in data['text']
                    if re.match(r'0[123]  ', item['text'])]
        self.assertEqual(len(headings), 3)
        body = []
        for index, item in enumerate(data['text']):
            self.assertAlmostEqual(item['height'], measured[index], places=5, msg=item['text'])
            if item['y'] < 775:
                continue
            bounds = (item['x'], item['y'], item['x'] + item['width'],
                      item['y'] + measured[index])
            self.assertGreater(measured[index], 0, item['text'])
            self.assertGreaterEqual(bounds[1], 775, item['text'])
            self.assertLessEqual(bounds[3], 2505, item['text'])
            self.assertTrue(any(bounds[0] >= column['x'] - .01 and
                                bounds[2] <= column['x'] + column['width'] + .01
                                for column in headings), item['text'])
            body.append((item['text'], bounds))
        self.assertGreater(len(body), 50)
        for index, (label, bounds) in enumerate(body):
            for other_label, other in body[index + 1:]:
                overlap_x = min(bounds[2], other[2]) - max(bounds[0], other[0])
                overlap_y = min(bounds[3], other[3]) - max(bounds[1], other[1])
                self.assertFalse(overlap_x > .1 and overlap_y > .1,
                                 f'Body text overlaps: {label} / {other_label}')

    def test_original_and_rendered_qr(self):
        sys.path.insert(0, str(ROOT / 'tmp/pdfs/ascomp-poster/deps'))
        import zxingcpp
        from PIL import Image
        inputs = [(OUT / 'assets/QRcode.png', ['https://hlnks.co/63671101']),
                  (OUT / 'assets/tele_code.png', ['https://t.me/+79516564253']),
                  (OUT / 'ASCOMP2026_MinPI_Poster_preview.png',
                   ['https://hlnks.co/63671101', 'https://t.me/+79516564253'])]
        for path, expected in inputs:
            self.assertTrue(path.is_file(), f'Missing QR-check input: {path}')
            results = zxingcpp.read_barcodes(Image.open(path).convert('RGB'))
            for payload in expected:
                self.assertIn(payload, [result.text for result in results], str(path))


if __name__ == '__main__':
    unittest.main(verbosity=2)
