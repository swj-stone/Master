"""Read-only preflight for the ASCOMP poster and its protected inputs."""
import hashlib
import json
import sys
import unittest
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
PDF = ROOT / 'output/pdf/ASCOMP2026_MinPI_Poster_120x90cm.pdf'
MANIFEST = ROOT / 'output/poster/layout_manifest.json'
HASHES = {
    'output/pdf/MinPI_arXiv_final.pdf': '2c7cc05471252f1e464230d3a01cd5d1e8d4fb8852fb46f37162a6ca5fc26424',
    'output/QRcode.png': '3b007243b94c94bcc09a92cc86a5c0900ace9d4a4c103e565a84a07e2245435d',
    'MinPI_arXiv_source(1).zip': '5e09469bd8731d4158b16921b737e5c99b56a46e0e95163c2ea93e03d09e4cf6',
}


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
        text = PdfReader(PDF).pages[0].extract_text()
        for expected in ['ASCOMP 2026 POSTER', 'Min-Marginal Power Indices',
                         'Multi-Objective Feature Selection', 'Wenjie Shi', 'Chi Zhao',
                         'Saint Petersburg State University', 'Universitetskaya',
                         '199034', 'Russia', 'Scan to read', '12.53%', '-0.000313',
                         'Madelon', 'Abalone', 'Credit-Card-Fraud', 'Spambase',
                         'Gisette', 'ISOLET', '0.4375', '800']:
            self.assertIn(expected, text)
        self.assertNotIn('MinPPI', text)
        self.assertNotIn('ICOMP', text)

    def test_embedded_used_fonts(self):
        page = PdfReader(PDF).pages[0]
        for resource in page['/Resources']['/Font'].values():
            font = resource.get_object()
            if font.get('/BaseFont') == '/Helvetica':
                # ReportLab creates an unused base resource on initialization.
                continue
            descriptor = font.get('/FontDescriptor')
            self.assertIsNotNone(descriptor, str(font.get('/BaseFont')))
            descriptor = descriptor.get_object()
            self.assertTrue(any(key in descriptor for key in ['/FontFile', '/FontFile2', '/FontFile3']))

    def test_layout_bounds_and_sizes(self):
        data = json.loads(MANIFEST.read_text(encoding='utf-8'))
        self.assertEqual(data['qr']['width_mm'], 90)
        self.assertGreaterEqual(data['qr']['effective_dpi'], 300)
        self.assertGreater(data['qr']['y_top_pt'], data['page']['height_pt'] * .75)
        self.assertLess(data['qr']['x_pt'], data['page']['width_pt'] * .15)
        self.assertEqual(data['qr']['payload'], 'https://hlnks.co/63671101')
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

    def test_two_columns_and_centered_titles(self):
        data = json.loads(MANIFEST.read_text(encoding='utf-8'))
        self.assertEqual(data.get('layout', {}).get('columns'), 2)
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
        objective = next(t for t in data['text'] if 'Minimize feature count' in t['text'])
        self.assertLess(objective['y'], 877)
        self.assertGreater(objective['y'], 730)
        self.assertIn('Maximize predictive utility', objective['text'])

    def test_body_is_simplified(self):
        original = ROOT / 'output/poster/versions/three-column-v1/ASCOMP2026_MinPI_Poster_120x90cm.pdf'
        old_words = len(PdfReader(original).pages[0].extract_text().split())
        new_words = len(PdfReader(PDF).pages[0].extract_text().split())
        self.assertLess(new_words, old_words * .85)

    def test_contribution_focused_copy(self):
        text = PdfReader(PDF).pages[0].extract_text().lower()
        for phrase in ['does not guarantee', 'not significant', 'no overall superiority',
                       'sampling can still miss', 'non-inferiority']:
            self.assertNotIn(phrase, text, phrase)
        for phrase in ['core insight', 'coalition witness', '0.09515', '0.4375',
                       'matched-count removal', '+0.0047', '-0.0732']:
            self.assertIn(phrase, text, phrase)

    def test_original_and_rendered_qr(self):
        sys.path.insert(0, str(ROOT / 'tmp/pdfs/ascomp-poster/deps'))
        import zxingcpp
        from PIL import Image
        for path in [ROOT / 'output/QRcode.png', ROOT / 'output/poster/ASCOMP2026_MinPI_Poster_preview.png']:
            self.assertTrue(path.is_file(), f'Missing QR-check input: {path}')
            results = zxingcpp.read_barcodes(Image.open(path).convert('RGB'))
            self.assertIn('https://hlnks.co/63671101', [result.text for result in results], str(path))


if __name__ == '__main__':
    unittest.main(verbosity=2)
