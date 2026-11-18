"""Synthetic fixtures only; no private photographs or network access."""

import json
import subprocess
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_preservation.py"


def write_rgb16_png(path, width=2, height=1, last_value=0x1E01):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    header = struct.pack(">IIBBBBB", width, height, 16, 2, 0, 0, 0)
    row = b"\x00" + struct.pack(">6H", 0x0A01, 0x1401, 0x1E01, 0x0A01, 0x1401, last_value)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(row)) + chunk(b"IEND", b""))


def write_rgb16_tiff(path):
    # Single strip, little-endian RGB16; Pillow exposes this as mode RGB.
    fields = [(256, 4, 1, 2), (257, 4, 1, 1), (258, 3, 3, 134),
              (259, 3, 1, 1), (262, 3, 1, 2), (273, 4, 1, 140),
              (277, 3, 1, 3), (278, 4, 1, 1), (279, 4, 1, 12), (284, 3, 1, 1)]
    encoded = b"II" + struct.pack("<HIH", 42, 8, len(fields))
    for tag, kind, count, value in fields:
        encoded += struct.pack("<HHI", tag, kind, count)
        encoded += struct.pack("<H", value) + b"\x00\x00" if kind == 3 and count == 1 else struct.pack("<I", value)
    encoded += struct.pack("<I3H6H", 0, 16, 16, 16, 0x0A01, 0x1401, 0x1E01, 0x0A01, 0x1401, 0x1E01)
    path.write_bytes(encoded)


class PreservationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="test-fixture-", dir=SCRIPT.parents[1])
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.png"
        self.result = self.root / "result.png"
        self.edit = self.root / "edit.png"
        Image.new("RGB", (4, 3), (10, 20, 30)).save(self.source)
        Image.open(self.source).save(self.result)
        mask = Image.new("L", (4, 3), 0)
        mask.putpixel((1, 1), 1)  # Any nonzero value is allowed, not opacity.
        mask.save(self.edit)

    def run_check(self, extra=(), edit=True, result=None):
        command = [sys.executable, str(SCRIPT), str(self.source), str(result or self.result)]
        if edit:
            command.extend(["--edit-mask", str(self.edit)])
        command.extend(map(str, extra))
        process = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(process.stderr, "", process.stderr)
        return process.returncode, json.loads(process.stdout)

    def change(self, position, color):
        with Image.open(self.result) as image:
            image.putpixel(position, color)
            image.save(self.result)

    def test_identical_passes_without_writing_inputs(self):
        before = {path: path.read_bytes() for path in (self.source, self.result, self.edit)}
        code, report = self.run_check()
        self.assertEqual(code, 0)
        self.assertEqual(report["counts"]["changed_pixels_total"], 0)
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_inside_change_permitted(self):
        self.change((1, 1), (99, 20, 30))
        code, report = self.run_check()
        self.assertEqual(code, 0)
        self.assertEqual(report["counts"]["changed_pixels_total"], 1)
        self.assertEqual(report["counts"]["violating_pixels_union"], 0)

    def test_outside_change_detected(self):
        self.change((3, 2), (10, 20, 31))
        code, report = self.run_check()
        self.assertEqual(code, 1)
        self.assertEqual(report["counts"]["changed_pixels_outside_edit_mask"], 1)
        self.assertEqual(report["violation_bbox"], [3, 2, 4, 3])

    def test_protected_mask_overrides_allowed(self):
        protected = self.root / "protected.png"
        Image.open(self.edit).save(protected)
        self.change((1, 1), (99, 20, 30))
        code, report = self.run_check(["--protected-mask", protected])
        self.assertEqual(code, 1)
        self.assertEqual(report["counts"]["changed_pixels_outside_edit_mask"], 0)
        self.assertEqual(report["counts"]["changed_pixels_in_protected_mask"], 1)

    def test_protected_and_outside_counts_union_once(self):
        protected = self.root / "protected.png"
        mask = Image.new("L", (4, 3), 0)
        mask.putpixel((3, 2), 255)
        mask.save(protected)
        self.change((3, 2), (99, 20, 30))
        code, report = self.run_check(["--protected-mask", protected])
        self.assertEqual(code, 1)
        self.assertEqual(report["counts"]["changed_pixels_outside_edit_mask"], 1)
        self.assertEqual(report["counts"]["changed_pixels_in_protected_mask"], 1)
        self.assertEqual(report["counts"]["violating_pixels_union"], 1)

    def test_source_result_dimension_mismatch_rejected(self):
        Image.new("RGB", (3, 3)).save(self.result)
        code, report = self.run_check()
        self.assertEqual(code, 2)
        self.assertIn("dimensions differ", report["error"])

    def test_mask_dimension_mismatch_rejected(self):
        Image.new("L", (3, 3), 255).save(self.edit)
        code, report = self.run_check()
        self.assertEqual(code, 2)
        self.assertIn("dimensions", report["error"])

    def test_missing_mask_argument_and_file_rejected(self):
        self.assertEqual(self.run_check(edit=False)[0], 2)
        self.edit.unlink()
        self.assertEqual(self.run_check()[0], 2)

    def test_invalid_or_empty_masks_rejected(self):
        Image.new("RGB", (4, 3), "white").save(self.edit)
        self.assertEqual(self.run_check()[0], 2)
        Image.new("L", (4, 3), 0).save(self.edit)
        self.assertEqual(self.run_check()[0], 2)
        self.edit.write_bytes(b"not an image")
        self.assertEqual(self.run_check()[0], 2)

    def test_invalid_protected_masks_rejected(self):
        protected = self.root / "protected.png"
        self.assertEqual(self.run_check(["--protected-mask", protected])[0], 2)
        Image.new("L", (3, 3), 255).save(protected)
        self.assertEqual(self.run_check(["--protected-mask", protected])[0], 2)
        Image.new("RGB", (4, 3), "white").save(protected)
        self.assertEqual(self.run_check(["--protected-mask", protected])[0], 2)

    def test_alpha_channel_change_detected(self):
        Image.new("RGBA", (4, 3), (10, 20, 30, 255)).save(self.result)
        self.change((3, 2), (10, 20, 30, 254))
        code, report = self.run_check()
        self.assertEqual(code, 1)
        self.assertEqual(report["normalized_mode"], "RGBA")
        self.assertEqual(report["counts"]["changed_pixels_total"], 1)

    def test_opaque_alpha_and_rgb_normalize_equal(self):
        Image.new("RGBA", (4, 3), (10, 20, 30, 255)).save(self.result)
        self.assertEqual(self.run_check()[0], 0)

    def test_hidden_rgb_change_under_zero_alpha_detected(self):
        Image.new("RGBA", (4, 3), (10, 20, 30, 0)).save(self.source)
        Image.open(self.source).save(self.result)
        self.change((3, 2), (11, 20, 30, 0))
        self.assertEqual(self.run_check()[0], 1)

    def test_exif_orientation_mismatch_and_non_normal_rejected(self):
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (4, 3), (10, 20, 30)).save(self.result, exif=exif)
        code, report = self.run_check()
        self.assertEqual(code, 2)
        self.assertIn("orientations differ", report["error"])
        Image.new("RGB", (4, 3), (10, 20, 30)).save(self.source, exif=exif)
        code, report = self.run_check()
        self.assertEqual(code, 2)
        self.assertIn("must be 1 or absent", report["error"])

    def test_icc_difference_reported_without_appearance_claim(self):
        Image.new("RGB", (4, 3), (10, 20, 30)).save(self.result, icc_profile=b"synthetic-metadata-not-a-real-profile")
        code, report = self.run_check()
        self.assertEqual(code, 0)
        self.assertFalse(report["icc_profiles_equal"])
        self.assertTrue(report["warnings"])

    def test_lossy_result_rejected_by_default(self):
        jpeg = self.root / "result.jpg"
        Image.open(self.source).save(jpeg)
        code, report = self.run_check(result=jpeg)
        self.assertEqual(code, 2)
        self.assertFalse(report["result_lossless_encoding_accepted"])
        code, report = self.run_check(["--allow-lossy-result"], result=jpeg)
        self.assertIn(code, (0, 1))
        self.assertFalse(report["result_lossless_encoding_accepted"])
        self.assertTrue(report["warnings"])

    def test_rgb16_png_hidden_precision_rejected(self):
        write_rgb16_png(self.source)
        write_rgb16_png(self.result, last_value=0x1E02)
        # These differ in a low sample byte that Pillow's RGB decode loses.
        with Image.open(self.source) as image:
            self.assertEqual(image.mode, "RGB")
        with Image.open(self.result) as image:
            self.assertEqual(image.mode, "RGB")
        Image.new("L", (2, 1), 0).save(self.edit)
        code, report = self.run_check()
        self.assertEqual(code, 2)
        self.assertIn("encoded sample depth", report["error"])

    def test_rgb16_tiff_hidden_precision_rejected(self):
        self.source = self.root / "source.tiff"
        write_rgb16_tiff(self.source)
        with Image.open(self.source) as image:
            self.assertEqual(image.mode, "RGB")
        code, report = self.run_check()
        self.assertEqual(code, 2)
        self.assertIn("encoded sample depth", report["error"])

    def test_decompression_bomb_mask_returns_invalid_json(self):
        # Header triggers Pillow's safety rejection before data allocation.
        write_rgb16_png(self.edit, width=20000, height=20000)
        code, report = self.run_check()
        self.assertEqual(code, 2)
        self.assertIn("edit-mask: cannot decode", report["error"])


if __name__ == "__main__":
    unittest.main()
