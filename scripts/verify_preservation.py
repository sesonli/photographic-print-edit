#!/usr/bin/env python3
"""Read-only exact decoded-pixel check; never a semantic or visual approval."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image, ImageChops


class InvalidInput(ValueError):
    pass


class JsonParser(argparse.ArgumentParser):
    def error(self, message):
        raise InvalidInput(message)


SUPPORTED_MODES = {"1", "L", "LA", "P", "RGB", "RGBA"}
SUPPORTED_FORMATS = {"PNG", "JPEG", "BMP", "TIFF"}
LOSSLESS_TIFF = {"raw", "tiff_lzw", "tiff_adobe_deflate", "tiff_deflate", "packbits", "group3", "group4", "tiff_ccitt"}


def encoded_depth(image, path, role):
    """Check precision before Pillow can hide RGB16 precision in an RGB mode."""
    if image.format not in SUPPORTED_FORMATS:
        raise InvalidInput(f"{role}: supported encoded formats are PNG, JPEG, BMP, TIFF")
    depths = None
    if image.format == "PNG":
        with open(path, "rb") as stream:
            header = stream.read(29)
        if len(header) != 29 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
            raise InvalidInput(f"{role}: invalid PNG IHDR")
        depths = [header[24]]
    elif image.format == "TIFF":
        raw = image.tag_v2.get(258, 1)  # TIFF default BitsPerSample is 1.
        depths = list(raw) if isinstance(raw, (tuple, list)) else [raw]
    elif image.format == "JPEG":
        depths = [getattr(image, "bits", None)]
    if depths is not None:
        if not depths or any(not isinstance(depth, int) or depth < 1 or depth > 8 for depth in depths):
            raise InvalidInput(f"{role}: encoded sample depth must be 1 to 8 bits; high-bit-depth input is unsupported")
    return depths


def read_image(path, role):
    """Decode one frame, without rotation, resampling, or color conversion."""
    try:
        with Image.open(path) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise InvalidInput(f"{role}: multi-frame images are unsupported")
            depths = encoded_depth(image, path, role)
            orientation = image.getexif().get(274, 1)
            if not isinstance(orientation, int) or orientation not in range(1, 9):
                raise InvalidInput(f"{role}: invalid EXIF orientation")
            image.load()
            icc = image.info.get("icc_profile", b"") or b""
            metadata = {
                "path": str(Path(path)),
                "size": list(image.size),
                "format": image.format,
                "mode": image.mode,
                "encoded_bits_per_sample": depths,
                "exif_orientation": orientation,
                "icc_profile_bytes": len(icc),
                "icc_profile_sha256": hashlib.sha256(icc).hexdigest() if icc else None,
                "color_metadata": {key: image.info[key] for key in ("gamma", "srgb", "chromaticity") if key in image.info},
                "dpi": image.info.get("dpi"),
                "compression": image.info.get("compression"),
            }
            has_alpha = "A" in image.getbands() or "transparency" in image.info
            return image.copy(), metadata, has_alpha
    except InvalidInput:
        raise
    except (OSError, ValueError, TypeError, SyntaxError, Image.DecompressionBombError) as exc:
        raise InvalidInput(f"{role}: cannot decode {path}: {exc}") from exc


def require_normal_orientation(metadata, role):
    if metadata["exif_orientation"] != 1:
        raise InvalidInput(f"{role}: EXIF orientation must be 1 or absent; no automatic rotation is performed")


def read_mask(path, role, expected_size):
    mask, metadata, _ = read_image(path, role)
    require_normal_orientation(metadata, role)
    if mask.size != expected_size:
        raise InvalidInput(f"{role}: dimensions must equal source dimensions")
    if mask.mode not in {"1", "L"}:
        raise InvalidInput(f"{role}: use a single-channel grayscale L or 1 mask")
    binary = mask.convert("L").point(lambda value: 255 if value else 0)
    return binary, metadata


def nonzero_count(binary):
    return binary.histogram()[255]


def is_lossless_result(metadata):
    if metadata["format"] in {"PNG", "BMP"}:
        return True
    return metadata["format"] == "TIFF" and metadata["compression"] in LOSSLESS_TIFF


def verify(args, report):
    source, source_meta, source_alpha = read_image(args.source, "source")
    result, result_meta, result_alpha = read_image(args.result, "result")
    report.update(source=source_meta, result=result_meta)
    if source.size != result.size:
        raise InvalidInput("source/result dimensions differ; images are never resized or aligned")
    if source_meta["exif_orientation"] != result_meta["exif_orientation"]:
        raise InvalidInput("source/result EXIF orientations differ")
    require_normal_orientation(source_meta, "source")
    require_normal_orientation(result_meta, "result")
    if source.mode not in SUPPORTED_MODES or result.mode not in SUPPORTED_MODES:
        raise InvalidInput("source/result must decode to supported 8-bit modes (1, L, LA, P, RGB, RGBA); high-bit-depth/CMYK data are not silently reduced")
    report["result_lossless_encoding_accepted"] = is_lossless_result(result_meta)
    if not report["result_lossless_encoding_accepted"] and not args.allow_lossy_result:
        raise InvalidInput("result must be PNG, BMP, or a recognized lossless TIFF; --allow-lossy-result permits only a decoded-sample diagnostic")
    allowed, allowed_meta = read_mask(args.edit_mask, "edit-mask", source.size)
    if not nonzero_count(allowed):
        raise InvalidInput("edit-mask has no nonzero allowed pixels")
    protected = Image.new("L", source.size, 0)
    protected_meta = None
    if args.protected_mask:
        protected, protected_meta = read_mask(args.protected_mask, "protected-mask", source.size)
    report.update(edit_mask=allowed_meta, protected_mask=protected_meta)

    mode = "RGBA" if source_alpha or result_alpha else "RGB"
    # Only mode normalization; no ICC transform, EXIF transpose, or compositing.
    delta = ImageChops.difference(source.convert(mode), result.convert(mode))
    max_channel_delta = delta.getchannel(0)
    for band in delta.split()[1:]:
        max_channel_delta = ImageChops.lighter(max_channel_delta, band)
    changed = max_channel_delta.point(lambda value: 255 if value else 0)
    outside = ImageChops.multiply(changed, ImageChops.invert(allowed))
    protected_changes = ImageChops.multiply(changed, protected)
    forbidden = ImageChops.lighter(ImageChops.invert(allowed), protected)
    violations = ImageChops.multiply(changed, forbidden)
    counts = {
        "pixels_total": source.width * source.height,
        "allowed_pixels": nonzero_count(allowed),
        "protected_pixels": nonzero_count(protected),
        "changed_pixels_total": nonzero_count(changed),
        "changed_pixels_outside_edit_mask": nonzero_count(outside),
        "changed_pixels_in_protected_mask": nonzero_count(protected_changes),
        "violating_pixels_union": nonzero_count(violations),
    }
    report.update(
        normalized_mode=mode,
        counts=counts,
        violation_bbox=list(violations.getbbox()) if violations.getbbox() else None,
        decoded_preservation_passed=counts["violating_pixels_union"] == 0,
        icc_profiles_equal=source_meta["icc_profile_sha256"] == result_meta["icc_profile_sha256"],
        color_metadata_equal=source_meta["color_metadata"] == result_meta["color_metadata"],
    )
    report["warnings"] = []
    if not report["icc_profiles_equal"] or not report["color_metadata_equal"]:
        report["warnings"].append("Color metadata differs; equal decoded samples do not establish equal displayed appearance.")
    if not report["result_lossless_encoding_accepted"]:
        report["warnings"].append("Lossy result override: this is only a comparison of these decoded files, not a lossless-delivery certification.")
    report["status"] = "passed" if report["decoded_preservation_passed"] else "failed"
    return 0 if report["decoded_preservation_passed"] else 1


def main(argv=None):
    report = {
        "schema_version": 1,
        "status": "invalid",
        "claim": "Exact 8-bit normalized decoded RGB(A) samples outside the edit mask and inside the protected mask; protected pixels override allowed pixels.",
        "limits": "No mask authorization, semantic identity, removal quality, visual appearance, original file-byte equality, or high-bit-depth preservation guarantee. No image writes, resizing, alignment, EXIF rotation, or ICC transform.",
    }
    parser = JsonParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("result")
    parser.add_argument("--edit-mask", required=True)
    parser.add_argument("--protected-mask")
    parser.add_argument("--allow-lossy-result", action="store_true")
    try:
        args = parser.parse_args(argv)
        code = verify(args, report)
    except InvalidInput as exc:
        report["error"] = str(exc)
        code = 2
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
