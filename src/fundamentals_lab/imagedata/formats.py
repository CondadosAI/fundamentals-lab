"""L5 — what the file did to the numbers before you read them.

The sweep runs on our own 8-bit development, which matters: the survey's rendered JPEG
is a 600x337 thumbnail that has already been through an encoder, and measuring a second
generation of loss would answer a question nobody asked. Here generation one is ours.

Four things get measured per encoding: what it costs in bytes, how far the pixels moved,
how far the *colours* moved on the chart, and how much of what an edge detector consumes
survives. The last column is the one a practitioner has not seen, and it is the reason
this lesson exists.
"""

import io

import cv2
import numpy as np

from fundamentals_lab.config import (
    CONTENT_CROP,
    IMAGEDATA_CANNY,
    JPEG_QUALITIES,
    WEBP_QUALITY,
)
from fundamentals_lab.imagedata.scene import Frame
from fundamentals_lab.sensing import charts
from fundamentals_lab.sensing.hdrps import _sensing_import

# Tolerance for the edge comparison, in pixels. An edge map is one pixel wide, so an
# exact-match score punishes a contour that moved by one pixel as hard as one that
# vanished. One pixel of slack is the usual allowance and it is stated rather than
# buried, because the number means nothing without it.
EDGE_TOLERANCE_PX = 1


def _encodings() -> list[dict]:
    """The formats compared, in the order the lesson walks them."""
    rows = [
        {"label": "PNG", "ext": ".png", "params": [cv2.IMWRITE_PNG_COMPRESSION, 9], "lossy": False},
        {
            "label": "WebP lossless",
            "ext": ".webp",
            "params": [cv2.IMWRITE_WEBP_QUALITY, 101],  # >100 selects lossless in OpenCV
            "lossy": False,
        },
        {
            "label": f"WebP q{WEBP_QUALITY}",
            "ext": ".webp",
            "params": [cv2.IMWRITE_WEBP_QUALITY, WEBP_QUALITY],
            "lossy": True,
        },
    ]
    rows.append(
        {"label": "AVIF q82", "ext": ".avif", "params": [cv2.IMWRITE_AVIF_QUALITY, 82], "lossy": True}
    )
    rows += [
        {
            "label": f"JPEG q{quality}",
            "ext": ".jpg",
            "params": [cv2.IMWRITE_JPEG_QUALITY, quality],
            "lossy": True,
        }
        for quality in JPEG_QUALITIES
    ]
    return rows


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def _patch_lab_for(image: np.ndarray, patches: list) -> dict[int, np.ndarray]:
    """Mean CIELAB of each given patch, on OpenCV's 8-bit Lab scale."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab).astype(np.float64)
    return {p.number: p.sample(lab).reshape(-1, 3).mean(axis=0) for p in patches}


def _patch_lab(image: np.ndarray) -> dict[int, np.ndarray]:
    return _patch_lab_for(image, charts.patches("bright"))


def _inside(patch, crop: tuple[int, int, int, int]) -> bool:
    x0, y0, x1, y1 = crop
    px0, py0, px1, py1 = patch.box
    return px0 >= x0 and py0 >= y0 and px1 <= x1 and py1 <= y1


def _shift(patch, dx: int, dy: int):
    """The same patch, addressed in a crop's coordinates."""
    from dataclasses import replace

    px0, py0, px1, py1 = patch.box
    return replace(
        patch,
        box=(px0 - dx, py0 - dy, px1 - dx, py1 - dy),
        centre=(patch.centre[0] - dx, patch.centre[1] - dy),
    )


def _delta_e_8bit(a: np.ndarray, b: np.ndarray) -> float:
    """Delta-E between two OpenCV 8-bit Lab triples.

    OpenCV packs L into 0..255 and offsets a,b by 128, so L is rescaled by 100/255
    before the distance is taken. Skipping that rescale inflates every lightness
    difference by a factor of 2.55.
    """
    scale = np.array([100.0 / 255.0, 1.0, 1.0])
    return float(np.linalg.norm((a - b) * scale))


def _edges(image: np.ndarray) -> np.ndarray:
    low, high = IMAGEDATA_CANNY
    return cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), low, high) > 0


def _edge_f1(reference: np.ndarray, candidate: np.ndarray) -> dict:
    """How much of the reference edge set the candidate still produces.

    Matching is within `EDGE_TOLERANCE_PX`, done with a dilation rather than a nearest
    neighbour search: an edge pixel counts as found if any edge pixel lies within the
    tolerance of it.
    """
    kernel = np.ones((2 * EDGE_TOLERANCE_PX + 1,) * 2, np.uint8)
    reference_near = cv2.dilate(reference.astype(np.uint8), kernel) > 0
    candidate_near = cv2.dilate(candidate.astype(np.uint8), kernel) > 0

    if candidate.sum() == 0 or reference.sum() == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    precision = float((candidate & reference_near).sum() / candidate.sum())
    recall = float((reference & candidate_near).sum() / reference.sum())
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "reference_edge_pixels": int(reference.sum()),
        "candidate_edge_pixels": int(candidate.sum()),
    }


def _sweep_region(source: np.ndarray, patches: list) -> dict:
    reference_lab = _patch_lab_for(source, patches)
    reference_edges = _edges(source)
    raw_bytes = source.nbytes

    rows = []
    for encoding in _encodings():
        ok, buffer = cv2.imencode(encoding["ext"], source, encoding["params"])
        if not ok:
            rows.append(
                {"label": encoding["label"], "error": "this build has no encoder for it"}
            )
            continue
        decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        psnr = _psnr(source, decoded)
        row = {
            "label": encoding["label"],
            "lossy": encoding["lossy"],
            "bytes": int(buffer.nbytes),
            "compression_ratio": round(raw_bytes / buffer.nbytes, 1),
            "psnr_db": None if np.isinf(psnr) else round(psnr, 2),
            "byte_identical_to_source": bool(np.array_equal(source, decoded)),
            "edges": _edge_f1(reference_edges, _edges(decoded)),
        }
        if reference_lab:
            lab = _patch_lab_for(decoded, patches)
            errors = [_delta_e_8bit(reference_lab[n], lab[n]) for n in reference_lab]
            row["patch_delta_e_mean"] = round(float(np.mean(errors)), 3)
            row["patch_delta_e_max"] = round(float(np.max(errors)), 3)
        rows.append(row)
    return {
        "shape": list(source.shape),
        "uncompressed_bytes": raw_bytes,
        "rows": rows,
    }


def sweep(frame: Frame) -> dict:
    """The format table, over the whole frame and over the content in it.

    Both, because the two disagree by a factor that would otherwise be invisible: this
    frame is a dark room, and a dark room compresses beautifully no matter what the
    encoder does. Quoting the whole-frame ratio as if it described photographs in
    general is the mistake this pair of rows exists to prevent.
    """
    patches = charts.patches("bright")
    x0, y0, x1, y1 = CONTENT_CROP
    cropped = [_shift(p, x0, y0) for p in patches if _inside(p, CONTENT_CROP)]
    return {
        "source": "our own 8-bit development of " + frame.name,
        "canny_thresholds": list(IMAGEDATA_CANNY),
        "edge_tolerance_px": EDGE_TOLERANCE_PX,
        "delta_e_patches": len(patches),
        "whole_frame": _sweep_region(frame.bgr8, patches),
        "content_crop": {
            "crop_xyxy": list(CONTENT_CROP),
            "why": "the lamp, the chart and the lit table; the rest of the frame is near-black",
            **_sweep_region(frame.bgr8[y0:y1, x0:x1], cropped),
        },
    }


def chroma_subsampling(frame: Frame) -> dict:
    """What 4:2:0 costs, isolated from everything else the encoder does.

    Same image, same quality, one setting changed: the chroma planes are kept at full
    resolution or at half. Any difference between the two rows is subsampling alone.
    """
    source = frame.bgr8
    reference_lab = _patch_lab(source)
    rows = []
    for label, factor in (("4:2:0", cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420),
                          ("4:4:4", cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444)):
        ok, buffer = cv2.imencode(
            ".jpg",
            source,
            [cv2.IMWRITE_JPEG_QUALITY, 85, cv2.IMWRITE_JPEG_SAMPLING_FACTOR, factor],
        )
        if not ok:
            continue
        decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        lab = _patch_lab(decoded)
        errors = [_delta_e_8bit(reference_lab[n], lab[n]) for n in reference_lab]
        rows.append(
            {
                "sampling": label,
                "bytes": int(buffer.nbytes),
                "psnr_db": round(_psnr(source, decoded), 2),
                "patch_delta_e_mean": round(float(np.mean(errors)), 3),
                "patch_delta_e_max": round(float(np.max(errors)), 3),
            }
        )
    return {"quality": 85, "rows": rows}


def decoder_disagreement(frame: Frame) -> dict:
    """Two libraries, the same encoded bytes, side by side.

    The folk version of this is "the same JPEG decoded by OpenCV and PIL can differ".
    Measured here, across four qualities and both sampling factors, it does not: both
    link libjpeg-turbo and the decoded arrays are bit-for-bit identical, and the same
    holds for 8-bit PNG and for WebP.

    The disagreement is real, but it is somewhere else. A 16-bit PNG comes back from
    OpenCV as uint16 and from Pillow as **uint8** -- Pillow opens it in mode "RGB" and
    drops the low eight bits of every sample, with no error and no warning. That is the
    one worth knowing, because it is silent and because it undoes exactly what unit
    1.3's third lesson spends its time earning.
    """
    from PIL import Image, features

    import PIL

    source = frame.bgr8
    # Crops taken from the lit chart rather than the frame corner: the corner of this
    # photograph is black, and two decoders agreeing about black proves nothing.
    lit = source[1880:2280, 2400:2800]
    patch = frame.rgb16[1880:1980, 2440:2540]

    def compare(buffer, label: str, expect_uint16: bool = False) -> dict:
        opencv = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
        with Image.open(io.BytesIO(buffer.tobytes())) as handle:
            mode = handle.mode
            array = np.array(handle)
        pillow = array if array.ndim == 2 else cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        row = {
            "case": label,
            "opencv_dtype": str(opencv.dtype),
            "opencv_range": [int(opencv.min()), int(opencv.max())],
            "pillow_mode": mode,
            "pillow_dtype": str(pillow.dtype),
            "pillow_range": [int(pillow.min()), int(pillow.max())],
            "same_dtype": opencv.dtype == pillow.dtype,
        }
        if row["same_dtype"]:
            difference = np.abs(opencv.astype(np.int32) - pillow.astype(np.int32))
            row["identical"] = bool(difference.max() == 0)
            row["differing_values"] = int((difference > 0).sum())
            row["max_difference"] = int(difference.max())
        else:
            row["identical"] = False
            row["note"] = "not comparable value by value: the two libraries returned different types"
        if expect_uint16:
            row["bits_pillow_kept"] = 8
            row["bits_in_the_file"] = 16
        return row

    rows = []
    for quality in (95, 85, 50, 25):
        ok, buffer = cv2.imencode(".jpg", source, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            rows.append(compare(buffer, f"JPEG q{quality} (4:2:0)"))
    ok, buffer = cv2.imencode(
        ".jpg",
        source,
        [cv2.IMWRITE_JPEG_QUALITY, 85, cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
         cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444],
    )
    if ok:
        rows.append(compare(buffer, "JPEG q85 (4:4:4)"))
    ok, buffer = cv2.imencode(".png", lit)
    if ok:
        rows.append(compare(buffer, "PNG 8-bit"))
    ok, buffer = cv2.imencode(".webp", lit, [cv2.IMWRITE_WEBP_QUALITY, WEBP_QUALITY])
    if ok:
        rows.append(compare(buffer, f"WebP q{WEBP_QUALITY}"))
    ok, buffer = cv2.imencode(".png", patch[:, :, ::-1].copy())
    if ok:
        rows.append(compare(buffer, "PNG 16-bit", expect_uint16=True))

    return {
        "opencv_version": cv2.__version__,
        "pillow_version": PIL.__version__,
        "pillow_libjpeg_turbo": bool(features.check_feature("libjpeg_turbo")),
        "pillow_jpeg_version": features.version("jpg"),
        "rows": rows,
        "agree_on": [r["case"] for r in rows if r.get("identical")],
        "disagree_on": [r["case"] for r in rows if not r.get("identical")],
    }


def exif_orientation(frame: Frame) -> dict:
    """What the file says about which way is up.

    This frame was shot on a tripod and its flag reads "Horizontal (normal)", so nothing
    here rotates. The failure mode is named in the lesson and not demonstrated on it:
    a rotated flag plus a decoder that ignores the flag is a silently transposed array,
    and nothing downstream complains. Reporting the tag we actually have is the honest
    version of that.
    """
    exifread = _sensing_import("exifread")
    from fundamentals_lab.imagedata.scene import frame_path

    with frame_path(frame.name).open("rb") as handle:
        tags = exifread.process_file(handle, details=False)
    wanted = ("Image Orientation", "Image Make", "Image Model", "EXIF ExposureTime", "EXIF ISOSpeedRatings")
    return {tag: str(tags[tag]) for tag in wanted if tag in tags}
