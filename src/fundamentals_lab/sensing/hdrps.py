"""The scene unit 1.2 measures: HDRPS "Luxo Double Checker".

Everything here downloads and nothing is redistributed. The survey's terms are
research use and non-commercial publication, with the source acknowledged as
"Mark Fairchild's HDR Photographic Survey" wherever an image of it appears.

Per scene the survey publishes a linear OpenEXR, an Excel file of the points
metered in the room, a JPEG map showing where those points are, a mosaic of the
individual exposures and a rendered version. The RAW exposures live in a separate
archive, which `archive.py` reads one scene at a time.
"""

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from fundamentals_lab.config import HDRPS_BASE_URL, SCENE, SCENE_SLUG, SENSING_DIR
from fundamentals_lab.sensing import archive


def _sensing_import(module: str):
    """Import an optional dependency, or say plainly how to get it.

    RAW decoding is behind `[project.optional-dependencies] sensing`, so a plain
    `uv sync` in a shared checkout removes it again. The message is the fix.
    """
    from importlib import import_module

    try:
        return import_module(module)
    except ModuleNotFoundError as missing:  # pragma: no cover — environment, not logic
        raise ModuleNotFoundError(
            f"{module} is part of the optional 'sensing' extra — "
            "run `uv sync --extra sensing`"
        ) from missing


ASSETS = {
    "exr": "EXRs/{slug}.exr",
    "data": "Data/{slug}Data.xls",
    "map": "Maps/{slug}Map.jpg",
    "rendered": "Rendered/{slug}Rendered.jpg",
}


def scene_dir(slug: str = SCENE_SLUG) -> Path:
    return SENSING_DIR / slug


def download_assets(slug: str = SCENE_SLUG) -> dict[str, Path]:
    """Fetch the scene's EXR, measurement table, map and rendering."""
    out: dict[str, Path] = {}
    for key, template in ASSETS.items():
        relative = template.format(slug=slug)
        target = scene_dir(slug) / Path(relative).name
        out[key] = target
        if target.exists() and target.stat().st_size > 0:
            continue
        url = f"{HDRPS_BASE_URL}/{relative}"
        target.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"downloading {url}")
        urllib.request.urlretrieve(url, target)  # noqa: S310 — fixed http URL, see config
        logger.info(f"saved {target.name} ({target.stat().st_size / 1e6:.1f} MB)")
    return out


def download_raws(scene: str = SCENE, slug: str = SCENE_SLUG) -> list[Path]:
    """Fetch the scene's RAW exposures out of the 14 GB archive, with checksums."""
    members = archive.scene_members(scene)
    target_dir = scene_dir(slug) / "raw"
    checksums_path = target_dir / "sha256.json"
    checksums = json.loads(checksums_path.read_text()) if checksums_path.exists() else {}
    paths = []
    for member in members:
        target = target_dir / Path(member.name).name
        paths.append(target)
        if target.exists() and target.stat().st_size == member.size:
            continue
        logger.info(f"fetching {member.name} ({member.compressed_size / 1e6:.0f} MB on the wire)")
        checksums[target.name] = archive.fetch(member, target)
    checksums_path.write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n")
    logger.info(f"{len(paths)} exposures in {target_dir}")
    return sorted(paths)


@dataclass(frozen=True)
class MeteredPoint:
    """One colorimeter reading: luminance in cd/m2 plus CIE 1931 chromaticity.

    `chart` is "bright", "dim" or "" — the survey numbers the two ColorCheckers
    from 1 to 24 each, so the patch number alone is ambiguous and the chart has to
    travel with it.
    """

    name: str
    luminance: float
    x: float | None
    y: float | None
    chart: str = ""
    patch: int | None = None

    @property
    def key(self) -> str:
        return f"{self.chart}:{self.patch}" if self.chart else self.name


def _workbook(slug: str):
    xlrd = _sensing_import("xlrd")

    path = scene_dir(slug) / f"{slug}Data.xls"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `uv run sensing-download` first")
    return xlrd.open_workbook(str(path))


def _sheet(slug: str):
    return _workbook(slug).sheet_by_index(0)


def metered_points(slug: str = SCENE_SLUG) -> list[MeteredPoint]:
    """The scene's measured points, read from the survey's own Excel file.

    These readings are the unit's ground truth: they were taken with a Konica
    Minolta CS-100 in the room, so nothing in our pipeline can have influenced
    them. The published EXR cannot serve that role — it is a Photoshop merge whose
    absolute scale was fitted on this scene's own bright chart.

    The sheet labels the first patch of each chart ("Bright Checker 1", "Dim
    Checker 1") and then lets the numbering run, restarting the header block every
    dozen rows. So the chart label is carried forward, and a bare number is read as
    the next patch of the chart currently in force.
    """
    sheet = _sheet(slug)
    points: list[MeteredPoint] = []
    chart = ""
    for row in range(sheet.nrows):
        values = sheet.row_values(row)
        if len(values) < 3:
            continue
        name, luminance = str(values[1]).strip(), values[2]
        if not name or not isinstance(luminance, float):
            continue
        chroma = [v if isinstance(v, float) else None for v in values[3:5]]
        patch: int | None = None
        lowered = name.lower()
        if lowered.startswith("bright checker") or lowered.startswith("dim checker"):
            chart = "bright" if lowered.startswith("bright") else "dim"
            patch = int(float(name.split()[-1]))
        else:
            try:
                patch = int(float(name))
            except ValueError:
                chart, patch = "", None
        points.append(MeteredPoint(name, luminance, chroma[0], chroma[1], chart, patch))
    if not points:
        raise ValueError(f"no metered rows parsed from {scene_dir(slug)}")
    return points


def scene_metadata(slug: str = SCENE_SLUG) -> dict[str, str]:
    """The capture conditions the survey recorded beside the measurements."""
    book = _workbook(slug)
    sheet = book.sheet_by_index(0)
    wanted = {
        "Scene Name:": "scene",
        "Location:": "location",
        "Date:": "date",
        "Camera:": "camera",
        "F/#:": "aperture",
        "Focal Length:": "focal_length",
        "Num. of Exposures:": "exposures",
        "Colorimeter:": "colorimeter",
        "Meas. Angle:": "measurement_angle",
        "Weather:": "conditions",
    }
    # The workbook was written on a Mac, so its dates count from 1904, not 1900.
    # Reading the serial with the wrong epoch moves the shoot four years earlier —
    # 2002 instead of 2006 — which is the kind of quiet error a Data row inherits.
    xlrd = _sensing_import("xlrd")

    found: dict[str, str] = {}
    for row in range(sheet.nrows):
        cells = sheet.row(row)
        values = [str(cell.value).strip() for cell in cells]
        for index, label in enumerate(values):
            if label not in wanted or index + 1 >= len(cells):
                continue
            cell = cells[index + 1]
            if not str(cell.value).strip():
                continue
            if cell.ctype == xlrd.XL_CELL_DATE:
                stamp = xlrd.xldate_as_datetime(cell.value, book.datemode)
                text = stamp.date().isoformat() if stamp.year > 1904 else stamp.time().isoformat()
            else:
                text = str(cell.value).strip()
            found.setdefault(wanted[label], text)
    return found


@dataclass(frozen=True)
class RawFrame:
    """One exposure as the sensor recorded it, before any demosaicing."""

    path: Path
    counts: "object"  # np.ndarray, uint16, one plane, still a Bayer mosaic
    black_level: tuple[int, ...]
    white_level: int
    cfa_pattern: str
    camera: str
    shutter: float | None
    iso: float | None


def load_raw(path: Path) -> RawFrame:
    """Decode one NEF to its raw counts, keeping the numbers the file declares."""
    rawpy = _sensing_import("rawpy")

    with rawpy.imread(str(path)) as raw:
        counts = raw.raw_image_visible.copy()
        black = tuple(int(b) for b in raw.black_level_per_channel)
        white = int(raw.white_level)
        colours = "".join(raw.color_desc.decode())
        pattern = "".join(colours[i] for i in raw.raw_pattern.flatten())
    camera, shutter, iso = _exif(path)
    return RawFrame(path, counts, black, white, pattern, camera, shutter, iso)


def _exif(path: Path) -> tuple[str, float | None, float | None]:
    """Camera, exposure time and ISO, read from the file rather than assumed."""
    exifread = _sensing_import("exifread")

    with path.open("rb") as handle:
        tags = exifread.process_file(handle, details=False)

    def ratio(key: str) -> float | None:
        tag = tags.get(key)
        if tag is None or not tag.values:
            return None
        value = tag.values[0]
        return float(value.num) / float(value.den) if hasattr(value, "den") else float(value)

    model = str(tags.get("Image Model", "")).strip()
    iso = tags.get("EXIF ISOSpeedRatings")
    return model, ratio("EXIF ExposureTime"), float(iso.values[0]) if iso else None
