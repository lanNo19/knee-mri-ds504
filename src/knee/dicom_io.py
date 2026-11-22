"""DICOM reading, slice order, canonical frame, laterality and the 32 x 224 x 224 volume.

Each step follows the plan's "Image preparation" list. Nothing here prints or
returns header values to the caller's log; errors carry only types (see
safe_log), and summaries count outcomes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

N_SLICES = 32
SIZE = 224

# Target in-plane axes in patient LPS coordinates (x = patient left,
# y = posterior, z = superior). along_row: direction of increasing column
# index; down_col: direction of increasing row index; stack: slice order.
# Sagittal: anterior on the left, superior at the top, slices right -> left.
# Coronal: patient right on the image left (radiological), superior at the top.
# Axial: patient right on the image left, anterior at the top.
TARGET = {
    "sagittal": {"along_row": (0, 1, 0), "down_col": (0, 0, -1), "stack": (1, 0, 0)},
    "coronal": {"along_row": (1, 0, 0), "down_col": (0, 0, -1), "stack": (0, 1, 0)},
    "axial": {"along_row": (1, 0, 0), "down_col": (0, 1, 0), "stack": (0, 0, 1)},
}

UNCOMPRESSED = {
    "1.2.840.10008.1.2",  # Implicit VR Little Endian
    "1.2.840.10008.1.2.1",  # Explicit VR Little Endian
    "1.2.840.10008.1.2.2",  # Explicit VR Big Endian
    "1.2.840.10008.1.2.1.99",  # Deflated Explicit VR Little Endian
}


def list_dcm(series_dir: Path) -> list[Path]:
    return sorted(Path(e.path) for e in os.scandir(series_dir) if e.is_file() and e.name.lower().endswith(".dcm"))


def _floats(v, n):
    try:
        a = [float(x) for x in v]
        return a if len(a) == n and all(np.isfinite(a)) else None
    except Exception:  # noqa: BLE001
        return None


@dataclass
class SliceHeader:
    path: Path
    ipp: list[float] | None
    iop: list[float] | None
    instance: int | None
    rows: int | None
    cols: int | None
    spacing: list[float] | None


@dataclass
class SeriesInfo:
    """Header facts for one series (first file), as used by N0."""
    transfer_syntax: str | None = None
    manufacturer: str | None = None
    model: str | None = None
    field_strength: float | None = None
    laterality_tag: str | None = None
    has_ipp: bool = False
    has_iop: bool = False
    center_x: float | None = None
    plane_from_iop: str | None = None
    extra: dict = field(default_factory=dict)


def read_header(path: Path):
    import pydicom

    return pydicom.dcmread(str(path), stop_before_pixels=True, force=True)


def slice_header(path: Path) -> SliceHeader:
    ds = read_header(path)
    inst = getattr(ds, "InstanceNumber", None)
    try:
        inst = int(inst) if inst is not None and str(inst).strip() != "" else None
    except Exception:  # noqa: BLE001
        inst = None
    return SliceHeader(
        path=path,
        ipp=_floats(getattr(ds, "ImagePositionPatient", None) or [], 3),
        iop=_floats(getattr(ds, "ImageOrientationPatient", None) or [], 6),
        instance=inst,
        rows=int(ds.Rows) if getattr(ds, "Rows", None) else None,
        cols=int(ds.Columns) if getattr(ds, "Columns", None) else None,
        spacing=_floats(getattr(ds, "PixelSpacing", None) or [], 2),
    )


def norm_laterality(v) -> str | None:
    s = str(v or "").strip().upper()
    if s in {"R", "RIGHT"}:
        return "R"
    if s in {"L", "LEFT"}:
        return "L"
    return None


def plane_of(iop: list[float] | None) -> str | None:
    """Plane from the slice normal: largest component x -> sagittal, y -> coronal, z -> axial."""
    if iop is None:
        return None
    n = np.cross(np.array(iop[:3]), np.array(iop[3:]))
    k = int(np.argmax(np.abs(n)))
    return ("sagittal", "coronal", "axial")[k]


def center_x(h: SliceHeader) -> float | None:
    """Patient x (mm) of the slice centre. In LPS, x > 0 is the patient's left."""
    if h.ipp is None or h.iop is None or h.rows is None or h.cols is None or h.spacing is None:
        return None
    r = np.array(h.iop[:3])
    c = np.array(h.iop[3:])
    row_sp, col_sp = h.spacing  # PixelSpacing = [row spacing, column spacing]
    centre = np.array(h.ipp) + r * (h.cols - 1) / 2 * col_sp + c * (h.rows - 1) / 2 * row_sp
    return float(centre[0])


def series_info(series_dir: Path) -> SeriesInfo:
    files = list_dcm(series_dir)
    info = SeriesInfo()
    if not files:
        return info
    import pydicom

    ds = pydicom.dcmread(str(files[0]), stop_before_pixels=True, force=True)
    fm = getattr(ds, "file_meta", None)
    info.transfer_syntax = str(getattr(fm, "TransferSyntaxUID", "")) or None
    info.manufacturer = (str(getattr(ds, "Manufacturer", "")).strip().upper() or None)
    info.model = (str(getattr(ds, "ManufacturerModelName", "")).strip().upper() or None)
    try:
        fs = getattr(ds, "MagneticFieldStrength", None)
        info.field_strength = round(float(fs), 1) if fs not in (None, "") else None
    except Exception:  # noqa: BLE001
        info.field_strength = None
    info.laterality_tag = norm_laterality(getattr(ds, "Laterality", None))
    h = slice_header(files[0])
    info.has_ipp = h.ipp is not None
    info.has_iop = h.iop is not None
    info.center_x = center_x(h)
    info.plane_from_iop = plane_of(h.iop)
    return info


def geometry_laterality(cx: float | None, margin_mm: float = 20.0) -> str | None:
    """Right knee if the imaged centre is on the patient's right (x < -margin), left if x > margin."""
    if cx is None or not np.isfinite(cx):
        return None
    if cx < -margin_mm:
        return "R"
    if cx > margin_mm:
        return "L"
    return None


# ---------------------------------------------------------------- decoding

def decode(path: Path) -> tuple[np.ndarray, str]:
    """Pixel array of one file and the decoder that worked.

    Order (plan): pydicom (gdcm, then pylibjpeg plugins), then dicomsdl, then
    SimpleITK. Uncompressed files go straight to pydicom's native decoder.
    """
    import pydicom

    ds = pydicom.dcmread(str(path), force=True)
    ts = str(getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", ""))
    errors = []
    if ts in UNCOMPRESSED or ts == "":
        try:
            return _rescale(ds, ds.pixel_array), "pydicom"
        except Exception as e:  # noqa: BLE001
            errors.append(type(e).__name__)
    for plugin in ("gdcm", "pylibjpeg"):
        try:
            from pydicom.pixels import pixel_array as _pa  # pydicom >= 3

            arr = _pa(ds, decoding_plugin=plugin)
            return _rescale(ds, arr), f"pydicom-{plugin}"
        except Exception as e:  # noqa: BLE001
            errors.append(type(e).__name__)
    try:
        return _rescale(ds, ds.pixel_array), "pydicom-auto"
    except Exception as e:  # noqa: BLE001
        errors.append(type(e).__name__)
    try:
        import dicomsdl  # type: ignore

        arr = dicomsdl.open(str(path)).pixelData()
        return _rescale(ds, np.asarray(arr)), "dicomsdl"
    except Exception as e:  # noqa: BLE001
        errors.append(type(e).__name__)
    try:
        import SimpleITK as sitk  # type: ignore

        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img)
        arr = arr[0] if arr.ndim == 3 and arr.shape[0] == 1 else arr
        return np.asarray(arr, dtype=np.float32), "simpleitk"
    except Exception as e:  # noqa: BLE001
        errors.append(type(e).__name__)
    raise DecodeError("+".join(errors))


class DecodeError(RuntimeError):
    """All decoders failed (message lists exception types only)."""


def _rescale(ds, arr) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 3 and a.shape[-1] in (3, 4) and str(getattr(ds, "PhotometricInterpretation", "")).startswith("RGB"):
        a = a[..., :3].mean(-1)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    slope = getattr(ds, "RescaleSlope", None)
    inter = getattr(ds, "RescaleIntercept", None)
    try:
        if slope is not None:
            a = a * float(slope)
        if inter is not None:
            a = a + float(inter)
    except Exception:  # noqa: BLE001
        pass
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        a = a.max() - a
    return a


# ---------------------------------------------------------------- ordering and frame

def order_slices(headers: list[SliceHeader], plane: str) -> tuple[list[SliceHeader], str]:
    """Sort by position along the plane's fixed stack axis; else InstanceNumber; else file order."""
    axis = np.array(TARGET[plane]["stack"], dtype=float)
    if all(h.ipp is not None for h in headers):
        pos = [float(np.dot(h.ipp, axis)) for h in headers]
        idx = sorted(range(len(headers)), key=lambda i: (pos[i], i))
        return [headers[i] for i in idx], "position"
    if all(h.instance is not None for h in headers):
        idx = sorted(range(len(headers)), key=lambda i: (headers[i].instance, i))
        return [headers[i] for i in idx], "instance"
    return list(headers), "file"


def nearest_index(n: int, k: int = N_SLICES) -> np.ndarray:
    """k indices spread over 0..n-1 by nearest index (repeats when n < k)."""
    if n <= 0:
        raise ValueError("empty series")
    return np.rint(np.linspace(0, n - 1, k)).astype(int)


def frame_ops(iop: list[float] | None, plane: str) -> tuple[bool, bool, bool]:
    """(transpose, flip_columns, flip_rows) that bring a slice into the canonical frame."""
    if iop is None:
        return False, False, False
    r = np.array(iop[:3], dtype=float)
    c = np.array(iop[3:], dtype=float)
    tr = np.array(TARGET[plane]["along_row"], dtype=float)
    tc = np.array(TARGET[plane]["down_col"], dtype=float)
    transpose = abs(np.dot(c, tr)) > abs(np.dot(r, tr))
    if transpose:
        r, c = c, r
    flip_cols = np.dot(r, tr) < 0  # along-row axis points the wrong way
    flip_rows = np.dot(c, tc) < 0  # down axis points the wrong way
    return bool(transpose), bool(flip_cols), bool(flip_rows)


def apply_frame(img: np.ndarray, ops: tuple[bool, bool, bool]) -> np.ndarray:
    transpose, flip_cols, flip_rows = ops
    if transpose:
        img = img.T
    if flip_cols:
        img = img[:, ::-1]
    if flip_rows:
        img = img[::-1, :]
    return img


def to_uint8(vol: np.ndarray, lo_q: float = 0.5, hi_q: float = 99.5) -> tuple[np.ndarray, dict]:
    """Clip to the volume's 0.5 and 99.5 percentiles, scale to 0-255."""
    lo, hi = np.percentile(vol, [lo_q, hi_q])
    stats = {"lo": float(lo), "hi": float(hi)}
    if not np.isfinite(hi - lo) or hi - lo <= 1e-6:
        return np.zeros(vol.shape, dtype=np.uint8), {**stats, "blank": True}
    v = (np.clip(vol, lo, hi) - lo) / (hi - lo) * 255.0
    return np.rint(v).astype(np.uint8), {**stats, "blank": False}


def resize_crop(img: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Shorter side to `size`, then centre crop size x size (uint8 in, uint8 out)."""
    from PIL import Image

    h, w = img.shape
    scale = size / min(h, w)
    nh, nw = max(size, int(round(h * scale))), max(size, int(round(w * scale)))
    pil = Image.fromarray(np.ascontiguousarray(img))
    pil = pil.resize((nw, nh), resample=Image.Resampling.BILINEAR, reducing_gap=3.0)
    a = np.asarray(pil)
    top, left = (nh - size) // 2, (nw - size) // 2
    return a[top : top + size, left : left + size]


@dataclass
class VolumeResult:
    volume: np.ndarray  # uint8 [32, 224, 224]
    order_source: str
    frame_ok: bool
    decoders: list[str]
    blank: bool
    n_files: int
    mixed_iop: bool


def prepare_volume(series_dir: Path, plane: str, laterality: str | None) -> VolumeResult:
    """The plan's pipeline for one series: order, sample 32, decode, frame, laterality, intensity, size."""
    files = list_dcm(series_dir)
    if not files:
        raise FileNotFoundError("no dcm files")
    headers = [slice_header(p) for p in files]
    ordered, source = order_slices(headers, plane)
    pick = nearest_index(len(ordered))
    iops = [h.iop for h in ordered if h.iop is not None]
    mixed = len({tuple(np.round(i, 3)) for i in iops}) > 1
    frames, decs = [], []
    cache: dict[int, np.ndarray] = {}
    for i in pick:
        i = int(i)
        if i not in cache:
            arr, dec = decode(ordered[i].path)
            arr = apply_frame(arr, frame_ops(ordered[i].iop, plane))
            cache[i] = arr
            decs.append(dec)
        frames.append(cache[i])
    # Slices of one series can differ in size only if the series is broken; crop to the common size.
    hmin = min(f.shape[0] for f in frames)
    wmin = min(f.shape[1] for f in frames)
    vol = np.stack([f[:hmin, :wmin] for f in frames]).astype(np.float32)
    if laterality == "L":
        if plane == "sagittal":
            vol = vol[::-1]  # reverse the stack: medial and lateral swap sides
        else:
            vol = vol[:, :, ::-1]  # mirror left-right
    vol8, st = to_uint8(vol)
    out = np.stack([resize_crop(s) for s in vol8])
    return VolumeResult(
        volume=out,
        order_source=source,
        frame_ok=all(h.iop is not None for h in ordered),
        decoders=sorted(set(decs)),
        blank=bool(st["blank"]),
        n_files=len(files),
        mixed_iop=mixed,
    )
