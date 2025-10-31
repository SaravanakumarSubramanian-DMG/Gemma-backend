import math
from typing import Any, Dict

from PIL import Image, ExifTags


def _ratio_to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        try:
            if isinstance(x, tuple) and len(x) == 2 and x[1]:
                return float(x[0]) / float(x[1])
        except Exception:
            pass
    return float("nan")


def _dms_to_degrees(dms: Any, ref: str | None) -> float | None:
    try:
        d, m, s = dms
        deg = _ratio_to_float(d) + _ratio_to_float(m) / 60.0 + _ratio_to_float(s) / 3600.0
        if ref in ("S", "W"):
            deg = -deg
        return deg
    except Exception:
        return None


def extract_exif_fields(img: Image.Image) -> Dict[str, Any]:
    """Extract GPS data (lat/lon), timestamp, and user comments from EXIF.
    Returns a dict with keys: gps (or None), timestamp (or None), userComments (or None).
    """
    # Initialize with explicit keys so downstream always sees consistent structure
    result: Dict[str, Any] = {"gps": None, "timestamp": None, "userComments": None}
    try:
        raw = None
        try:
            raw = img.getexif()
        except Exception:
            try:
                raw = img._getexif()  # type: ignore[attr-defined]
            except Exception:
                raw = None
        if not raw:
            return result

        # Timestamps
        ts = None
        for tag_id in (36867, 36868, 306):  # DateTimeOriginal, DateTimeDigitized, DateTime
            if tag_id in raw:
                val = raw.get(tag_id)
                if val:
                    ts = str(val)
                    break
        if ts:
            result["timestamp"] = ts

        # User comment (note: often bytes, may need decoding)
        user_cmt = None
        # Prefer EXIF UserComment (37510)
        if 37510 in raw:
            val = raw.get(37510)
            try:
                if isinstance(val, (bytes, bytearray)):
                    prefixes = [b"ASCII\x00\x00\x00", b"UNICODE\x00", b"JIS\x00\x00\x00"]
                    b = bytes(val)
                    for p in prefixes:
                        if b.startswith(p):
                            b = b[len(p):]
                            break
                    user_cmt = b.decode("utf-8", errors="ignore").strip() or None
                else:
                    user_cmt = str(val).strip() or None
            except Exception:
                user_cmt = None
        # Fallback to ImageDescription (270)
        if not user_cmt and 270 in raw:
            try:
                val270 = raw.get(270)
                if isinstance(val270, (bytes, bytearray)):
                    user_cmt = bytes(val270).decode("utf-8", errors="ignore").strip() or None
                else:
                    user_cmt = str(val270).strip() or None
            except Exception:
                pass
        # Fallback to XPComment (40092) - often UTF-16LE and may be a list of ints
        if not user_cmt and 40092 in raw:
            try:
                v = raw.get(40092)
                if isinstance(v, (bytes, bytearray)):
                    user_cmt = bytes(v).decode("utf-16le", errors="ignore").rstrip("\x00").strip() or None
                elif isinstance(v, (list, tuple)):
                    b = bytes(int(x) & 0xFF for x in v)
                    user_cmt = b.decode("utf-16le", errors="ignore").rstrip("\x00").strip() or None
                else:
                    user_cmt = str(v).strip() or None
            except Exception:
                pass
        if user_cmt:
            result["userComments"] = user_cmt

        # GPS
        gps_block = raw.get(34853)  # GPSInfo
        if gps_block:
            inv = ExifTags.GPSTAGS
            gps_named = {inv.get(k, k): v for k, v in dict(gps_block).items()}
            lat = lon = None
            lat_vals = gps_named.get("GPSLatitude")
            lat_ref = gps_named.get("GPSLatitudeRef")
            lon_vals = gps_named.get("GPSLongitude")
            lon_ref = gps_named.get("GPSLongitudeRef")
            if lat_vals and lat_ref:
                lat = _dms_to_degrees(lat_vals, str(lat_ref))
            if lon_vals and lon_ref:
                lon = _dms_to_degrees(lon_vals, str(lon_ref))
            gps_out: Dict[str, Any] = {}
            if lat is not None and not math.isnan(lat):
                gps_out["lat"] = lat
            if lon is not None and not math.isnan(lon):
                gps_out["lon"] = lon
            if gps_out:
                result["gps"] = gps_out
    except Exception:
        return result
    return result


