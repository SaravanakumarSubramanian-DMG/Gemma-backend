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
    """Extract only GPS data, timestamp, and user comments from EXIF.
    Returns a dict with optional keys: gps {lat, lon}, timestamp, userComments.
    """
    result: Dict[str, Any] = {}
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
        if 37510 in raw:  # UserComment
            val = raw.get(37510)
            try:
                if isinstance(val, (bytes, bytearray)):
                    # EXIF user comment may start with an 8-byte charset code like ASCII\x00\x00\x00
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


