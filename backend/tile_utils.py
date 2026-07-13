"""
Tile utility functions for NETRA backend.
Converts lat/lng to XYZ tile coordinates, fetches real satellite imagery
from Esri World Imagery, and processes tiles to simulate SAR radar texture.
"""

import math
import io
import httpx
from PIL import Image, ImageEnhance, ImageOps, ImageFilter


def lat_lng_to_tile(lat: float, lng: float, zoom: int = 13):
    """Convert decimal lat/lng to Slippy Map XYZ tile coordinates."""
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    x = int((lng + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    # Clamp to valid tile range
    x = max(0, min(n - 1, x))
    y = max(0, min(n - 1, y))
    return x, y, zoom


async def fetch_optical_tile(x: int, y: int, z: int) -> bytes:
    """
    Fetch a real optical satellite tile from Esri World Imagery.
    Free public service, no API key required.
    """
    url = (
        f"https://server.arcgisonline.com/ArcGIS/rest/services/"
        f"World_Imagery/MapServer/tile/{z}/{y}/{x}"
    )
    headers = {
        "User-Agent": "NETRA-Anudrishti/1.0 (Bharatiya Antariksh Hackathon 2026)"
    }
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.content


async def fetch_geocode(query: str) -> list[dict]:
    """
    Geocode a place name using OpenStreetMap Nominatim.
    Free public service, no API key required.
    Returns list of {name, lat, lng, display_name} results.
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": query,
        "format": "json",
        "limit": 5,
        "addressdetails": 0,
    }
    headers = {"User-Agent": "NETRA-Anudrishti/1.0"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url, params=params, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    results = []
    for item in data:
        results.append({
            "name": item.get("name", item.get("display_name", "")[:50]),
            "display_name": item.get("display_name", ""),
            "lat": float(item["lat"]),
            "lng": float(item["lon"]),
        })
    return results


def process_as_sar(img_bytes: bytes) -> bytes:
    """
    Transform an optical satellite tile into a SAR-like radar image.
    Pipeline: RGB → Grayscale → High Contrast → Histogram Equalize →
              Edge enhancement → Posterize (simulate speckle quantization)
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Step 1: Desaturate to luminance grayscale
    gray = img.convert("L")

    # Step 2: Boost contrast sharply (radar images are high-contrast)
    gray = ImageEnhance.Contrast(gray).enhance(2.8)

    # Step 3: Histogram equalization to spread intensities
    gray = ImageOps.equalize(gray)

    # Step 4: Sharpen edges (SAR images show crisp structure boundaries)
    gray = gray.filter(ImageFilter.SHARPEN)
    gray = gray.filter(ImageFilter.SHARPEN)

    # Step 5: Posterize to simulate SAR quantization / speckle blocks
    gray = ImageOps.posterize(gray, 5)

    # Step 6: Slight brightness reduction (SAR tends to be dark)
    gray = ImageEnhance.Brightness(gray).enhance(0.85)

    out = io.BytesIO()
    gray.save(out, format="PNG", optimize=True)
    return out.getvalue()


def scale_tile_image(img_bytes: bytes, size: int = 512) -> bytes:
    """Resize a tile to a standard display size."""
    img = Image.open(io.BytesIO(img_bytes))
    img = img.resize((size, size), Image.LANCZOS)
    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()
