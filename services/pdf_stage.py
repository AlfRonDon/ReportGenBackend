from __future__ import annotations

from pathlib import Path
import fitz
from app.core.config import settings
from playwright.async_api import async_playwright


def pdf_to_pngs(pdf_path: Path, out_dir: Path, dpi: int) -> list[Path]:
    assert pdf_path.exists(), f"PDF not found: {pdf_path}"
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    page = doc[0]
    pix = page.get_pixmap(matrix=mat, alpha=False)
    out_png = out_dir / "reference_p1.png"
    pix.save(out_png)
    doc.close()
    return [out_png]


async def render_html_to_png(html_path: Path, out_png: Path) -> None:
    if settings.MOCK_RENDER:
        # produce a placeholder PNG for tests
        from PIL import Image
        img = Image.new("RGB", (3308, 4677), color=(255, 255, 255))
        img.save(out_png)
        return
    html_abs = "file://" + str(html_path.resolve())
    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page = await browser.new_page(viewport={"width": 3308, "height": 4677})
        await page.goto(html_abs, wait_until="networkidle")
        await page.screenshot(path=str(out_png), full_page=True)
        await browser.close()
