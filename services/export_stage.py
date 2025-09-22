from __future__ import annotations

from pathlib import Path
from app.core.config import settings

async def html_to_pdf_async(html_path: Path, pdf_path: Path) -> None:
    # Use Playwright like original. If MOCK_RENDER is set, create a tiny PDF via fitz instead for tests.
    if settings.MOCK_RENDER:
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Test PDF", fontsize=12)
        doc.save(str(pdf_path))
        doc.close()
        return

    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(html_path.as_uri(), wait_until="networkidle")
        await page.emulate_media(media="print")
        await page.pdf(
            path=str(pdf_path),
            format="A4",
            print_background=True,
            margin={"top": "10mm", "right": "10mm", "bottom": "10mm", "left": "10mm"},
        )
        await browser.close()

