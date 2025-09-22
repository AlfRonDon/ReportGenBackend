from __future__ import annotations

import os
from pathlib import Path
from typing import List


class Settings:
    # OpenAI
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-5")

    # Pipeline
    PDF_DPI: int = int(os.getenv("PDF_DPI", "400"))
    REFINE_ITERS: int = int(os.getenv("REFINE_ITERS", "1"))
    BASE_ARTIFACTS_DIR: Path = Path(os.getenv("BASE_ARTIFACTS_DIR", "./jobs")).resolve()

    # API
    ALLOWED_ORIGINS: List[str] = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]

    # Test/mocks
    MOCK_OPENAI: bool = os.getenv("MOCK_OPENAI", "0") == "1"
    MOCK_RENDER: bool = os.getenv("MOCK_RENDER", "0") == "1"


settings = Settings()

