import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    # Uvicorn config is separate; this sets root/app loggers.
    handlers = [logging.StreamHandler(sys.stdout)]
    logging.basicConfig(level=level, handlers=handlers, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Quiet noisy libraries if needed
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.INFO)


logger = logging.getLogger("app")

