"""FastAPI application factory for the pyzm ML detection server."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any

import requests as http_requests
from fastapi import Body, Depends, FastAPI, File, Form, HTTPException, UploadFile

from pyzm.ml.detector import Detector
from pyzm.models.config import FrameStrategy, ServerConfig
from pyzm.models.detection import DetectionResult
from pyzm.models.zm import Zone
from pyzm.serve.auth import create_login_route, create_token_dependency

logger = logging.getLogger("pyzm.serve")


def _parse_zones(raw_zones: list[dict] | str | None) -> list[Zone] | None:
    """Parse raw zone data into Zone objects."""
    if not raw_zones:
        return None
    if isinstance(raw_zones, str):
        raw_zones = json.loads(raw_zones)
    return [
        Zone(
            name=z.get("name", ""),
            points=z.get("value", z.get("points", [])),
            pattern=z.get("pattern"),
            ignore_pattern=z.get("ignore_pattern"),
        )
        for z in raw_zones
    ]


def create_app(config: ServerConfig | None = None) -> FastAPI:
    """Build and return a configured FastAPI application.

    The :class:`Detector` is created during the lifespan startup phase so
    models are loaded once and persist across requests.
    """
    config = config or ServerConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if config.detector_config is not None:
            detector = Detector(config=config.detector_config)
            lazy = not config.detector_config.models
        else:
            lazy = config.models == ["all"]
            detector = Detector(
                models=None if lazy else config.models,
                base_path=config.base_path,
                processor=config.processor,
            )
        detector._ensure_pipeline(lazy=lazy)
        app.state.detector = detector
        mode = "lazy" if lazy else "eager"
        logger.info(
            "Detector ready (%s): %d model(s)", mode, len(detector._config.models)
        )
        yield

    app = FastAPI(title="pyzm ML Detection Server", lifespan=lifespan)

    # -- Optional auth -------------------------------------------------------
    auth_deps: list[Any] = []
    if config.auth_enabled:
        verify_token = create_token_dependency(config)
        auth_deps = [Depends(verify_token)]
    # Always register /login so clients with credentials configured don't
    # get a 404.  When auth is disabled the route accepts any credentials.
    app.post("/login")(create_login_route(config))

    # -- Routes --------------------------------------------------------------

    @app.get("/health")
    def health():
        models_loaded = (
            hasattr(app.state, "detector") and app.state.detector._pipeline is not None
        )
        return {"status": "ok", "models_loaded": models_loaded}

    @app.get("/models")
    def list_models():
        """Return the list of available models and their load status."""
        detector: Detector = app.state.detector
        pipeline = detector._pipeline
        if pipeline is None:
            return {"models": []}
        result = []
        for mc, backend in pipeline._backends:
            result.append({
                "name": mc.name or mc.framework.value,
                "type": mc.type.value,
                "framework": mc.framework.value,
                "loaded": backend.is_loaded,
            })
        return {"models": result}

    @app.post("/detect", dependencies=auth_deps)
    async def detect(
        file: UploadFile = File(...),
        zones: str | None = Form(None),
    ):
        import cv2
        import numpy as np

        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")

        arr = np.frombuffer(contents, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        zone_list = None
        if zones:
            try:
                zone_list = _parse_zones(zones)
            except (json.JSONDecodeError, TypeError) as exc:
                raise HTTPException(
                    status_code=400, detail=f"Invalid zones JSON: {exc}"
                )

        detector: Detector = app.state.detector
        result = detector.detect(image, zones=zone_list)

        data = result.to_dict()
        # Strip non-serializable fields
        data.pop("image", None)
        return data

    @app.post("/detect_urls", dependencies=auth_deps)
    async def detect_urls(payload: dict = Body(...)):
        """Detect objects in images fetched from URLs.

        The client sends a list of image URLs (typically ZM frame URLs)
        and the server fetches each one, decodes JPEG, and runs detection.
        """
        import cv2
        import numpy as np

        from pyzm.ml.detector import _is_better

        urls = payload.get("urls", [])
        zm_auth = payload.get("zm_auth", "")
        verify_ssl = payload.get("verify_ssl", True)

        if not urls:
            raise HTTPException(status_code=400, detail="No URLs provided")

        try:
            zone_list = _parse_zones(payload.get("zones"))
        except (TypeError, KeyError) as exc:
            raise HTTPException(
                status_code=400, detail=f"Invalid zones: {exc}"
            )

        orig_shape = None
        raw_shape = payload.get("original_shape")
        if raw_shape:
            orig_shape = tuple(raw_shape)

        detector: Detector = app.state.detector
        strategy = detector._config.frame_strategy
        results = []

        for entry in urls:
            fid = str(entry.get("frame_id", ""))
            url = entry.get("url", "")
            if not url:
                continue

            # Append ZM auth
            if zm_auth:
                sep = "&" if "?" in url else "?"
                url = f"{url}{sep}{zm_auth}"

            try:
                resp = http_requests.get(url, timeout=10, verify=verify_ssl)
                resp.raise_for_status()
                arr = np.frombuffer(resp.content, dtype=np.uint8)
                image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if image is None:
                    logger.warning("Could not decode image from URL for frame %s", fid)
                    continue
            except Exception:
                logger.exception("Failed to fetch frame %s", fid)
                continue

            result = detector.detect(image, zones=zone_list, original_shape=orig_shape)
            result.frame_id = fid
            results.append(result)

            # Short-circuit for 'first' / 'first_new' strategy
            if strategy in (FrameStrategy.FIRST, FrameStrategy.FIRST_NEW) and result.matched:
                break

        if not results:
            return DetectionResult().to_dict()

        best = results[0]
        for r in results[1:]:
            if _is_better(r, best, strategy):
                best = r

        data = best.to_dict()
        data.pop("image", None)
        return data

    return app
