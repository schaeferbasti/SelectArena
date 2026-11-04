"""A local preview of the leaderboard website with locally generated data.

The leaderboard's page (``index.html``, ``assets/``) lives in the Space repository
:data:`SPACE_REPO_ID`; this repo only generates its ``data/`` folder. :func:`prepare_preview`
puts both together in one folder: the page from a local clone of the Space or, by default,
downloaded from Hugging Face (only the page files, not the published data), and the generated
``website_data/`` as its ``data/``. :func:`serve` then serves that folder: the page reads its data
with ``fetch()``, which browsers do not allow for files opened from disk.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

SPACE_REPO_ID = "SelectArena/Leaderboard"
#: The Space's page files (everything but its data/ folder).
SITE_PATTERNS = ["index.html", "assets/*"]


def prepare_preview(
    data_dir: str | Path,
    out_dir: str | Path,
    *,
    site_dir: str | Path | None = None,
    log: Callable[[str], None] = print,
) -> Path:
    """Write the preview site to ``out_dir`` and return it.

    Args:
        data_dir: the generated ``website_data/`` folder (``meta.json``, ``subsets/``, ...).
        out_dir: the preview folder; its ``data/`` is replaced on every call.
        site_dir: a local clone of the Space; ``None`` downloads the page files from
            :data:`SPACE_REPO_ID`.
    """
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    if not (data_dir / "meta.json").is_file():
        raise FileNotFoundError(
            f"{data_dir} has no meta.json: run scripts/run_generate_website_artifacts.py first "
            "(its website data is in generated_website_artifacts/clean_website_artifacts/website_data/)."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    if site_dir is None:
        from huggingface_hub import snapshot_download

        log(f"Downloading the leaderboard page from huggingface.co/spaces/{SPACE_REPO_ID} (without its data)")
        snapshot_download(repo_id=SPACE_REPO_ID, repo_type="space", allow_patterns=SITE_PATTERNS, local_dir=out_dir)
    else:
        site_dir = Path(site_dir)
        if not (site_dir / "index.html").is_file():
            raise FileNotFoundError(f"{site_dir} has no index.html: pass a clone of the Space ({SPACE_REPO_ID}).")
        shutil.copy2(site_dir / "index.html", out_dir / "index.html")
        if (out_dir / "assets").exists():
            shutil.rmtree(out_dir / "assets")
        shutil.copytree(site_dir / "assets", out_dir / "assets")
    # Delete the old data first, so nothing of an earlier preview survives.
    if (out_dir / "data").exists():
        shutil.rmtree(out_dir / "data")
    shutil.copytree(data_dir, out_dir / "data")
    log(f"Preview site: {out_dir}")
    return out_dir


def serve(site_dir: str | Path, port: int = 8000) -> None:
    """Serve ``site_dir`` at http://localhost:<port> until interrupted (Ctrl+C)."""
    import functools
    import http.server

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(site_dir))
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as server:
        print(f"Leaderboard preview at http://localhost:{port} (Ctrl+C to stop)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
