"""Show the leaderboard website locally with the data you generated.

    python scripts/run_generate_website_artifacts.py [--with-published-results --results ...]
    python scripts/preview_leaderboard.py          # http://localhost:8000

The leaderboard's page (index.html, assets/) comes from the Space repository on Hugging Face
(``selectarena.website.preview.SPACE_REPO_ID``; only the page, not its published data) or from a
local clone of it (``--site``); its data/ folder is the generated website data. The preview folder
(default: generated_website_artifacts/leaderboard_preview/, git-ignored) is rewritten on every run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from selectarena.website.preview import prepare_preview, serve

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO_ROOT / "generated_website_artifacts"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data",
        default=str(ARTIFACTS / "clean_website_artifacts" / "website_data"),
        help="Generated website data (default: the output of run_generate_website_artifacts.py).",
    )
    parser.add_argument("--site", default=None, help="A local clone of the Space (default: download its page).")
    parser.add_argument("--out", default=str(ARTIFACTS / "leaderboard_preview"), help="Preview folder.")
    parser.add_argument("--port", type=int, default=8000, help="Port of the local server.")
    parser.add_argument("--no-serve", action="store_true", help="Only write the preview folder.")
    args = parser.parse_args()

    site = prepare_preview(args.data, args.out, site_dir=args.site)
    if not args.no_serve:
        serve(site, port=args.port)


if __name__ == "__main__":
    main()
