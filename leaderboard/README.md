---
title: SelectArena Leaderboard
emoji: 🎯
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: true
license: apache-2.0
short_description: Leaderboard of feature selection methods on tabular data
---

# SelectArena Leaderboard

Leaderboard of the [SelectArena](https://github.com/schaeferbasti/SelectArena) feature selection
benchmark ([paper](https://arxiv.org/abs/2610.13644)).

This folder is the Hugging Face Space. The app reads the tables in `data/`, which
`scripts/export_leaderboard.py` builds from the benchmark results.

Deploy or update the whole Space (app and data):

```bash
hf upload SelectArena/Leaderboard leaderboard . --repo-type=space
```

Update only the results:

```bash
python scripts/export_leaderboard.py --push
```

Run it locally with `pip install "gradio==5.9.1" pandas && python leaderboard/app.py`.
