"""SelectArena leaderboard (Hugging Face Space).

Shows the tables written by ``scripts/export_leaderboard.py`` into ``data/``. The app has no
dependency on SelectArena itself, so the Space only needs Gradio and pandas.
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

VIEWS = {
    "Overall": ("overall", None),
    "By budget": ("by_budget", "budget_index"),
    "By downstream model": ("by_model", "model"),
    "By problem type": ("by_problem_type", "problem_type"),
}
COLUMN_NAMES = {
    "method": "Method",
    "avg_rank": "Avg. rank ↓",
    "win_rate": "Win rate ↑",
    "improvability": "Improvability ↓",
    "rel_error_vs_all_features": "Error vs. all features ↓",
    "median_fs_time_s": "Median selection time (s)",
    "fallback_rate": "Random fallback rate",
    "n_comparisons": "#Comparisons",
    "budget_index": "Budget",
    "model": "Model",
    "problem_type": "Problem type",
    "stability": "Stability ↑",
    "validity": "Validity ↑",
    "noise": "Noise level",
}

ABOUT = """
## How the leaderboard is computed

Each feature selection method keeps a budgeted number of features of a dataset; a downstream
model is then trained on the selected features only. Methods are compared within a *comparison
group*: one dataset split, one downstream model configuration and one budget.

- **Avg. rank**: mean rank of the method's downstream error within the groups (1 is best).
- **Win rate**: share of pairwise comparisons won against the other methods (ties count half).
- **Improvability**: mean relative gap `(error - best error) / error` to the best method of the
  group (0 is best).
- **Error vs. all features**: mean ratio of the error to the same model trained on all features.
  Values below 1 mean the selection improved the model.
- **Random fallback rate**: share of runs where the method could not fill its budget in time and
  random features were added.
- **Stability**: agreement of the selected features across bootstrap resamples
  (Nogueira, Sechidis & Brown, JMLR 2018; 1 means identical selections).
- **Validity**: share of selected features that are original features when synthetic noise
  features are added to the data.

The budget index `i` of a total budget `b` keeps `ceil(d * (i+1)/(b+1))` of the `d` features when
`d <= 100`, and `ceil(d ** ((i+1)/(b+1)))` otherwise.

## Links

- Code: [github.com/schaeferbasti/SelectArena](https://github.com/schaeferbasti/SelectArena)
- Paper: [arXiv:2610.13644](https://arxiv.org/abs/2610.13644)
- Built on [TabArena](https://github.com/autogluon/tabarena)

To add a method, implement it as described in the repository's README and open a pull request.
"""


def _read(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _meta() -> dict:
    path = DATA_DIR / "meta.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _display(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: COLUMN_NAMES.get(c, c) for c in df.columns})


def _header() -> str:
    meta = _meta()
    if not meta:
        return "# SelectArena\n\nNo results have been published yet."
    models = ", ".join(meta.get("models", []))
    return (
        "# SelectArena: benchmarking feature selection on tabular data\n\n"
        f"{meta.get('n_methods', '?')} methods on {meta.get('n_datasets', '?')} datasets "
        f"({meta.get('n_results', '?')} runs; downstream models: {models}). "
        f"Last update: {meta.get('updated', '?')}."
    )


def _filter_choices(view: str) -> list[str]:
    name, column = VIEWS[view]
    df = _read(name)
    if column is None or df.empty:
        return ["All"]
    return ["All", *[str(v) for v in df[column].dropna().unique()]]


def leaderboard(view: str, value: str) -> pd.DataFrame:
    name, column = VIEWS[view]
    df = _read(name)
    if df.empty:
        return pd.DataFrame({"Method": [], "Avg. rank ↓": []})
    if column is not None and value not in (None, "All"):
        df = df[df[column].astype(str) == value]
    return _display(df)


def update_filter(view: str):
    choices = _filter_choices(view)
    return gr.update(choices=choices, value="All", visible=VIEWS[view][1] is not None)


def rank_plot_data() -> pd.DataFrame:
    df = _read("overall")
    return df[["method", "avg_rank"]] if not df.empty else pd.DataFrame({"method": [], "avg_rank": []})


with gr.Blocks(title="SelectArena Leaderboard", theme=gr.themes.Soft()) as demo:
    gr.Markdown(_header())
    with gr.Tabs():
        with gr.Tab("Leaderboard"):
            with gr.Row():
                view = gr.Dropdown(list(VIEWS), value="Overall", label="View")
                value = gr.Dropdown(["All"], value="All", label="Filter", visible=False)
            table = gr.Dataframe(value=leaderboard("Overall", "All"), interactive=False, wrap=True)
            gr.BarPlot(
                value=rank_plot_data(),
                x="method",
                y="avg_rank",
                title="Average rank (lower is better)",
                sort="y",
                height=360,
            )
            view.change(update_filter, inputs=view, outputs=value).then(
                leaderboard, inputs=[view, value], outputs=table
            )
            value.change(leaderboard, inputs=[view, value], outputs=table)
        with gr.Tab("Per dataset"):
            gr.Markdown("Average rank of each method on each dataset (over splits, models and budgets).")
            gr.Dataframe(value=_display(_read("per_dataset")), interactive=False)
        with gr.Tab("Stability & validity"):
            stability, validity = _read("stability"), _read("validity")
            gr.Markdown("### Stability over bootstrap resamples")
            if stability.empty:
                gr.Markdown("No stability results published yet.")
            else:
                gr.Dataframe(value=_display(stability), interactive=False)
            gr.Markdown("### Validity under injected noise features")
            if validity.empty:
                gr.Markdown("No validity results published yet.")
            else:
                gr.Dataframe(value=_display(validity), interactive=False)
        with gr.Tab("About"):
            gr.Markdown(ABOUT)


if __name__ == "__main__":
    demo.launch()
