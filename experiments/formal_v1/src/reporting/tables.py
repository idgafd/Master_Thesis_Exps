"""Regenerate the markdown tables in RESULTS.md from `outputs/_index.csv`.

Looks for lines of the form:
    <!-- AUTOGEN:TABLE name=group_a -->
    ... existing content ...
    <!-- /AUTOGEN:TABLE -->

and rewrites the content between the markers in place. Tables outside
AUTOGEN blocks are left untouched — the thesis narrative stays hand-written.

Supported table names:
    group_a       Group A (causal) full-utterance results
    group_b       Group B (bidirectional) full-utterance results
    chunked       Reset-mode chunked CER for both groups
    carry_state   Carry-state chunked CER for Group A only
    param_counts  Parameter count parity check across all runs
    timing        Average epoch time and peak memory per run

Usage:
    uv run python -m src.reporting.collect  # refresh _index.csv first
    uv run python -m src.reporting.tables   # rewrite RESULTS.md
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Callable

import pandas as pd
from tabulate import tabulate


MARKER_OPEN = re.compile(
    r"<!--\s*AUTOGEN:TABLE\s+name=([a-zA-Z_]+)\s*-->", re.IGNORECASE
)
MARKER_CLOSE = "<!-- /AUTOGEN:TABLE -->"


# ── table renderers ────────────────────────────────────────────────────────

def _fmt(val, fmt="{:.4f}"):
    if val is None or pd.isna(val):
        return "—"
    try:
        return fmt.format(float(val))
    except (TypeError, ValueError):
        return str(val)


def _group_rows(df: pd.DataFrame, group: str) -> pd.DataFrame:
    return df[df["tags"].fillna("").str.contains(group, case=False)]


def table_group_full(df: pd.DataFrame, group: str) -> str:
    sub = _group_rows(df, group).copy()
    if sub.empty:
        return "_No completed runs in this group yet._"
    sub = sub.sort_values("best_dev_cer", na_position="last")
    headers = ["Run", "Backbone", "Params", "Dev CER", "Test CER", "Test WER", "Best epoch"]
    rows = []
    for _, r in sub.iterrows():
        rows.append([
            r["run_id"],
            r["backbone"],
            _fmt(r.get("params_total"), "{:,.0f}"),
            _fmt(r.get("best_dev_cer")),
            _fmt(r.get("test_cer")),
            _fmt(r.get("test_wer")),
            _fmt(r.get("best_dev_epoch"), "{:.0f}"),
        ])
    return tabulate(rows, headers=headers, tablefmt="github", stralign="left")


def table_group_a_full(df):
    return table_group_full(df, "groupA")


def table_group_b_full(df):
    return table_group_full(df, "groupB")


def table_chunked(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No completed runs yet._"
    sub = df.sort_values(["tags", "best_dev_cer"], na_position="last")
    headers = ["Run", "Backbone", "Full Test CER", "2 s reset", "5 s reset", "10 s reset"]
    rows = []
    for _, r in sub.iterrows():
        rows.append([
            r["run_id"],
            r["backbone"],
            _fmt(r.get("test_cer")),
            _fmt(r.get("chunked_2s_reset_cer")),
            _fmt(r.get("chunked_5s_reset_cer")),
            _fmt(r.get("chunked_10s_reset_cer")),
        ])
    return tabulate(rows, headers=headers, tablefmt="github", stralign="left")


def table_carry_state(df: pd.DataFrame) -> str:
    sub = _group_rows(df, "groupA").copy()
    if sub.empty:
        return "_No Group A runs with carry-state eval yet._"
    # Only rows that actually have carry-state numbers
    sub = sub[sub["chunked_2s_carry_cer"].notna() | sub["chunked_5s_carry_cer"].notna()]
    if sub.empty:
        return "_No Group A runs with carry-state eval yet._"
    sub = sub.sort_values("best_dev_cer", na_position="last")
    headers = ["Run", "Backbone", "2 s carry", "5 s carry", "10 s carry"]
    rows = []
    for _, r in sub.iterrows():
        rows.append([
            r["run_id"], r["backbone"],
            _fmt(r.get("chunked_2s_carry_cer")),
            _fmt(r.get("chunked_5s_carry_cer")),
            _fmt(r.get("chunked_10s_carry_cer")),
        ])
    return tabulate(rows, headers=headers, tablefmt="github", stralign="left")


def table_param_counts(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No completed runs yet._"
    sub = df.sort_values("backbone")
    # Reference parameter count = LION, fallback to median
    ref = None
    lion_rows = sub[sub["backbone"] == "lion"]
    if not lion_rows.empty and not pd.isna(lion_rows.iloc[0]["params_total"]):
        ref = float(lion_rows.iloc[0]["params_total"])
    elif "params_total" in sub.columns:
        vals = sub["params_total"].dropna()
        if not vals.empty:
            ref = float(vals.median())

    headers = ["Backbone", "Params total", "Encoder", "vs LION %"]
    rows = []
    for backbone, group in sub.groupby("backbone"):
        r = group.iloc[0]
        pct = (
            f"{(float(r['params_total']) / ref - 1.0) * 100:+.1f}%"
            if ref and not pd.isna(r.get("params_total")) else "—"
        )
        rows.append([
            backbone,
            _fmt(r.get("params_total"), "{:,.0f}"),
            _fmt(r.get("params_encoder"), "{:,.0f}"),
            pct,
        ])
    return tabulate(rows, headers=headers, tablefmt="github", stralign="left")


def table_timing(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No completed runs yet._"
    sub = df.sort_values("avg_epoch_sec")
    headers = ["Run", "Backbone", "Epochs", "Avg epoch", "Total train", "Peak VRAM"]
    rows = []
    for _, r in sub.iterrows():
        rows.append([
            r["run_id"],
            r["backbone"],
            _fmt(r.get("n_epochs"), "{:.0f}"),
            _fmt(r.get("avg_epoch_sec"), "{:.0f} s"),
            _fmt(r.get("total_train_sec"), "{:.0f} s"),
            _fmt(r.get("peak_mem_gb"), "{:.1f} GB"),
        ])
    return tabulate(rows, headers=headers, tablefmt="github", stralign="left")


RENDERERS: dict[str, Callable[[pd.DataFrame], str]] = {
    "group_a": table_group_a_full,
    "group_b": table_group_b_full,
    "chunked": table_chunked,
    "carry_state": table_carry_state,
    "param_counts": table_param_counts,
    "timing": table_timing,
}


# ── file rewriting ────────────────────────────────────────────────────────

def rewrite(results_md: Path, df: pd.DataFrame) -> int:
    """Overwrite AUTOGEN blocks in `results_md`. Returns count of replaced blocks."""
    text = results_md.read_text()
    out: list[str] = []
    replaced = 0

    lines = text.splitlines(keepends=False)
    i = 0
    while i < len(lines):
        line = lines[i]
        match = MARKER_OPEN.search(line)
        if match is None:
            out.append(line)
            i += 1
            continue

        name = match.group(1).lower()
        out.append(line)  # keep the open marker

        # Seek close marker
        j = i + 1
        while j < len(lines) and MARKER_CLOSE not in lines[j]:
            j += 1

        renderer = RENDERERS.get(name)
        if renderer is None:
            # Unknown name — keep original block untouched
            out.extend(lines[i + 1 : j + 1] if j < len(lines) else lines[i + 1 :])
            i = j + 1 if j < len(lines) else len(lines)
            continue

        rendered = renderer(df)
        out.append("")  # blank line for readability
        out.append(rendered)
        out.append("")
        if j < len(lines):
            out.append(lines[j])  # close marker
            i = j + 1
        else:
            out.append(MARKER_CLOSE)
            i = len(lines)
        replaced += 1

    results_md.write_text("\n".join(out) + "\n")
    return replaced


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="outputs/_index.csv")
    parser.add_argument("--results", default="RESULTS.md")
    args = parser.parse_args()

    index_path = Path(args.index)
    results_md = Path(args.results)

    if not index_path.exists():
        raise SystemExit(
            f"{index_path} not found. Run `python -m src.reporting.collect` first."
        )
    if not results_md.exists():
        raise SystemExit(f"{results_md} not found.")

    df = pd.read_csv(index_path)
    replaced = rewrite(results_md, df)
    print(f"Rewrote {replaced} AUTOGEN block(s) in {results_md}")


if __name__ == "__main__":
    main()
