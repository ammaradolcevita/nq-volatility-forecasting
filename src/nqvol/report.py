"""Helpers to save result tables as CSV and LaTeX.

The .tex files need \\usepackage{booktabs} and \\usepackage{graphicx}.
"""
import re

import pandas as pd

from . import config as C

TABLES = C.RESULTS_DIR / "tables"
PREDICTIONS = C.RESULTS_DIR / "predictions"


def save_table(df: pd.DataFrame, name: str, caption: str = "", label: str = "",
               float_format="%.4f", index=False, notes: str = ""):
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / f"{name}.csv", index=index)
    if name.endswith("_ratio"):
        caption = f"{caption} (scale-free specification)" if caption else caption
        label = f"{label}_ratio" if label else label
    tex = df.to_latex(index=index, float_format=float_format, escape=True, na_rep="--",
                      caption=caption or None, label=label or None, position="htbp")
    # fit wide tables to the text width
    tex = tex.replace("\\begin{table}[htbp]", "\\begin{table}[htbp]\n\\centering\\small")
    tex = tex.replace("\\begin{tabular}", "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}", 1)
    tex = tex.replace("\\end{tabular}", "\\end{tabular}%\n}", 1)
    if notes:
        notes = re.sub(r"(?<!\\)_", r"\\_", notes)
        tex = tex.replace("\\end{tabular}%\n}",
                          "\\end{tabular}%\n}\n\\par\\vspace{2pt}\\footnotesize\\textit{Notes:} " + notes)
    (TABLES / f"{name}.tex").write_text(tex)
    return TABLES / f"{name}.csv"


def save_predictions(df: pd.DataFrame, name: str):
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    path = PREDICTIONS / f"{name}.csv"
    df.to_csv(path, index=False)
    return path
