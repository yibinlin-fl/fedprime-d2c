"""Convert the fact-audited CLE-HFL Markdown manuscript into portable LaTeX."""

from __future__ import annotations

import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "deliverables" / "cle_hfl_paper_draft_v0_4_20260915" / "CLE_HFL_PAPER_DRAFT_V0_4.md"
SOURCE_BIB = ROOT / "deliverables" / "cle_hfl_paper_draft_v0_4_20260915" / "references.bib"
OUT = ROOT / "deliverables" / "cle_hfl_latex_v0_1_20260915"


PREAMBLE = r"""\documentclass[10pt,twocolumn]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage[margin=0.72in,columnsep=0.24in]{geometry}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{tabularx}
\usepackage{xcolor}
\usepackage[numbers,sort&compress]{natbib}
\usepackage[hidelinks]{hyperref}
\usepackage{enumitem}
\setlist{nosep,leftmargin=*}
\newtheorem{proposition}{Proposition}
\newtheorem{theorem}{Theorem}
\definecolor{evidencered}{HTML}{A32020}
\newcommand{\evidenceneeded}[1]{\textcolor{evidencered}{\textbf{[EVIDENCE NEEDED: #1]}}}

\title{When Corruption Becomes a Label: Diagnosing and Mitigating\\
Class--Corruption Shortcuts in Model-Heterogeneous Federated Learning}
\author{Anonymous Authors}
\date{}

\begin{document}
\maketitle
"""


def normalize_unicode(value: str) -> str:
    return (
        value.replace("–", "--")
        .replace("—", "---")
        .replace("‑", "-")
        .replace("→", r"$\rightarrow$")
        .replace("↑", r"$\uparrow$")
        .replace("↓", r"$\downarrow$")
        .replace("≤", r"$\leq$")
        .replace("≥", r"$\geq$")
        .replace("×", r"$\times$")
        .replace("−", "-")
        .replace("“", "``")
        .replace("”", "''")
        .replace("’", "'")
    )


def escape_plain(value: str) -> str:
    value = value.replace("\\&", "ZZESCAPEDAMPZZ")
    value = value.replace("&", r"\&")
    value = value.replace("%", r"\%")
    value = value.replace("#", r"\#")
    value = value.replace("_", r"\_")
    value = value.replace("ZZESCAPEDAMPZZ", r"\&")
    return value


def inline(value: str) -> str:
    value = normalize_unicode(value.strip())
    protected: dict[str, str] = {}

    def hold(content: str) -> str:
        token = f"ZZPROTECTED{len(protected)}ZZ"
        protected[token] = content
        return token

    # Bold is processed recursively so inline math and citations inside it remain valid.
    while "**" in value:
        match = re.search(r"\*\*(.+?)\*\*", value)
        if not match:
            break
        value = value[: match.start()] + hold(r"\textbf{" + inline(match.group(1)) + "}") + value[match.end() :]

    while True:
        match = re.search(r"(?<!\*)\*([^*]+)\*(?!\*)", value)
        if not match:
            break
        value = value[: match.start()] + hold(r"\emph{" + inline(match.group(1)) + "}") + value[match.end() :]

    value = re.sub(r"`([^`]+)`", lambda m: hold(r"\texttt{" + escape_plain(m.group(1)) + "}"), value)
    value = re.sub(r"\\\(.+?\\\)", lambda m: hold(m.group(0)), value)
    value = re.sub(r"\\citep\{[^}]+\}", lambda m: hold(m.group(0)), value)
    value = re.sub(r"\\(?:ref|autoref|eqref)\{[^}]+\}", lambda m: hold(m.group(0)), value)
    value = escape_plain(value)
    for token, content in protected.items():
        value = value.replace(token, content)
    return value


def table_latex(rows: list[str], caption: str | None, index: int) -> str:
    cells = [[cell.strip() for cell in row.strip().strip("|").split("|")] for row in rows]
    data = [cells[0]] + cells[2:]
    ncols = len(data[0])
    spec = "l" + "r" * (ncols - 1)
    label = f"tab:auto{index}"
    caption_text = caption or f"Experimental summary {index}."
    caption_text = re.sub(r"^Table\s+\d+\.\s*", "", caption_text)
    out = [r"\begin{table*}[t]", r"\centering", r"\small", rf"\caption{{{inline(caption_text)}}}", rf"\label{{{label}}}"]
    out += [r"\resizebox{\textwidth}{!}{%", rf"\begin{{tabular}}{{{spec}}}", r"\toprule"]
    for ridx, row in enumerate(data):
        out.append(" & ".join(inline(cell) for cell in row) + r" \\")
        if ridx == 0:
            out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"]
    return "\n".join(out)


def figure_latex(number: int) -> str:
    if number == 1:
        stem = "fig1_cle_hfl_evidence_chain"
        caption = (
            "CLE-HFL evidence chain and information boundary. Training-time PEW+BER uses only a coarse "
            "family proxy, whereas operator metadata and the binding map are reserved for sealed, reporting-only DSA evaluation."
        )
        label = "fig:evidence-chain"
    else:
        stem = "fig2_dsa_proxy_boundary"
        caption = (
            "Paired directional diagnosis and proxy-identifiability boundary. DSA compares bound-class probability mass across "
            "operators for the same source; identical observable label--proxy evidence can correspond to latent worlds with different true dependence."
        )
        label = "fig:dsa-proxy"
    return "\n".join(
        [
            r"\begin{figure*}[t]",
            r"\centering",
            rf"\includegraphics[width=0.98\textwidth]{{figures/{stem}.pdf}}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{figure*}",
        ]
    )


def convert() -> str:
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    out = [PREAMBLE]
    paragraph: list[str] = []
    pending_caption: str | None = None
    table_index = 0
    in_math = False
    in_abstract = False
    appendix_started = False
    list_mode: str | None = None

    def flush_paragraph():
        nonlocal paragraph
        if paragraph:
            out.append(inline(" ".join(part.strip() for part in paragraph)))
            out.append("")
            paragraph = []

    def close_list():
        nonlocal list_mode
        if list_mode:
            out.append(rf"\end{{{list_mode}}}")
            out.append("")
            list_mode = None

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if i < 5 and (stripped.startswith("# Version") or stripped.startswith("> **Internal status") or stripped.startswith("# When Corruption")):
            i += 1
            continue

        if in_math:
            out.append(normalize_unicode(line))
            if stripped == r"\]":
                in_math = False
                out.append("")
            i += 1
            continue

        if stripped == r"\[":
            flush_paragraph()
            close_list()
            out.append(r"\[")
            in_math = True
            i += 1
            continue

        if stripped.startswith("**Table ") and stripped.endswith("**"):
            flush_paragraph()
            pending_caption = stripped[2:-2]
            i += 1
            continue

        if stripped.startswith("|"):
            flush_paragraph()
            close_list()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i])
                i += 1
            table_index += 1
            out.append(table_latex(rows, pending_caption, table_index))
            out.append("")
            pending_caption = None
            continue

        if stripped.startswith("> **Figure 1 placeholder"):
            flush_paragraph()
            close_list()
            out.append(figure_latex(1))
            out.append("")
            i += 2
            continue
        if stripped.startswith("> **Figure 2 placeholder"):
            flush_paragraph()
            close_list()
            out.append(figure_latex(2))
            out.append("")
            i += 2
            continue

        ordered = re.match(r"^\d+\.\s+(.+)$", stripped)
        bullet = re.match(r"^-\s+(.+)$", stripped)
        if ordered or bullet:
            flush_paragraph()
            desired = "enumerate" if ordered else "itemize"
            if list_mode != desired:
                close_list()
                out.append(rf"\begin{{{desired}}}")
                list_mode = desired
            out.append(r"\item " + inline((ordered or bullet).group(1)))
            i += 1
            continue
        elif list_mode:
            close_list()

        if stripped.startswith("# References"):
            flush_paragraph()
            if in_abstract:
                out.append(r"\end{abstract}")
                in_abstract = False
            out += [r"\bibliographystyle{plainnat}", r"\bibliography{references}", r"\end{document}"]
            break

        if stripped.startswith("# Appendix "):
            flush_paragraph()
            if not appendix_started:
                out.append(r"\appendix")
                appendix_started = True
            title = re.sub(r"^# Appendix [A-Z]\.\s*", "", stripped)
            out.append(rf"\section{{{inline(title)}}}")
            out.append("")
            i += 1
            continue

        if stripped.startswith("## "):
            flush_paragraph()
            heading = re.sub(r"^##\s+(?:\d+\.\s*)?", "", stripped)
            if heading == "Abstract":
                out.append(r"\begin{abstract}")
                in_abstract = True
            else:
                if in_abstract:
                    out.append(r"\end{abstract}")
                    out.append("")
                    in_abstract = False
                if appendix_started:
                    heading = re.sub(r"^[A-Z]\.\d+\s+", "", heading)
                    out.append(rf"\subsection{{{inline(heading)}}}")
                else:
                    out.append(rf"\section{{{inline(heading)}}}")
            out.append("")
            i += 1
            continue

        if stripped.startswith("### "):
            flush_paragraph()
            heading = re.sub(r"^###\s+(?:\d+\.\d+\s+|[A-Z]\.\d+\s+)?", "", stripped)
            out.append(rf"\subsection{{{inline(heading)}}}")
            out.append("")
            i += 1
            continue

        if stripped in {"", "---"}:
            flush_paragraph()
            i += 1
            continue

        if stripped.startswith("> "):
            paragraph.append(stripped[2:])
        else:
            paragraph.append(stripped)
        i += 1

    return "\n".join(out) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "main.tex").write_text(convert(), encoding="utf-8", newline="\n")
    shutil.copyfile(SOURCE_BIB, OUT / "references.bib")


if __name__ == "__main__":
    main()
