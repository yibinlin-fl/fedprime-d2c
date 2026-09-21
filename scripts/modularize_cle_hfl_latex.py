from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def slice_between(text: str, start: str, end: str) -> str:
    start_index = text.index(start)
    end_index = text.index(end, start_index)
    return text[start_index:end_index]


def extract_table(text: str, caption: str, environment: str) -> tuple[str, str]:
    caption_index = text.index(caption)
    begin = rf"\begin{{{environment}}}"
    end = rf"\end{{{environment}}}"
    start_index = text.rfind(begin, 0, caption_index)
    if start_index < 0:
        raise ValueError(f"Missing {begin} before {caption}")
    end_index = text.index(end, caption_index) + len(end)
    block = text[start_index:end_index]
    return block, text[:start_index] + "{replacement}" + text[end_index:]


def canonical(text: str) -> str:
    return re.sub(r"\s+", "", text)


def expand_inputs(
    path: Path,
    base_dir: Path | None = None,
    seen: set[Path] | None = None,
) -> str:
    resolved = path.resolve()
    base_dir = path.parent if base_dir is None else base_dir
    seen = set() if seen is None else seen
    if resolved in seen:
        raise RuntimeError(f"Recursive LaTeX input: {resolved}")
    seen.add(resolved)
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(r"\\input\{([^}]+)\}")

    def replace(match: re.Match[str]) -> str:
        relative = match.group(1)
        input_path = base_dir / relative
        if input_path.suffix == "":
            input_path = input_path.with_suffix(".tex")
        return expand_inputs(input_path, base_dir, seen.copy())

    return pattern.sub(replace, text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mechanically modularize CLE-HFL LaTeX V0.3.")
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "deliverables" / "cle_hfl_latex_v0_3_20260916",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "deliverables" / "cle_hfl_latex_v0_4_20260921",
    )
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists() and not args.overwrite_existing:
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    original = (source / "main.tex").read_text(encoding="utf-8")
    markers = {
        "abstract": r"\begin{abstract}",
        "introduction": r"\section{Introduction}",
        "related_work": r"\section{Related Work}",
        "method": r"\section{Problem Setup}",
        "experiments": r"\section{Experiments}",
        "discussion": r"\section{Discussion and Limitations}",
        "conclusion": r"\section{Conclusion}",
        "appendix": r"\appendix",
        "bibliography": r"\bibliographystyle{plainnat}",
    }
    for marker in markers.values():
        if marker not in original:
            raise ValueError(f"Missing required marker: {marker}")

    prefix = original[: original.index(markers["abstract"])]
    sections = {
        "abstract": slice_between(original, markers["abstract"], markers["introduction"]),
        "introduction": slice_between(original, markers["introduction"], markers["related_work"]),
        "related_work": slice_between(original, markers["related_work"], markers["method"]),
        "method": slice_between(original, markers["method"], markers["experiments"]),
        "experiments": slice_between(original, markers["experiments"], markers["discussion"]),
        "discussion": slice_between(original, markers["discussion"], markers["conclusion"]),
        "conclusion": slice_between(original, markers["conclusion"], markers["appendix"]),
        "appendix": slice_between(original, markers["appendix"], markers["bibliography"]),
    }
    suffix = original[original.index(markers["bibliography"]):]

    table_specs = [
        (
            "heldout_map2",
            r"\caption{Held-out map2 matched control comparison.}",
            "table*",
            "experiments",
        ),
        (
            "cross_setting",
            r"\caption{Cross-setting pooled DSA mitigation summary.}",
            "table*",
            "experiments",
        ),
        (
            "oracle_granularity",
            r"\caption{Oracle correspondence and granularity audit.}",
            "table",
            "appendix",
        ),
    ]
    tables: dict[str, str] = {}
    for name, caption, environment, section_name in table_specs:
        block, templated = extract_table(sections[section_name], caption, environment)
        tables[name] = block + "\n"
        sections[section_name] = templated.replace(
            "{replacement}", rf"\input{{tables/{name}}}"
        )

    (output / "sections").mkdir(parents=True, exist_ok=True)
    (output / "tables").mkdir(exist_ok=True)
    shutil.copytree(source / "figures", output / "figures", dirs_exist_ok=True)
    shutil.copy2(source / "references.bib", output / "references.bib")

    for name, content in sections.items():
        (output / "sections" / f"{name}.tex").write_text(
            content.rstrip() + "\n", encoding="utf-8", newline="\n"
        )
    for name, content in tables.items():
        (output / "tables" / f"{name}.tex").write_text(
            content.rstrip() + "\n", encoding="utf-8", newline="\n"
        )

    inputs = "\n".join(
        rf"\input{{sections/{name}}}"
        for name in (
            "abstract",
            "introduction",
            "related_work",
            "method",
            "experiments",
            "discussion",
            "conclusion",
            "appendix",
        )
    )
    modular_main = prefix + inputs + "\n\n" + suffix
    (output / "main.tex").write_text(modular_main, encoding="utf-8", newline="\n")

    expanded = expand_inputs(output / "main.tex")
    if canonical(expanded) != canonical(original):
        raise RuntimeError("Expanded V0.4 content differs from V0.3 after whitespace normalization")

    print(f"Created modular paper: {output}")
    print("CONTENT_EQUIVALENCE=PASS")


if __name__ == "__main__":
    main()
