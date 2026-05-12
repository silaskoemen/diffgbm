from __future__ import annotations

import argparse
import copy
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    variant_names = parse_variant_names(args.variants)
    if variant_names is not None:
        config = filter_variants(config, variant_names)
    output_format = resolve_output_format(args.output, args.output_format)
    output_path = Path(args.output) if args.output else default_output_path(config_path, variant_names, output_format)
    from benchmarks.harness import run_benchmark  # noqa: PLC0415

    run_benchmark(config=config, output_path=output_path, output_format=output_format)
    print(f"Wrote benchmark results to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Treeffuser development benchmarks.")
    parser.add_argument("--config", required=True, help="Path to a benchmark YAML config.")
    parser.add_argument("--output", default=None, help="Optional output path. The suffix can be .jsonl or .csv.")
    parser.add_argument(
        "--output-format",
        choices=["jsonl", "csv"],
        default="jsonl",
        help="Benchmark output format. Ignored when --output has a .jsonl or .csv suffix.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Optional variant names to run. Accepts space-separated names or comma-separated groups.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml  # noqa: PLC0415
    except ModuleNotFoundError:
        return _load_simple_yaml(path)

    with path.open() as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Config {path} must contain a mapping at the top level.")
    return config


def parse_variant_names(raw_variants: list[str] | None) -> list[str] | None:
    if raw_variants is None:
        return None
    variants = []
    for item in raw_variants:
        variants.extend(part.strip() for part in item.split(","))
    variants = [variant for variant in variants if variant]
    if not variants:
        return None
    return variants


def filter_variants(config: dict[str, Any], variant_names: list[str]) -> dict[str, Any]:
    config = copy.deepcopy(config)
    variants_by_name = {variant["name"]: variant for variant in config["variants"]}
    missing = [name for name in variant_names if name not in variants_by_name]
    if missing:
        available = ", ".join(sorted(variants_by_name))
        raise ValueError(f"Unknown benchmark variants {missing}. Available: {available}")
    config["variants"] = [variants_by_name[name] for name in variant_names]
    return config


def resolve_output_format(output: str | None, requested_format: str) -> str:
    if output is None:
        return requested_format
    suffix = Path(output).suffix
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".csv":
        return "csv"
    return requested_format


def default_output_path(config_path: Path, variant_names: list[str] | None, output_format: str) -> Path:
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    variant_slug = "all" if variant_names is None else "vs".join(_slugify(name) for name in variant_names)
    if len(variant_slug) > 160:
        variant_slug = f"{variant_slug[:140]}_{abs(hash(variant_slug))}"
    return Path("benchmarks") / "results" / "raw" / f"{config_path.stem}__{variant_slug}_{timestamp}.{output_format}"


def _slugify(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_")


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    lines = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if line:
            lines.append((len(line) - len(line.lstrip(" ")), line.strip()))

    if not lines:
        return {}

    def parse_block(index: int, indent: int):
        if index >= len(lines):
            return {}, index
        if lines[index][1].startswith("- "):
            return parse_list(index, indent)
        return parse_dict(index, indent)

    def parse_dict(index: int, indent: int):
        result = {}
        while index < len(lines):
            current_indent, text = lines[index]
            if current_indent < indent:
                break
            if current_indent > indent:
                raise ValueError(f"Unexpected indentation in {path}: {text}")
            if text.startswith("- "):
                break

            key, sep, value_text = text.partition(":")
            if sep == "":
                raise ValueError(f"Expected key/value pair in {path}: {text}")
            index += 1
            value_text = value_text.strip()
            if value_text:
                result[key] = parse_scalar(value_text)
            else:
                result[key], index = parse_block(index, indent + 2)
        return result, index

    def parse_list(index: int, indent: int):
        result = []
        while index < len(lines):
            current_indent, text = lines[index]
            if current_indent < indent:
                break
            if current_indent > indent:
                raise ValueError(f"Unexpected indentation in {path}: {text}")
            if not text.startswith("- "):
                break

            item_text = text[2:].strip()
            index += 1
            if not item_text:
                item, index = parse_block(index, indent + 2)
            elif ":" in item_text:
                key, _, value_text = item_text.partition(":")
                item = {key: parse_scalar(value_text.strip())}
                if index < len(lines) and lines[index][0] > indent:
                    continuation, index = parse_block(index, indent + 2)
                    if not isinstance(continuation, dict):
                        raise ValueError(f"Expected mapping continuation in {path}: {text}")
                    item.update(continuation)
            else:
                item = parse_scalar(item_text)
            result.append(item)
        return result, index

    def parse_scalar(text: str):
        if text in {"true", "True"}:
            return True
        if text in {"false", "False"}:
            return False
        if text in {"null", "None", "~"}:
            return None
        if text.startswith("[") and text.endswith("]"):
            inner = text[1:-1].strip()
            if not inner:
                return []
            return [parse_scalar(part.strip()) for part in inner.split(",")]
        try:
            return int(text)
        except ValueError:
            pass
        try:
            return float(text)
        except ValueError:
            pass
        return text.strip("\"'")

    config, next_index = parse_block(0, lines[0][0])
    if next_index != len(lines):
        raise ValueError(f"Could not parse all of {path}.")
    if not isinstance(config, dict):
        raise ValueError(f"Config {path} must contain a mapping at the top level.")
    return config


if __name__ == "__main__":
    main()
