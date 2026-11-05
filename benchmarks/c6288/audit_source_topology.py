"""Compare upstream multiplier source topology after dialect syntax normalization."""

import argparse
import itertools
import json
from pathlib import Path


def normalize(path, ngspice):
    result = []
    for line in path.read_text().splitlines():
        line = line.strip().lower()
        if not line or line.startswith(("*", "//", "ground ")):
            continue
        tokens = line.replace("(", " ").replace(")", " ").split()
        if ngspice:
            if tokens[0].startswith("."):
                tokens[0] = tokens[0][1:]
            elif tokens[0].startswith("x"):
                tokens[0] = tokens[0][1:]
        result.append(tokens)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vacask-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.vacask_root / "benchmark/c6288"
    a = normalize(root / "ngspice/multiplier.inc", True)
    b = normalize(root / "vacask/multiplier.inc", False)
    differences = [{"line": i, "ngspice": x, "vacask": y} for i, (x, y) in enumerate(itertools.zip_longest(a, b), 1) if x != y]
    result = {
        "scope": "Source gate topology and ordered subcircuit ports after dialect syntax/name-prefix normalization; not an independent audit of the flattened Python netlist or simulator algorithms.",
        "normalized_lines_ngspice": len(a),
        "normalized_lines_vacask": len(b),
        "mismatches": differences,
    }
    Path(__file__).with_name("source-topology-comparison.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
