"""Run directly with python -I to smoke-test an installed wheel outside pytest."""

import subprocess
import sys
import unittest


class NetlistIOInstallTest(unittest.TestCase):
    def test_sources_without_parser(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                """
import importlib.abc
import math
import sys
from pathlib import Path

class NoNetlistParser(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'netlist_parser':
            raise ModuleNotFoundError('parser deliberately unavailable', name=fullname)

sys.meta_path.insert(0, NoNetlistParser())
from circulax.components.electronic import WaveformVoltageSource, WaveformCurrentSource
from circulax.netlist_io import parse_source, parse_waveform
from circulax.netlist_io.expressions import evaluate_source
from circulax.netlist_io.syntax import parse_file

assert math.isclose(evaluate_source('width * 1u', {'width': 2}), 2e-6)
assert evaluate_source('5M', {}, 'spice') == 5e-3
assert evaluate_source('5M', {}, 'spectre') == 5e6
assert parse_source('DC 0.5 SIN(0.9 1 1k)')['freq'] == 1000
assert parse_waveform('PWL(0 0 1m 1)')['pwl_v'] == (0.0, 1.0)
settings = parse_source('PULSE(0 1 1n 1n 1n 10n 20n)')
WaveformVoltageSource(**settings)
WaveformCurrentSource(**settings)
assert 'netlist_parser' not in sys.modules
try:
    parse_file(Path('unused.sp'))
except ImportError as exc:
    assert 'SPICE/Spectre file parsing requires NetlistParse' in str(exc)
else:
    raise AssertionError('missing parser should fail only when parsing a file')
""",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)  # noqa: PT009 -- also run without pytest


if __name__ == "__main__":
    unittest.main()
