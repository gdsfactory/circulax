# Jupyter traitlets config file — loaded by `jupyter nbconvert --config docs/nbconvert_config.py`.
# `c` is a Config object automatically injected by Jupyter before this file is executed;
# assigning to it is equivalent to passing the corresponding command-line flags.
import glob

# Discover all example notebooks; equivalent to listing them on the command line.
c.NbConvertApp.notebooks = sorted(glob.glob("examples/**/*.ipynb", recursive=True))
# Flat output: all .md files and their _files/ image dirs land in docs/examples/.
c.FilesWriter.build_directory = "docs/examples"
# Strip hidden style/setup cells (tagged remove-cell) and schematic cells (tagged remove-input).
c.TagRemovePreprocessor.enabled = True
c.TagRemovePreprocessor.remove_cell_tags = ["remove-cell"]
c.TagRemovePreprocessor.remove_input_tags = ["remove-input", "hide-input"]
