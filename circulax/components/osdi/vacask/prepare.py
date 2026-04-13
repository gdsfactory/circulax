import os

def run(settings):
    settings["pre_options"] = [ "--skip-embed", "--skip-postprocess", "--no-output" ]
    return True
