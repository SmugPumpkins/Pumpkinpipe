# from importlib import resources
# from contextlib import contextmanager
#
# @contextmanager
# def get_model_path(filename: str):
#     with resources.as_file(
#         resources.files("pumpkinpipe.models") / filename
#     ) as path:
#         yield str(path)


from importlib import resources
from contextlib import contextmanager
import sys
import os

@contextmanager
def get_model_path(filename: str):
    # PyInstaller onefile runtime
    if hasattr(sys, "_MEIPASS"):
        path = os.path.join(sys._MEIPASS, "pumpkinpipe", "models", filename)
        yield path
        return

    # Normal Python (your current logic)
    with resources.as_file(
        resources.files("pumpkinpipe.models") / filename
    ) as path:
        yield str(path)