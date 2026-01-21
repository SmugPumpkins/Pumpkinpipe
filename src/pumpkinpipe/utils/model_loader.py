from importlib import resources
from contextlib import contextmanager

@contextmanager
def get_model_path(filename: str):
    with resources.as_file(
        resources.files("pumpkinpipe.models") / filename
    ) as path:
        yield str(path)
