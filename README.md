# Pumpkinpipe
This is a library that combines different tools for computer vision in a way that is simplified and student facing.

```
pip install pumpkinpipe
```

See the documentation here:

https://smugpumpkins.github.io/Pumpkinpipe/


Rebuild code:

*Remember to update the version number in `pyproject.toml` first!

```
py -m pip install --upgrade build twine
```
```
py -m build
```

Reupload code to pyPI:
```
py -m twine upload --verbose dist/*
```