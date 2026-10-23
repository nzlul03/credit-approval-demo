"""Inject model.json into template.html and write index.html (the file you deploy).
Run:  python3 landing-page/build.py"""
import pathlib
here = pathlib.Path(__file__).resolve().parent
model = (here / "model.json").read_text().strip()
html = (here / "template.html").read_text().replace("/*MODEL*/", model)
(here / "index.html").write_text(html)
print("wrote index.html", len(html), "bytes")
