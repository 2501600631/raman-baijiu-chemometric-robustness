from pathlib import Path
import ast
import hashlib
import io
import re
import tokenize
import zipfile
from xml.etree import ElementTree

HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002fa1f]")
TEXT_SUFFIXES = {".md", ".txt", ".py", ".json", ".csv", ".cff", ".sha256"}


def fail(message):
    raise SystemExit(message)


def docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if ast.get_docstring(node, clean=False) is not None:
                yield node


def verify_release(root):
    root = Path(root)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    for path in root.rglob("*"):
        if any(part in {"__pycache__", ".git"} for part in path.relative_to(root).parts):
            continue
        if HAN.search(path.relative_to(root).as_posix()):
            fail("Chinese characters found in a repository path")
    files = [p for p in files if not any(part in {"__pycache__", ".git"} for part in p.relative_to(root).parts)]
    for path in files:
        rel = path.relative_to(root).as_posix()
        if path.suffix in TEXT_SUFFIXES or path.name in {"LICENSE", ".gitignore"}:
            text = path.read_text(encoding="utf-8-sig")
            if HAN.search(text):
                fail("Chinese characters found in " + rel)
        if path.suffix == ".py":
            text = path.read_text(encoding="utf-8-sig")
            compile(text, rel, "exec")
            tree = ast.parse(text)
            if any(docstrings(tree)):
                fail("Code docstrings found in " + rel)
            if any(t.type == tokenize.COMMENT for t in tokenize.generate_tokens(io.StringIO(text).readline)):
                fail("Code comments found in " + rel)
        if path.suffix in {".xlsx", ".xlsm", ".docx", ".pptx"}:
            with zipfile.ZipFile(path) as package:
                for name in package.namelist():
                    if HAN.search(name):
                        fail("Chinese characters found in an archive member of " + rel)
                    if name.endswith((".xml", ".rels")):
                        payload = package.read(name)
                        text = payload.decode("utf-8-sig")
                        if HAN.search(text):
                            fail("Chinese characters found in " + rel + ":" + name)
                        xml = ElementTree.fromstring(payload)
                        for element in xml.iter():
                            values = [element.text or "", element.tail or ""] + list(element.attrib.values())
                            if any(HAN.search(value) for value in values):
                                fail("Chinese characters found in decoded XML of " + rel + ":" + name)
    expected = {}
    for line in (root / "CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(maxsplit=1)
        rel = rel.lstrip("*")
        if rel in expected or not re.fullmatch(r"[0-9a-f]{64}", digest):
            fail("Invalid or duplicate checksum entry: " + rel)
        relative = Path(rel)
        if relative.is_absolute() or ".." in relative.parts:
            fail("Unsafe checksum path")
        expected[rel] = digest
    actual_files = {p.relative_to(root).as_posix(): p for p in files if p.name != "CHECKSUMS.sha256"}
    if set(actual_files) != set(expected):
        fail("Checksum inventory does not match the repository files")
    for rel, path in actual_files.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected[rel]:
            fail("SHA-256 mismatch: " + rel)
    print("Release integrity passed: " + str(len(expected)) + " SHA-256 entries")
    print("No Chinese characters found in paths, text files, or decoded workbook XML")
    print("All Python files are free of comments and docstrings")
    return {"files": len(files), "checksum_entries": len(expected)}


if __name__ == "__main__":
    verify_release(Path(__file__).resolve().parents[1])
