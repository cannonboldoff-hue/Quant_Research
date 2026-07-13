"""Re-run static analysis over notebooks/ to regenerate the catalog.
Usage: python scripts/analyze_notebooks.py <notebooks_dir>
"""
import json, glob, os, re, sys
from collections import Counter

def main(root):
    files = glob.glob(os.path.join(root, "**", "*.ipynb"), recursive=True)
    fn = Counter()
    for f in files:
        nb = json.load(open(f, encoding="utf-8"))
        for c in nb.get("cells", []):
            if c.get("cell_type") == "code":
                for m in re.findall(r"^\s*def\s+(\w+)", "".join(c.get("source", [])), re.M):
                    fn[m] += 1
    print(f"{len(files)} notebooks; top functions:")
    for k, c in fn.most_common(25):
        print(f"  {k}: {c}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "notebooks")
