"""Application LOC of the matched W1-W4 implementations.

Rule: non-blank, non-comment lines (Python docstrings count as comments).
Each workload is a complete application; the table reports a Shared column plus, per
framework, the framework-specific lines (framework + config); total = Shared + column.
  common      apps/common/ -- the workload's own data sources, model definitions and
              loading, preprocessing, inference and post-processing.  Identical code is
              used by all five frameworks, so it is reported once per workload in the Shared column.
  framework   apps/<framework>/ -- stage declarations, placement/resource annotations,
              model residency, pipeline composition, concurrent invocation driver,
              service wrappers.
  config      container images and deployment manifests (KubeEdge, Knative).
Excluded uniformly: cluster-level setup every system needs once (node tier labels and
resources, cluster install).
  python tools/app_loc.py [--csv out.csv] [--tex out.tex]
"""
import argparse
import csv
import io
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "apps"
FW = ["hedices", "ray", "dask", "kubeedge", "knative"]
NAMES = {"hedices": "Hedices", "ray": "Ray", "dask": "Dask", "kubeedge": "KubeEdge", "knative": "Knative"}
WL = {"w1": "W1 Recognition", "w2": "W2 Tracking", "w3": "W3 Detection", "w4": "W4 Fusion"}
COMMON = {"w1": ["assets.py", "w1.py"], "w2": ["assets.py", "yolox.py", "w2.py"],
          "w3": ["assets.py", "w3.py"], "w4": ["assets.py", "yolox.py", "w4.py"]}


def py_loc(path: Path) -> int:
    src = path.read_text()
    lines = src.splitlines()
    skip = {i for i, l in enumerate(lines, 1) if not l.strip() or l.strip().startswith("#")}
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.STRING and tok.line.strip().startswith(('"""', "'''")):
            skip.update(range(tok.start[0], tok.end[0] + 1))  # docstrings
    return len(lines) - len(skip)


def cfg_loc(path: Path) -> int:
    return sum(1 for l in path.read_text().splitlines() if l.strip() and not l.strip().startswith("#"))


def files(fw: str, w: str) -> list[Path]:
    d = ROOT / fw
    out = [d / f"{w}.py"] if (d / f"{w}.py").exists() else sorted((d / w).glob("*"))
    return [p for p in out if p.is_file() and p.name != "__init__.py"]


def count(fw: str, w: str) -> dict:
    common = sum(py_loc(ROOT / "common" / f) for f in COMMON[w])
    code = cfg = 0
    for p in files(fw, w):
        if p.suffix == ".py":
            code += py_loc(p)
        else:
            cfg += cfg_loc(p)
    paths = [f"apps/common/{f}" for f in COMMON[w]] + [str(p.relative_to(ROOT.parent)) for p in files(fw, w)]
    return {"common": common, "framework": code, "config": cfg, "total": common + code + cfg, "files": paths}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--tex")
    a = ap.parse_args()
    rows = {(w, fw): count(fw, w) for w in WL for fw in FW}
    print(f"{'workload':16}{'Shared':>8}" + "".join(f"{NAMES[f]:>16}" for f in FW) + "   (framework code + config)")
    for w in WL:
        print(f"{WL[w]:16}{rows[w, FW[0]]['common']:>8}" + "".join(
            f"{r['framework'] + r['config']:>5} ({r['framework']}+{r['config']})".rjust(16)
            for r in (rows[w, f] for f in FW)))
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["workload", "framework", "total_loc", "common_app_loc", "framework_code_loc",
                         "deployment_config_loc", "included_files", "counting_rule"])
            for (w, fw), r in rows.items():
                wr.writerow([w.upper(), NAMES[fw], r["total"], r["common"], r["framework"], r["config"],
                             ";".join(r["files"]),
                             "non-blank non-comment lines; full application incl. common app code; cluster setup excluded"])
    if a.tex:
        body = "\n".join(f"{WL[w]} & {rows[w, FW[0]]['common']} & " + " & ".join(
            (r"\textbf{%d}" if fw == "hedices" else "%d") % (rows[w, fw]["framework"] + rows[w, fw]["config"])
            for fw in FW) + r" \\" for w in WL)
        Path(a.tex).write_text(body + "\n")


if __name__ == "__main__":
    main()
