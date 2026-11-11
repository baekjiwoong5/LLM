# -*- coding: utf-8 -*-
"""
웹 챗봇(index.html)에 학습 결과를 넣는 스크립트

  python export_web.py        # data/ 폴더의 모든 데이터를 학습해 index.html 안의 DATA를 갱신

index.html은 여기서 넣은 계수·가중치만으로 브라우저에서 답하므로 서버가 필요 없습니다.
data/ 폴더의 데이터를 바꾸거나 추가했다면 이 스크립트를 다시 실행하세요.
"""
import json
import re
from pathlib import Path

import numpy as np

from chat import Dataset, default_paths

ROOT = Path(__file__).resolve().parent
HTML_PATH = ROOT / "index.html"
KINDS = {"선형회귀": "lin", "비선형(2차 다항)": "poly", "비선형(거듭제곱)": "pow",
         "비선형(지수)": "exp", "인공신경망": "mlp"}


def _list(a, digits=None):
    a = np.asarray(a, dtype=float)
    return (a.round(digits) if digits is not None else a).tolist()


def export_model(name, m, names):
    out = {"name": name, "kind": KINDS[name], "equation": m.equation(names)}
    if out["kind"] == "lin":
        out.update(b0=m.b0, b=_list(m.b))
    elif out["kind"] == "poly":
        out.update(c0=m.c0, c1=_list(m.c1), c2=_list(m.c2))
    elif out["kind"] in ("pow", "exp"):
        out.update(a=m.a, b=_list(m.b))
    else:
        out.update(W=[_list(w, 6) for w in m.W], B=[_list(b.ravel(), 6) for b in m.B],
                   xm=_list(m.xm), xs=_list(m.xs), ym=m.ym, ys=m.ys)
    return out


def export_dataset(ds):
    cols = ds.features + [ds.target]
    names = [ds.label[c] for c in ds.features]
    first = ds.features[0]
    return {
        "title": ds.title, "file": ds.path.name, "keywords": ds.keywords, "unitWord": ds.unit_word,
        "n": len(ds.df), "features": ds.features, "target": ds.target,
        "label": ds.label, "unit": ds.unit, "aliases": [list(a) for a in ds.aliases],
        "stats": {c: {"mean": float(ds.df[c].mean()), "std": float(ds.df[c].std()),
                      "min": float(ds.df[c].min()), "max": float(ds.df[c].max())} for c in cols},
        "fill": ds.fill,
        "models": [export_model(n, m, names) for n, m in ds.models.items()],
        "metrics": ds.metrics, "metricOrder": list(ds.metrics), "best": ds.best,
        "sizes": list(ds.sizes), "skipped": ds.skipped,
        "example": f"{ds.label[first]} {round(ds.df[first].median()):g}일 때 {ds.label[ds.target]} 예측해줘",
    }


def main():
    data = [export_dataset(Dataset(p)) for p in default_paths()]
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = HTML_PATH.read_text(encoding="utf-8")
    block = "/*DATA_START*/const DATA = " + payload + ";/*DATA_END*/"
    new, n = re.subn(r"/\*DATA_START\*/.*?/\*DATA_END\*/", lambda _: block, html, count=1, flags=re.S)
    if n != 1:
        raise SystemExit("index.html에서 DATA 자리를 찾지 못했습니다")
    HTML_PATH.write_text(new, encoding="utf-8")
    print(f"index.html 갱신: {', '.join(d['title'] for d in data)} ({len(payload) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
