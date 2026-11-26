# -*- coding: utf-8 -*-
"""
웹 챗봇(index.html)에 학습 결과와 질문 이해 규칙을 넣는 스크립트

  python export_web.py        # data/ 폴더의 모든 데이터를 학습해 index.html 안의 DATA와 NLU를 갱신

index.html은 여기서 넣은 계수·가중치만으로 브라우저에서 답하므로 서버가 필요 없습니다.
열마다 그 열을 맞히는 모델 묶음을 모두 넣으므로 반대 방향 질문에도 답합니다.
data/ 폴더의 데이터나 nlu.json의 단어를 바꿨다면 이 스크립트를 다시 실행하세요.
"""
import json
import re
from pathlib import Path

import numpy as np

from chat import Dataset, default_paths
from nlu import KNOW, NLU

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
        out.update(W=[_list(w, 8) for w in m.W], B=[_list(b.ravel(), 8) for b in m.B],
                   xm=_list(m.xm), xs=_list(m.xs), ym=m.ym, ys=m.ys,
                   history={"train": _list(m.history["train"], 6), "val": _list(m.history["val"], 6)},
                   epochs=m.epochs, bestEpoch=m.best_epoch)
    return out


def export_dataset(ds):
    bundles = {}
    for t in ds.columns:
        b = ds.bundle(t)
        names = [ds.label[c] for c in b["features"]]
        bundles[t] = {"features": b["features"],
                      "models": [export_model(n, m, names) for n, m in b["models"].items()],
                      "metrics": b["metrics"], "metricOrder": list(b["metrics"]), "best": b["best"],
                      "sizes": list(b["sizes"]), "skipped": b["skipped"],
                      "testY": b["test_y"], "testPred": b["test_pred"]}
    return {"spec": ds.spec, "file": ds.path.name, "n": len(ds.df), "unitWord": ds.unit_word,
            "dropped": ds.dropped, "nMissing": ds.n_missing, "corr": ds.corr, "cov": ds.cov,
            "bmiDist": ds.bmi_dist, "bundles": bundles, "rows": ds.df.to_numpy().tolist()}


def _js(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def _put(html, name, payload):
    block = f"/*{name}_START*/const {name} = {payload};/*{name}_END*/"
    new, n = re.subn(rf"/\*{name}_START\*/.*?/\*{name}_END\*/", lambda _: block, html, count=1, flags=re.S)
    if n != 1:
        raise SystemExit(f"index.html에서 {name} 자리를 찾지 못했습니다")
    return new


def main():
    data = []
    for p in default_paths():
        ds = Dataset(p)
        data.append(export_dataset(ds))
        print(f"  {ds.title}: 열 {len(ds.columns)}개 × 모델 5종 학습")
    html = HTML_PATH.read_text(encoding="utf-8")
    html = _put(html, "DATA", _js(data))
    nlu = {k: v for k, v in NLU.items() if not k.startswith("_")}
    nlu["knowledge"] = {k: v for k, v in KNOW.items() if not k.startswith("_")}
    html = _put(html, "NLU", _js(nlu))
    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"index.html 갱신 ({len(html) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
