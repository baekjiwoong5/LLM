# -*- coding: utf-8 -*-
"""
그래프 질문에 답하는 부분: "인공신경망 그래프 보여줘", "학습 곡선", "잔차 그래프", "회귀곡선" …

nlu.py가 정한 plan["chart"] = {"kinds", "x", "target", "xs", "models"}를 받아
  · chart_data()   그래프에 쓸 숫자 (웹 버전 index.html도 같은 규칙으로 계산)
  · summary()      그래프를 글로 요약한 답
  · draw_png()     matplotlib으로 그린 그림 파일 (charts/ 폴더)
을 만듭니다. 그래프 종류는 -1·-2 저장소에서 만들었던 그림을 따릅니다.
  learning   학습 곡선 (에폭별 학습·검증 손실)              ← -2 learning_curve.png
  actual     실제값 vs 예측값 (테스트 데이터, y = x 선)      ← -2 prediction.png
  residual   잔차 그래프 (예측값 vs 실제 − 예측)             ← -1 잔차분석
  fit        산점도 + 모델별 회귀곡선                         ← -1 회귀 그래프
  network    신경망 구조 (입력 → 32 → 16 → 1)
  importance 열별 영향 막대그래프
"""
from pathlib import Path

import numpy as np

from nlu import fixed, fmt

ROOT = Path(__file__).resolve().parent
CHART_DIR = ROOT / "charts"
# 모델 색 (웹 화면과 같은 순서·색. 색맹 구분 검사를 통과한 팔레트)
COLORS = {"선형회귀": "#257de0", "비선형(2차 다항)": "#f47230", "비선형(거듭제곱)": "#00b894",
          "비선형(지수)": "#f2a600", "인공신경망": "#6a4ce2"}
KIND_TITLE = {"learning": "학습 곡선", "actual": "실제값 vs 예측값", "residual": "잔차 그래프",
              "fit": "회귀곡선", "network": "신경망 구조", "importance": "열별 영향"}
GRID = 60


def chart_models(b, chart, kind):
    """그래프에 넣을 모델: 질문에서 고른 모델, 없으면 종류별 기본"""
    names = list(b["models"])
    want = [m for m in chart["models"] if m in names]
    if kind in ("learning", "network"):
        return ["인공신경망"] if "인공신경망" in names else []
    if kind == "actual" and want == ["인공신경망"] and "선형회귀" in names:
        return ["선형회귀", "인공신경망"]               # -2 prediction.png처럼 선형회귀와 나란히
    if want:
        return want
    if kind == "actual":                                  # -2 prediction.png처럼 선형회귀와 신경망
        return [m for m in ("선형회귀", "인공신경망") if m in names] or [b["best"]]
    return names


def x_column(ds, chart):
    """회귀곡선의 가로축: 질문에서 말한 열, 없으면 목표와 상관이 가장 큰 열"""
    t = chart["target"]
    if chart["x"] and chart["x"] != t:
        return chart["x"]
    feats = [c for c in ds.columns if c != t and c not in ds.binary] or [c for c in ds.columns if c != t]
    return max(feats, key=lambda c: (abs(ds.corr[c][t]), -feats.index(c)))


def chart_data(ds, plan, kind, importance_rows=None):
    ch = plan["chart"]
    t = ch["target"]
    b = ds.bundle(t)
    if kind == "learning":
        m = b["models"].get("인공신경망")
        if m is None:
            return None
        h = m.history
        return {"train": h["train"], "val": h["val"], "epochs": m.epochs, "best": m.best_epoch}
    if kind in ("actual", "residual"):
        y = b["test_y"]
        series = []
        for name in chart_models(b, ch, kind):
            p = b["test_pred"][name]
            series.append({"name": name, "pred": p, "r2": b["metrics"][name]["r2"],
                           "rmse": b["metrics"][name]["rmse"]})
        return {"y": y, "series": series}
    if kind == "fit":
        x = x_column(ds, ch)
        st = ds.stats[x]
        lo, hi = st["min"], st["max"]
        if ch["xs"]:
            lo, hi = min(lo, min(ch["xs"])), max(hi, max(ch["xs"]))
        grid = [lo + (hi - lo) * i / (GRID - 1) for i in range(GRID)] if x not in ds.binary else list(ds.binary[x])
        names = chart_models(b, ch, kind)
        curves = {n: [] for n in names}
        for v in grid:
            p = ds.predict(t, {x: v})
            for n in names:
                curves[n].append(p[n])
        marks = []
        for v in ch["xs"]:
            p = ds.predict(t, dict(plan["values"], **{x: v}))
            marks.append({"x": v, "preds": {n: p[n] for n in names}})
        return {"x": x, "px": ds.df[x].tolist(), "py": ds.df[t].tolist(), "grid": grid,
                "curves": curves, "names": names, "marks": marks, "rep": b["best"] if b["best"] in names else names[0]}
    if kind == "network":
        m = b["models"].get("인공신경망")
        if m is None:
            return None
        acts = None
        out = None
        if plan["values"]:
            full = ds.complete(plan["values"], b["features"])
            x = np.array([[full[c] for c in b["features"]]])
            a = (x - m.xm) / m.xs
            acts = [a.ravel().tolist()]
            for i, (W, B) in enumerate(zip(m.W, m.B)):
                z = a @ W + B
                a = z if i == len(m.W) - 1 else np.maximum(0, z)
                acts.append(a.ravel().tolist())
            out = float(acts[-1][0] * m.ys + m.ym)
        n_params = int(sum(w.size + bb.size for w, bb in zip(m.W, m.B)))
        return {"sizes": list(m.sizes), "inputs": [ds.label[c] for c in b["features"]], "params": n_params,
                "W": [w.tolist() for w in m.W], "acts": acts, "out": out}
    if kind == "importance":
        rows, rep = importance_rows(ds, t, plan)
        return {"rows": [{"col": c, "d": d, "rep": r, "lin": lin, "corr": cr} for c, d, r, lin, cr in rows],
                "rep": rep}
    return None


def summary(ds, plan, kind, d):
    """그래프를 글로 요약 (웹 버전의 그림 아래 글과 같은 숫자)"""
    t = plan["chart"]["target"]
    lt = ds.label[t]
    b = ds.bundle(t)
    if d is None:
        return f"· {KIND_TITLE[kind]}: 이 데이터에는 인공신경망 모델이 없어 그릴 수 없습니다."
    if kind == "learning":
        best = d["best"]
        return (f"· 학습 곡선 ({lt} 예측 인공신경망): {d['epochs']} 에폭 학습, {best} 에폭에서 검증 손실이 가장 낮았습니다 "
                f"({fixed(d['val'][best - 1], 3)}, 표준화한 값의 MSE). 그 뒤 50 에폭 동안 나아지지 않아 멈추고 "
                f"{best} 에폭의 가중치를 썼습니다. 마지막 에폭의 학습 손실 {fixed(d['train'][-1], 3)}, "
                f"검증 손실 {fixed(d['val'][-1], 3)}.")
    if kind == "actual":
        parts = [f"{s['name']} R² {fixed(s['r2'], 3)}, RMSE {fixed(s['rmse'], 2)}" for s in d["series"]]
        return (f"· 실제값 vs 예측값 (테스트 데이터 {len(d['y'])}{ds.unit_word}): " + " / ".join(parts)
                + ". 점이 대각선(예측 = 실제)에 가까울수록 정확합니다.")
    if kind == "residual":
        parts = []
        for s in d["series"]:
            r = [y - p for y, p in zip(d["y"], s["pred"])]
            k = max(range(len(r)), key=lambda i: (abs(r[i]), -i))
            parts.append(f"{s['name']} 잔차 평균 {ds.val(t, sum(r) / len(r), sign=True)}, 가장 큰 잔차 {ds.val(t, r[k], sign=True)}")
        return (f"· 잔차 그래프 (테스트 데이터 {len(d['y'])}{ds.unit_word}, 잔차 = 실제 − 예측): " + " / ".join(parts)
                + ". 잔차가 0 주위에 고르게 흩어져 있으면 모델이 놓친 패턴이 적다는 뜻입니다.")
    if kind == "fit":
        x = d["x"]
        lx = ds.label[x]
        s = (f"· {lx}에 따른 {lt}: 점 {len(d['px'])}개는 실제 데이터, 선은 모델별 예측 "
             f"({', '.join(d['names'])}).")
        if len(ds.columns) > 2:
            s += f" 다른 열은 {lx}에 맞춰 데이터에서 함께 움직이는 값으로 채웠습니다."
        for mk in d["marks"]:
            p = mk["preds"][d["rep"]]
            s += f"\n  표시한 점: {ds.given(x, mk['x'])} → {d['rep']} {ds.val(t, p)}"
        return s
    if kind == "network":
        arch = " → ".join(str(v) for v in d["sizes"])
        s = (f"· 신경망 구조: 입력 {d['sizes'][0]}개({', '.join(d['inputs'])}) → 은닉층 {d['sizes'][1]} → "
             f"은닉층 {d['sizes'][2]} → 출력 1 ({lt}). 구조 {arch}, 학습 파라미터 {d['params']}개, 은닉층은 ReLU. "
             "선 색은 가중치의 부호(파랑 +, 주황 −), 굵기는 크기입니다.")
        if d["acts"] is not None:
            on = [sum(1 for v in a if v > 0) for a in d["acts"][1:-1]]
            s += (f"\n  입력한 값으로 순전파: 은닉층에서 켜진(0보다 큰) 노드 {on[0]}/{d['sizes'][1]}개, "
                  f"{on[1]}/{d['sizes'][2]}개 → 예측 {lt} {ds.val(t, d['out'])}")
        return s
    if kind == "importance":
        top = d["rows"][:3]
        return (f"· {lt}에 대한 영향 (각 열이 표준편차만큼 높을 때, {d['rep']} 기준): "
                + ", ".join(f"{ds.label[r['col']]} {ds.val(t, r['rep'], sign=True)}" for r in top)
                + (" …" if len(d["rows"]) > 3 else ""))
    return ""


# ── 그림 파일 ─────────────────────────────────────────
def _font():
    import matplotlib
    from matplotlib import font_manager
    for name in ("Noto Sans CJK KR", "Noto Sans CJK JP", "Noto Sans CJK SC", "Malgun Gothic", "AppleGothic", "NanumGothic"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = name
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


def _style(ax, title):
    ax.set_title(title, fontsize=12, loc="left", color="#191919")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9ccd1")
    ax.tick_params(colors="#666666", labelsize=9)
    ax.grid(True, color="#ececec", linewidth=0.8)
    ax.set_axisbelow(True)


def draw_png(ds, plan, datas, path):
    """datas: [(kind, data)] → 한 장에 나란히 그린 그림"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _font()
    t = plan["chart"]["target"]
    lt = ds.label[t] + (f" ({ds.unit[t]})" if ds.unit[t] else "")
    panels = []
    for kind, d in datas:
        if d is None:
            continue
        if kind in ("actual", "residual"):
            panels += [(kind, d, s) for s in d["series"]]
        else:
            panels.append((kind, d, None))
    if not panels:
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.4), squeeze=False)
    for ax, (kind, d, s) in zip(axes[0], panels):
        if kind == "learning":
            ep = list(range(1, len(d["train"]) + 1))
            ax.plot(ep, d["train"], color=COLORS["선형회귀"], lw=2, label="학습 손실")
            ax.plot(ep, d["val"], color=COLORS["비선형(2차 다항)"], lw=2, label="검증 손실")
            ax.axvline(d["best"], color="#666666", ls="--", lw=1)
            ax.annotate(f"최적 {d['best']} 에폭", (d["best"], d["val"][d["best"] - 1]), xytext=(8, 12),
                        textcoords="offset points", fontsize=9, color="#191919")
            ax.set_xlabel("에폭")
            ax.set_ylabel("손실 (표준화한 값의 MSE)")
            ax.legend(frameon=False, fontsize=9)
            _style(ax, f"학습 곡선 · {d['epochs']} 에폭에서 멈춤")
        elif kind == "actual":
            lo, hi = min(d["y"] + s["pred"]), max(d["y"] + s["pred"])
            ax.plot([lo, hi], [lo, hi], color="#999999", ls="--", lw=1, label="예측 = 실제")
            ax.scatter(d["y"], s["pred"], s=26, color=COLORS[s["name"]], edgecolor="white", linewidth=0.8, label=s["name"])
            ax.set_xlabel("실제 " + lt)
            ax.set_ylabel("예측 " + lt)
            ax.legend(frameon=False, fontsize=9)
            _style(ax, f"{s['name']} · R² {fixed(s['r2'], 3)}")
        elif kind == "residual":
            r = [y - p for y, p in zip(d["y"], s["pred"])]
            ax.axhline(0, color="#999999", ls="--", lw=1)
            ax.scatter(s["pred"], r, s=26, color=COLORS[s["name"]], edgecolor="white", linewidth=0.8)
            ax.set_xlabel("예측 " + lt)
            ax.set_ylabel("잔차 (실제 − 예측)")
            _style(ax, f"잔차 · {s['name']}")
        elif kind == "fit":
            x = d["x"]
            lx = ds.label[x] + (f" ({ds.unit[x]})" if ds.unit[x] else "")
            ax.scatter(d["px"], d["py"], s=16, color="#b9bec6", edgecolor="none", label="실제 데이터")
            for n in d["names"]:
                ax.plot(d["grid"], d["curves"][n], color=COLORS[n], lw=2, label=n)
            for mk in d["marks"]:
                ax.scatter([mk["x"]], [mk["preds"][d["rep"]]], s=70, color=COLORS[d["rep"]], edgecolor="white",
                           linewidth=2, zorder=5)
                ax.annotate(f"{fmt(mk['x'], 4)} → {fixed(mk['preds'][d['rep']], ds.digits(t))}",
                            (mk["x"], mk["preds"][d["rep"]]), xytext=(6, -14), textcoords="offset points", fontsize=8)
            ax.set_xlabel(lx)
            ax.set_ylabel(lt)
            ax.legend(frameon=False, fontsize=8)
            _style(ax, f"{ds.label[x]}에 따른 {ds.label[t]}")
        elif kind == "network":
            _network(ax, d)
            _style(ax, f"신경망 구조 · 파라미터 {d['params']}개")
            ax.grid(False)
            ax.axis("off")
        elif kind == "importance":
            rows = d["rows"][::-1]
            vals = [r["rep"] for r in rows]
            ax.barh([ds.label[r["col"]] for r in rows], vals,
                    color=[COLORS["선형회귀"] if v >= 0 else COLORS["비선형(2차 다항)"] for v in vals], height=0.6)
            ax.axvline(0, color="#999999", lw=1)
            ax.set_xlabel(f"{ds.label[t]} 변화 (표준편차만큼 높을 때)")
            _style(ax, f"{ds.label[t]}에 대한 영향 · {d['rep']}")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _network(ax, d):
    sizes = d["sizes"]
    pos = []
    for li, n in enumerate(sizes):
        span = min(1.0, 0.09 * (n - 1))                   # 입력 이름이 겹치지 않게 층마다 간격
        ys = [span * (0.5 - i / (n - 1)) if n > 1 else 0.0 for i in range(n)]
        pos.append([(li, y) for y in ys])
    for li, W in enumerate(d["W"]):
        W = np.array(W)
        mx = np.abs(W).max() or 1
        for i, (x0, y0) in enumerate(pos[li]):
            for j, (x1, y1) in enumerate(pos[li + 1]):
                w = W[i, j]
                ax.plot([x0, x1], [y0, y1], color=COLORS["선형회귀"] if w > 0 else COLORS["비선형(2차 다항)"],
                        lw=0.2 + 1.6 * abs(w) / mx, alpha=0.15 + 0.6 * abs(w) / mx, zorder=1)
    for li, layer in enumerate(pos):
        act = d["acts"][li] if d["acts"] else None
        for i, (x, y) in enumerate(layer):
            on = act is not None and (li == 0 or act[i] > 0)
            ax.scatter([x], [y], s=60 if li in (0, len(sizes) - 1) else 30, zorder=3,
                       color=COLORS["인공신경망"] if (on or act is None) else "#d9dce1", edgecolor="white", linewidth=1)
        if li == 0:
            for i, (x, y) in enumerate(layer):
                ax.annotate(d["inputs"][i], (x, y), xytext=(-6, 0), textcoords="offset points", ha="right",
                            va="center", fontsize=8)
    names = ["입력"] + [f"은닉 {n} (ReLU)" for n in sizes[1:-1]] + ["출력"]
    for li, nm in enumerate(names):
        ax.text(li, -0.58, f"{nm}\n{sizes[li]}개", ha="center", va="top", fontsize=9, color="#666666")
    ax.set_xlim(-0.9, len(sizes) - 0.6)
    ax.set_ylim(-0.8, 0.58)
