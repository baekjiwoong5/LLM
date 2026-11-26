# -*- coding: utf-8 -*-
"""
회귀챗: 선형회귀 · 비선형회귀 · 인공신경망에게 문장으로 질문하는 대화형 모델

  python chat.py                              # data/ 폴더의 모든 데이터를 학습하고 대화 시작
  python chat.py -q "키 175면 몸무게는?"       # 질문 하나만 하고 종료 (여러 번 주면 이어지는 대화)
  python chat.py --data 내자료.csv --target 열이름   # 다른 표 데이터로 대화

질문 이해는 nlu.py(규칙은 nlu.json), 계산은 models.py가 합니다.
열마다 그 열을 맞히는 모델 묶음을 따로 학습해 두므로 "몸무게 70이면 키는?"처럼 반대 방향으로 물어도 됩니다.
대형 언어 모델(LLM)처럼 문장을 생성하지는 않으며, 불러온 데이터에 관한 질문에만 답합니다.
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from models import fit_all
from nlu import NLU, UNIT_OF, bmi_class, fixed, fmt, nice2, resolve, round_sig, understand

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SHORT = {"선형회귀": "선형", "비선형(2차 다항)": "2차 다항", "비선형(거듭제곱)": "거듭제곱",
         "비선형(지수)": "지수", "인공신경망": "신경망"}


# ── 한국어 조사와 숫자 ─────────────────────────────────
def batchim(word, rieul_ok=False):
    """마지막 글자에 받침이 있는지 ('키' → 없음, '혈당' → 있음). rieul_ok면 ㄹ받침은 없는 것으로"""
    w = re.sub(r"[^0-9A-Za-z가-힣]+$", "", word)
    if not w:
        return False
    code = ord(w[-1]) - 0xAC00
    if 0 <= code < 11172:
        return code % 28 != 0 and not (rieul_ok and code % 28 == 8)
    return w[-1] in ("013678mnMN" if rieul_ok else "0136781lmnLMN")


def josa(word, pair):
    """josa('혈당', '은는') → '혈당은', josa('키', '으로로') → '키로'"""
    if pair == "으로로":
        return word + ("으로" if batchim(word, rieul_ok=True) else "로")
    return word + (pair[0] if batchim(word) else pair[1])


def g(v):
    return fmt(v, 4)


def nice1(v):
    return round_sig(v, 1)


def half_up(v):
    return math.floor(v + 0.5)


def solve(A, b):
    """A x = b (부분 피벗 가우스 소거). 웹 버전과 같은 순서로 계산해 결과가 같게"""
    n = len(b)
    M = [list(map(float, A[i])) + [float(b[i])] for i in range(n)]
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(M[i][k]))
        M[k], M[p] = M[p], M[k]
        if abs(M[k][k]) < 1e-12:
            continue
        for i in range(k + 1, n):
            f = M[i][k] / M[k][k]
            for j in range(k, n + 1):
                M[i][j] -= f * M[k][j]
    x = [0.0] * n
    for k in range(n - 1, -1, -1):
        if abs(M[k][k]) < 1e-12:
            continue
        s = M[k][n]
        for j in range(k + 1, n):
            s -= M[k][j] * x[j]
        x[k] = s / M[k][k]
    return x


def bmi_criteria():
    parts, prev = [], None
    for lim, name in NLU["bmi_levels"]:
        if prev is None:
            parts.append(f"{fmt(lim)} 미만 {name}")
        elif lim is None:
            parts.append(f"{fmt(prev)} 이상 {name}")
        else:
            parts.append(f"{fmt(prev)}~{fmt(lim - 0.1)} {name}")
        prev = lim
    return ", ".join(parts)


# ── 데이터셋 ──────────────────────────────────────────
class Dataset:
    """표 데이터 하나 + 열마다 그 열을 맞히도록 학습한 모델 묶음"""

    def __init__(self, path, target=None):
        path = Path(path)
        self.path = path
        if path.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        else:
            try:
                df = pd.read_csv(path, encoding="utf-8-sig")
            except UnicodeDecodeError:
                df = pd.read_csv(path, encoding="cp949")
        df.columns = [str(c).strip() for c in df.columns]

        side = path.with_name(path.stem + ".aliases.json")
        cfg = json.loads(side.read_text(encoding="utf-8")) if side.exists() else {}
        self.title = cfg.get("title", path.stem)
        self.keywords = [k.lower() for k in [self.title, path.stem, *cfg.get("keywords", [])]]
        self.unit_word = cfg.get("unit_word", "행")

        num = df.select_dtypes(include="number")
        self.dropped = [c for c in df.columns if c not in num.columns]
        n_before = len(num)
        num = num.dropna()
        self.n_missing = n_before - len(num)
        const = [c for c in num.columns if num[c].nunique() < 2]
        self.dropped += const
        self.columns = [c for c in num.columns if c not in const]
        if len(self.columns) < 2:
            raise ValueError(f"{path.name}: 숫자로 된 열이 두 개 이상 있어야 합니다")
        if len(num) < 10:
            raise ValueError(f"{path.name}: 학습하기에는 데이터가 너무 적습니다 ({len(num)}행)")
        self.df = num[self.columns].astype(float)

        target = target or cfg.get("target") or self.columns[-1]
        if target not in self.columns:
            target = self._find_column(target, self.columns)
        self.target = target
        self.features = [c for c in self.columns if c != target]

        # 열 이름 "키 (cm)" → 이름 "키", 단위 "cm" / 별명·물음 표현·역할은 aliases.json
        self.label, self.unit, aliases, asks, match_unit, self.roles = {}, {}, [], [], {}, {}
        colcfg = cfg.get("columns", {})
        for c in self.columns:
            m = re.match(r"^(.*?)\s*\(([^)]*)\)\s*$", c)
            label, unit = (m.group(1), m.group(2)) if m else (c, "")
            entry = colcfg.get(c, {})
            if isinstance(entry, list):
                entry = {"names": entry}
            names = entry.get("names", [])
            self.label[c] = names[0] if names else label
            self.unit[c] = entry.get("unit", unit)
            match_unit[c] = UNIT_OF.get(self.unit[c].lower(), self.unit[c].lower()) or None
            if entry.get("role"):
                self.roles[entry["role"]] = c
            for a in {c, label, *names}:
                aliases.append((a.lower(), c))
            for a in entry.get("ask", []):
                asks.append((a.lower(), c))
            for w in NLU["units"].get(match_unit[c] or "", []):     # 단위가 있으면 "몇 kg", "몇 살" 자동 추가
                for a in (f"몇 {w}", f"몇{w}"):
                    if (a, c) not in asks:
                        asks.append((a, c))
        aliases.sort(key=lambda t: -len(t[0]))            # 긴 이름부터 찾기 ("TC/HDL" 다음 "HDL")
        asks.sort(key=lambda t: -len(t[0]))

        d = self.df
        self.stats = {c: {"mean": float(d[c].mean()), "std": float(d[c].std()), "min": float(d[c].min()),
                          "max": float(d[c].max()), "median": float(d[c].median())} for c in self.columns}
        # 값이 두 가지뿐인 열(예: 성별 1·2)과 그 열을 채울 값(더 많은 쪽)
        self.binary = {c: [float(v) for v in sorted(d[c].unique())] for c in self.columns if d[c].nunique() == 2}
        self.fill = {c: float(d[c].mode()[0] if c in self.binary else d[c].mean()) for c in self.columns}
        corr, cov = d.corr(), d.cov()
        self.corr = {a: {b: float(corr.loc[a, b]) for b in self.columns} for a in self.columns}
        self.cov = {a: {b: float(cov.loc[a, b]) for b in self.columns} for a in self.columns}
        self.bmi_dist = self._bmi_dist()
        self.spec = {"title": self.title, "keywords": self.keywords, "columns": self.columns,
                     "target": self.target, "label": self.label, "unit": self.unit,
                     "match_unit": match_unit, "aliases": aliases, "asks": asks,
                     "stats": self.stats, "fill": self.fill, "roles": self.roles, "binary": self.binary}
        self._bundles = {}
        self.bundle(self.target)

    @staticmethod
    def _find_column(name, columns):
        for c in columns:
            if name.lower() in c.lower():
                return c
        raise ValueError(f"목표 열 '{name}'을(를) 찾을 수 없습니다. 숫자 열: {', '.join(columns)}")

    def _bmi_dist(self):
        """데이터 안의 BMI 등급별 인원 (비만 질문에 쓰기)"""
        r = self.roles
        if "bmi" in r:
            bmi = self.df[r["bmi"]]
        elif "height" in r and "weight" in r:
            bmi = self.df[r["weight"]] / (self.df[r["height"]] / 100) ** 2
        else:
            return None
        cls = bmi.map(bmi_class)
        show_mean = self.target not in (r.get("height"), r.get("weight"), r.get("bmi"))
        out = []
        for _, name in NLU["bmi_levels"]:
            sel = cls == name
            out.append({"name": name, "count": int(sel.sum()),
                        "mean": float(self.df.loc[sel, self.target].mean()) if show_mean and sel.any() else None})
        return out

    def bundle(self, target):
        """target 열을 맞히는 다섯 모델 (처음 물을 때 한 번만 학습)"""
        if target not in self._bundles:
            feats = [c for c in self.columns if c != target]
            models, metrics, skipped, sizes = fit_all(self.df[feats].to_numpy(), self.df[target].to_numpy())
            best = min(models, key=lambda k: metrics[k]["val_rmse"])
            self._bundles[target] = {"features": feats, "models": models, "metrics": metrics,
                                     "skipped": skipped, "sizes": sizes, "best": best}
        return self._bundles[target]

    def complete(self, values, features):
        """빠진 입력 채우기: 입력한 값과 함께 움직이는 정도(공분산)를 반영한 기댓값.
        값이 두 가지뿐인 열은 더 많은 쪽. 아무것도 입력하지 않았으면 평균"""
        given = {c: v for c, v in values.items() if c in self.columns}
        obs = list(given)
        miss = [c for c in features if c not in given and c not in self.binary]
        adj = {c: 0.0 for c in miss}
        if miss and obs:
            w = solve([[self.cov[a][b] for b in obs] for a in obs],
                      [given[c] - self.stats[c]["mean"] for c in obs])
            for c in miss:
                s = 0.0
                for j, o in enumerate(obs):
                    s += self.cov[c][o] * w[j]
                adj[c] = s
        return {c: given[c] if c in given else self.fill[c] if c in self.binary
                else self.stats[c]["mean"] + adj[c] for c in features}

    def predict_full(self, target, full):
        b = self.bundle(target)
        x = np.array([[full[c] for c in b["features"]]])
        out = {}
        for name, m in b["models"].items():
            with np.errstate(all="ignore"):
                out[name] = float(np.asarray(m.predict(x)).ravel()[0])
        return out

    def predict(self, target, values):
        """{열: 값} → {모델: 예측}. 빠진 입력은 complete()로 채움"""
        return self.predict_full(target, self.complete(values, self.bundle(target)["features"]))

    def digits(self, col):
        sd = self.stats[col]["std"]
        return 1 if sd >= 5 else 2 if sd >= 0.5 else 3

    def val(self, col, v, sign=False):
        """예측값 표시: '69.0 kg'"""
        if not np.isfinite(v):
            return "계산 불가"
        s = fixed(v, self.digits(col))
        if sign and v >= 0:
            s = "+" + s
        return s + (" " + self.unit[col] if self.unit[col] else "")

    def given(self, col, v):
        """입력값 표시: '키 175cm'"""
        return f"{self.label[col]} {g(v)}{self.unit[col]}"


# ── 대화 ──────────────────────────────────────────────
class Brain:
    def __init__(self, datasets):
        self.datasets = list(datasets)
        self.last = self.datasets[0]
        self.prev = None

    def find(self, title):
        return next(d for d in self.datasets if d.title == title)

    def answer(self, raw):
        raw = raw.strip()
        if not raw:
            return self.help()
        if raw.startswith("/load"):
            return self.load(raw.split()[1:])
        chosen = resolve(raw, self.prev)                  # 선택지에 번호로 답한 경우
        if chosen:
            return f"→ {chosen}\n" + self.answer(chosen)
        plan = understand(raw, [d.spec for d in self.datasets], self.last.title, self.prev)
        ds = self.find(plan["dataset"])
        if plan["explicit"] or plan["intent"] not in ("help", "greet", "thanks", "about", "unknown"):
            self.last = ds
        self.prev = plan
        body = getattr(self, "do_" + plan["intent"])(ds, plan)
        extra = []
        if plan["notes"]:
            extra.append("※ 이렇게 읽었습니다: " + " / ".join(plan["notes"]))
        if plan["unused"]:
            extra.append("※ 어느 값인지 몰라 쓰지 않은 숫자: " + ", ".join(plan["unused"]))
        return "\n".join([body] + extra)

    # ── 예측류 ──
    @staticmethod
    def rep(b, plan):
        """대표 모델: 사용자가 고른 모델, 없으면 검증 오차가 가장 작은 모델"""
        return plan["pick"] if plan.get("pick") in b["models"] else b["best"]

    def _rows(self, ds, target, preds, b, plan):
        rep = self.rep(b, plan)
        return [f"  {name:<12} {ds.val(target, p):>12}"
                f"{'  ← 검증 오차 최소' if name == b['best'] else ''}{'  ← 고른 모델' if name == rep != b['best'] else ''}"
                for name, p in preds.items()]

    def _footer(self, ds, target, values, plan, rows_of=None):
        b = ds.bundle(target)
        lines = []
        if target != ds.target:
            lines.append(f"이 답은 {josa(', '.join(ds.label[c] for c in b['features']), '으로로')} "
                         f"{josa(ds.label[target], '을를')} 맞히도록 따로 학습한 모델에서 나왔습니다.")
        if rows_of and len(b["features"]) > 1:
            lines.append(f"입력하지 않은 값은 각 행의 {ds.label[rows_of]} 값에 맞춰 채웠습니다"
                         "(데이터에서 함께 움직이는 정도를 반영).")
        else:
            missing = [c for c in b["features"] if c not in values]
            if missing and len(missing) < len(b["features"]):
                full = ds.complete(values, b["features"])
                lines.append("입력하지 않은 값은 입력한 값에 맞춰 채웠습니다(데이터에서 함께 움직이는 정도를 반영, "
                             "값이 두 종류뿐인 열은 더 많은 쪽): "
                             + ", ".join(f"{ds.label[c]} {g(full[c])}" for c in missing))
        out = [f"{ds.label[c]} {g(v)} (데이터 범위 {g(ds.stats[c]['min'])}~{g(ds.stats[c]['max'])})"
               for c, v in values.items() if not ds.stats[c]["min"] <= v <= ds.stats[c]["max"]]
        if out:
            lines.append("주의: 학습 데이터 범위 밖이라 예측을 믿기 어렵습니다: " + ", ".join(out))
        return lines

    def _binary_line(self, ds, t, x):
        if t not in ds.binary or not np.isfinite(x):
            return []
        lo, hi = ds.binary[t]
        near = hi if x >= (lo + hi) / 2 else lo
        extra = " 어느 쪽이 남자인지는 데이터에 적혀 있지 않습니다." if ds.roles.get("sex") == t else ""
        return [f"{ds.label[t]} 값이 {g(lo)}·{g(hi)} 두 가지뿐이라, 예측 {fixed(x, 3)}은(는) {g(near)} 쪽에 더 가깝습니다.{extra}"]

    def do_predict(self, ds, plan):
        t, values = plan["target"], plan["values"]
        b = ds.bundle(t)
        rep = self.rep(b, plan)
        preds = ds.predict(t, values)
        given = ", ".join(ds.given(c, v) for c, v in values.items() if c != t)
        lines = [f"[{ds.title}] {given} → 예측 {ds.label[t]}"] + self._rows(ds, t, preds, b, plan)
        if np.isfinite(preds[rep]):
            lines.append(f"→ 약 {ds.val(t, preds[rep])} ({rep} 기준, 테스트 오차 RMSE "
                         f"{ds.val(t, b['metrics'][rep]['rmse'])})")
        lines += self._binary_line(ds, t, preds[rep])
        return "\n".join(lines + self._footer(ds, t, values, plan))

    def do_check(self, ds, plan):
        t = plan["target"]
        actual = plan["values"][t]
        values = {c: v for c, v in plan["values"].items() if c != t}
        b = ds.bundle(t)
        rep = self.rep(b, plan)
        preds = ds.predict(t, values)
        best = preds[rep]
        rm = b["metrics"][rep]["rmse"]
        diff = actual - best
        how = ("흔한 차이입니다" if abs(diff) <= rm else "조금 큰 편입니다" if abs(diff) <= 2 * rm
               else "꽤 드문 차이입니다")
        given = ", ".join(ds.given(c, v) for c, v in values.items())
        lines = [f"[{ds.title}] {given}에서 {ds.given(t, actual)}이면?",
                 f"  예측 {ds.label[t]}: 약 {ds.val(t, best)} ({rep})",
                 f"  입력한 값은 예측보다 {ds.val(t, abs(diff))} {'많습니다' if diff > 0 else '적습니다'}. "
                 f"이 모델의 테스트 오차(RMSE)가 {ds.val(t, rm)}이라 {how}."]
        lines += self._rows(ds, t, preds, b, plan)
        return "\n".join(lines + self._footer(ds, t, values, plan))

    def do_sweep(self, ds, plan):
        t, sw = plan["target"], plan["sweep"]
        c = sw["col"]
        b = ds.bundle(t)
        rep = self.rep(b, plan)
        names = list(b["models"])
        head = f"  {ds.label[c] + (f'({ds.unit[c]})' if ds.unit[c] else ''):<10}" + "".join(
            f"{SHORT.get(n, n) + ('*' if n == b['best'] else ''):>10}" for n in names)
        lines = [f"[{ds.title}] {ds.label[c]}별 예측 {ds.label[t]}" + (f" ({ds.unit[t]})" if ds.unit[t] else ""), head]
        reps = []
        for v in sw["values"]:
            preds = ds.predict(t, dict(plan["values"], **{c: v}))
            reps.append(preds[rep])
            lines.append(f"  {g(v):<10}" + "".join(
                f"{(fixed(preds[n], ds.digits(t)) if np.isfinite(preds[n]) else '–'):>10}" for n in names))
        lines.append("(*: 검증 오차가 가장 작은 대표 모델)")
        if len(reps) >= 2 and np.isfinite(reps[0]) and np.isfinite(reps[-1]):
            lines.append(f"→ {rep} 기준 {ds.given(c, sw['values'][0])} → {g(sw['values'][-1])}{ds.unit[c]}에서 "
                         f"{ds.label[t]} {ds.val(t, reps[-1] - reps[0], sign=True)}")
        lines += self._footer(ds, t, plan["values"], plan, rows_of=c)
        out = [g(v) for v in sw["values"] if not ds.stats[c]["min"] <= v <= ds.stats[c]["max"]]
        if out:
            lines.append(f"주의: {ds.label[c]} {', '.join(out)}{ds.unit[c]}은(는) 학습 데이터 범위"
                         f"({g(ds.stats[c]['min'])}~{g(ds.stats[c]['max'])}) 밖이라 믿기 어렵습니다.")
        return "\n".join(lines)

    def do_effect(self, ds, plan):
        t, ef = plan["target"], plan["effect"]
        c, d = ef["col"], ef["delta"]
        b = ds.bundle(t)
        feats = b["features"]
        rep = self.rep(b, plan)
        given = dict(plan["values"])
        base = ds.complete(given, feats)
        p0 = ds.predict_full(t, base)
        p_fix = ds.predict_full(t, dict(base, **{c: base[c] + d}))
        multi = len(feats) > 1
        if multi:
            keep = {k: v for k, v in given.items() if k != c}
            p_tog = ds.predict_full(t, ds.complete(dict(keep, **{c: base[c] + d}), feats))
        if ef["auto"]:
            head = f"{josa(ds.label[c], '이가')} {'높으면' if d > 0 else '낮으면'}"
        else:
            head = f"{josa(ds.label[c], '이가')} {g(abs(d))}{ds.unit[c]} {'늘면' if d > 0 else '줄면'}"
        lines = [f"[{ds.title}] {head} {josa(ds.label[t], '은는')}?"]
        mark = lambda n: f"{'  ← 검증 오차 최소' if n == b['best'] else ''}{'  ← 고른 모델' if n == rep != b['best'] else ''}"
        if multi:
            lines.append(f"  {'모델':<12} {'다른 값 고정':>12} {'함께 움직이는 값 반영':>14}")
            lines += [f"  {n:<12} {ds.val(t, p_fix[n] - p0[n], sign=True):>12} {ds.val(t, p_tog[n] - p0[n], sign=True):>14}{mark(n)}"
                      for n in b["models"]]
            fix_r, tog_r = p_fix[rep] - p0[rep], p_tog[rep] - p0[rep]
            first = [("함께 움직이는 값 반영", tog_r), ("다른 값 고정", fix_r)] if ef["auto"] else \
                [("다른 값 고정", fix_r), ("함께 움직이는 값 반영", tog_r)]
            lines.append(f"→ {rep} 기준 " + " / ".join(f"{name} {ds.val(t, v, sign=True)}" for name, v in first))
        else:
            lines += [f"  {n:<12} {ds.val(t, p_fix[n] - p0[n], sign=True):>12}{mark(n)}" for n in b["models"]]
            if np.isfinite(p_fix[rep] - p0[rep]):
                lines.append(f"→ 약 {ds.val(t, p_fix[rep] - p0[rep], sign=True)} ({rep} 기준)")
        how = "입력값" if c in given else "입력한 값에 맞춘 값" if given else "평균"
        lines.append(f"기준: {ds.label[c]} {g(base[c])}{ds.unit[c]}({how}) → {g(base[c] + d)}{ds.unit[c]}")
        if multi:
            lines.append(f"'다른 값 고정'은 {ds.label[c]}만 바꾼 경우, '함께 움직이는 값 반영'은 {josa(ds.label[c], '이가')} "
                         "그만큼 다른 사람은 보통 다른 값도 함께 다르다고 보고 데이터에서 함께 움직이는 정도만큼 "
                         "같이 바꾼 경우입니다(입력한 값은 그대로).")
        if ef["auto"]:
            lines.append(f"얼마나 바뀌는지 말하지 않아서 {ds.label[c]}의 표준편차({g(abs(d))}{ds.unit[c]})만큼 바꿔 계산했습니다. "
                         f"두 값의 상관계수는 {fixed(ds.corr[c][t], 2)}입니다.")
        tied = [(o, ds.corr[c][o]) for o in feats if o != c and abs(ds.corr[c][o]) >= 0.7]
        if tied:
            lines.append(f"참고: {josa(ds.label[c], '은는')} "
                         + ", ".join(f"{ds.label[o]}(상관 {fixed(r, 2)})" for o, r in tied)
                         + " 값과 함께 움직여서 두 계산이 크게 다를 수 있습니다. 현실에 가까운 쪽은 '함께 움직이는 값 반영'입니다.")
        return "\n".join(lines + self._footer(ds, t, {k: v for k, v in given.items() if k != c}, plan))

    def do_importance(self, ds, plan):
        t = plan["target"]
        b = ds.bundle(t)
        feats = b["features"]
        rep = self.rep(b, plan)
        base = ds.complete({}, feats)
        p0 = ds.predict_full(t, base)
        rows = []
        for c in feats:
            d = nice2(ds.stats[c]["std"])
            p1 = ds.predict_full(t, ds.complete({c: base[c] + d}, feats))
            rows.append((c, d, p1[rep] - p0[rep],
                         p1["선형회귀"] - p0["선형회귀"] if "선형회귀" in p1 else float("nan"), ds.corr[c][t]))
        rows.sort(key=lambda r: -abs(r[2]) if np.isfinite(r[2]) else 0)
        lines = [f"[{ds.title}] {ds.label[t]}에 대한 영향: 각 열이 표준편차만큼 높은 경우",
                 f"  {'열':<14}{'높인 양':>10}{'대표 모델':>12}{'선형회귀':>12}{'상관계수':>10}"]
        for c, d, r_, lin, r in rows:
            lines.append(f"  {ds.label[c]:<14}{'+' + g(d) + ds.unit[c]:>10}{ds.val(t, r_, sign=True):>12}"
                         f"{ds.val(t, lin, sign=True):>12}{fixed(r, 2):>10}")
        lines.append(f"→ {rep} 기준 {ds.label[t]}{'과' if batchim(ds.label[t]) else '와'} 가장 크게 함께 "
                     f"움직이는 열: {ds.label[rows[0][0]]}")
        if len(feats) > 1:
            lines.append("한 열이 높은 사람은 보통 다른 열도 함께 다르므로(예: 총콜레스테롤과 LDL), "
                         "다른 열도 데이터에서 함께 움직이는 정도만큼 같이 바꿔 계산했습니다. 원인과 결과를 뜻하지는 않습니다.")
        return "\n".join(lines)

    def do_obesity(self, ds, plan):
        r, v = ds.roles, plan["values"]
        lines = [f"[{ds.title}] 비만 판정 (BMI 기준: {NLU['bmi_source']})"]

        def model_pred(col, values):
            b = ds.bundle(col)
            rep = self.rep(b, plan)
            return ds.predict(col, values)[rep], rep

        if "bmi" in r:
            bc = r["bmi"]
            if bc in v:
                lines.append(f"  BMI {g(v[bc])} → {bmi_class(v[bc])}")
            elif v:
                x, rep = model_pred(bc, v)
                lines.append(f"  {', '.join(ds.given(c, val) for c, val in v.items())}인 경우 예측 BMI 약 "
                             f"{ds.val(bc, x)} ({rep}) → {bmi_class(x)}")
        else:
            hc, wc = r["height"], r["weight"]
            h, w = v.get(hc), v.get(wc)
            if h is not None and w is not None:
                bmi = w / (h / 100) ** 2
                pw, rep = model_pred(wc, {hc: h})
                lines.append(f"  {ds.given(hc, h)}, {ds.given(wc, w)} → BMI {fixed(bmi, 1)} → {bmi_class(bmi)}")
                lines.append(f"  같은 키의 예측 {ds.label[wc]} {ds.val(wc, pw)} ({rep})보다 "
                             f"{ds.val(wc, abs(w - pw))} {'많습니다' if w > pw else '적습니다'}.")
            elif h is not None:
                pw, rep = model_pred(wc, {hc: h})
                lines.append(f"  {ds.given(hc, h)}인 사람의 예측 {ds.label[wc]} 약 {ds.val(wc, pw)} ({rep}) → "
                             f"BMI {fixed(pw / (h / 100) ** 2, 1)} → {bmi_class(pw / (h / 100) ** 2)}")
            elif w is not None:
                ph, rep = model_pred(hc, {wc: w})
                lines.append(f"  {ds.given(wc, w)}인 사람의 예측 {ds.label[hc]} 약 {ds.val(hc, ph)} ({rep}) → "
                             f"BMI {fixed(w / (ph / 100) ** 2, 1)} → {bmi_class(w / (ph / 100) ** 2)}")
            if h is not None:
                m2 = (h / 100) ** 2
                parts, prev = [], None
                for lim, name in NLU["bmi_levels"]:
                    if prev is None:
                        parts.append(f"{name} {fixed(lim * m2, 1)} 미만")
                    elif lim is None:
                        parts.append(f"{name} {fixed(prev * m2, 1)} 이상")
                    else:
                        parts.append(f"{name} {fixed(prev * m2, 1)}~{fixed(lim * m2, 1)}")
                    prev = lim
                lines.append(f"  {ds.given(hc, h)}의 {ds.label[wc]} 구간({ds.unit[wc]}): " + " / ".join(parts))
        if len(lines) == 1 and ds.bmi_dist:
            lines.append(f"  이 데이터 {len(ds.df)}{ds.unit_word}의 BMI 등급:")
            for row in ds.bmi_dist:
                mean = f", 평균 {ds.label[ds.target]} {g(row['mean'])}" if row["mean"] is not None else ""
                lines.append(f"    {row['name']}: {row['count']}{ds.unit_word}{mean}")
        lines.append(f"기준: {bmi_criteria()}")
        lines.append("BMI는 근육과 지방을 구분하지 못해 참고용이며, 건강 판단은 전문의와 상담하세요.")
        return "\n".join(lines)

    # ── 정보류 ──
    def do_compare(self, ds, plan):
        t = plan["target"]
        b = ds.bundle(t)
        tr, va, te = b["sizes"]
        lines = [f"[{ds.title}] {ds.label[t]} 예측, 테스트 데이터 {te}{ds.unit_word} 기준 (학습 {tr} / 검증 {va} / 테스트 {te})",
                 f"  {'모델':<14}{'MAE':>9}{'RMSE':>9}{'R²':>8}"]
        for name, m in b["metrics"].items():
            mark = "  ← 검증 오차 최소" if name == b["best"] else ""
            lines.append(f"  {name:<14}{fixed(m['mae'], 2):>9}{fixed(m['rmse'], 2):>9}{fixed(m['r2'], 3):>8}{mark}")
        for name, why in b["skipped"].items():
            lines.append(f"  {name}: 건너뜀 ({why})")
        lines.append(f"예측에는 검증 오차가 가장 작은 '{b['best']}'을(를) 대표값으로 씁니다. "
                     "테스트 데이터가 적어 모델 간 작은 차이는 분할에 따라 뒤바뀔 수 있습니다.")
        return "\n".join(lines)

    def do_equation(self, ds, plan):
        t = plan["target"]
        b = ds.bundle(t)
        names = [ds.label[c] for c in b["features"]]
        lines = [f"[{ds.title}] y = {ds.label[t]}"]
        for name, m in b["models"].items():
            if not plan["models"] or name in plan["models"]:
                lines.append(f"  {name}: {m.equation(names)}")
        return "\n".join(lines)

    def do_stats(self, ds, plan):
        cols = list(dict.fromkeys(plan["columns"])) or ds.columns
        lines = [f"[{ds.title}] {len(ds.df)}{ds.unit_word}"]
        for c in cols:
            s = ds.stats[c]
            lines.append(f"  {ds.label[c]}: 평균 {g(s['mean'])}, 중앙값 {g(s['median'])}, 표준편차 {g(s['std'])}, "
                         f"범위 {g(s['min'])}~{g(s['max'])}{' ' + ds.unit[c] if ds.unit[c] else ''}")
        return "\n".join(lines)

    def do_data(self, ds, plan):
        return self.data_info(ds if plan["explicit"] else None)

    def do_need_input(self, ds, plan):
        t = plan["target"]
        feats = [c for c in ds.columns if c != t]
        ex = feats[0]
        return (f"[{ds.title}] {josa(ds.label[t], '을를')} 예측하려면 "
                f"{', '.join(ds.label[c] for c in feats[:4])}{' 등' if len(feats) > 4 else ''} 값을 알려 주세요.\n"
                f"  예: {ds.label[ex]} {g(nice2(ds.stats[ex]['median']))}일 때 {josa(ds.label[t], '은는')}?")

    def do_need_target(self, ds, plan):
        c, v = next(iter(plan["values"].items()))
        other = next(x for x in ds.columns if x not in plan["values"])
        return (f"[{ds.title}] {ds.given(c, v)} 값을 가지고 무엇을 예측할까요?\n"
                f"  예: {ds.given(c, v)}이면 {josa(ds.label[other], '은는')}?")

    def do_greet(self, ds, plan):
        return "안녕하세요! 회귀챗입니다.\n" + self.help()

    def do_thanks(self, ds, plan):
        return "천만에요. 더 궁금한 것이 있으면 물어보세요."

    def do_about(self, ds, plan):
        return ("저는 회귀챗입니다. 선형회귀, 비선형회귀(2차 다항·거듭제곱·지수), 인공신경망을 한 번에 학습해 두고, "
                "질문 문장에서 열 이름과 숫자를 찾아 답합니다. 대형 언어 모델(LLM)은 아니어서 "
                "아래 데이터에 관한 질문에만 답할 수 있어요.\n" + self.data_info())

    def do_help(self, ds, plan):
        return self.help()

    def do_clarify(self, ds, plan):
        if plan["reason"] == "numbers":
            head = "숫자만으로는 어느 값인지 확실하지 않아 데이터 범위로 짐작해 봤어요. 번호로 골라 주세요."
        else:
            head = "질문을 정확히 알아듣지 못해 이런 뜻인지 짐작해 봤어요. 번호로 고르거나 다르게 물어보세요."
        lines = [f"[{ds.title}] {head}"] + [f"  {i}) {o}" for i, o in enumerate(plan["options"], 1)]
        lines.append("열 이름이나 단위를 붙여 물으면 바로 답합니다. 예: " + examples(self.datasets)[0])
        return "\n".join(lines)

    def do_unknown(self, ds, plan):
        return "질문을 이해하지 못했습니다. " + self.help()

    def data_info(self, ds=None):
        lines = []
        for d in ([ds] if ds else self.datasets):
            lines.append(f"[{d.title}] {d.path.name}, {len(d.df)}{d.unit_word}")
            lines.append("  열: " + ", ".join(
                d.label[c] if d.label[c] in c else f"{d.label[c]}({c})" for c in d.columns))
            lines.append(f"  기본 예측: {d.label[d.target]} (다른 열도 물으면 그 열을 맞히는 모델로 답함)")
            lines.append("  모델: " + ", ".join(d.bundle(d.target)["models"]))
            if d.dropped:
                lines.append("  쓰지 않은 열(숫자가 아니거나 값이 하나뿐): " + ", ".join(d.dropped))
            if d.n_missing:
                lines.append(f"  빈 값이 있어 제외한 행: {d.n_missing}")
        return "\n".join(lines)

    def load(self, args):
        if not args:
            return "사용법: /load 파일경로 [목표열]"
        try:
            ds = Dataset(args[0], args[1] if len(args) > 1 else None)
        except Exception as e:
            return f"불러오지 못했습니다: {e}"
        self.datasets = [d for d in self.datasets if d.title != ds.title] + [ds]
        self.last = ds
        self.prev = None
        return "불러와서 학습했습니다.\n" + self.data_info(ds)

    def help(self):
        return ("이렇게 물어보세요. 말투는 자유롭게 바꿔도 됩니다.\n"
                + "\n".join("  " + e for e in examples(self.datasets))
                + "\n단위(1.75m, 70kg, 150파운드), 한글 숫자(백칠십오), 여러 값(160, 170, 180 / 160~180), "
                "변화량(10cm 크면), 반대 방향(몸무게로 키), 이어지는 질문(그럼 180은?)도 알아듣습니다."
                + "\n  /load 파일.csv [목표열]   (다른 표 데이터 불러오기)   /  종료")


def examples(datasets):
    """도움말과 웹 화면의 예시 질문 (데이터에서 만들기)"""
    def ex(d):
        f, t = d.features[0], d.target
        return d, f, t, d.label[f], d.label[t], d.stats[f], d.unit[f]
    small = min(datasets, key=lambda d: len(d.columns))
    big = max(datasets, key=lambda d: len(d.columns))
    out = []
    for d in datasets:
        _, f, t, lf, lt, sf, uf = ex(d)
        out.append(f"{lf} {g(half_up(sf['median']))}일 때 {lt} 예측해줘")
    d, f, t, lf, lt, sf, uf = ex(small)
    st = d.stats[t]
    out.append(f"{lt} {g(half_up(st['median']))}{d.unit[t]}이면 {josa(lf, '은는')}?")
    out.append(f"{lf} {g(nice2(sf['mean'] - sf['std']))}~{g(nice2(sf['mean'] + sf['std']))}일 때 {lt}")
    out.append(f"{josa(lf, '이가')} {g(nice1(sf['std']))}{uf} 늘면 {josa(lt, '은는')}?")
    if big is not small:
        out.append(f"{big.label[big.target]}에 영향이 큰 건?")
    if any(d.bmi_dist for d in datasets):
        out.append("비만이야?")
    out += ["어떤 모델이 제일 정확해?", "회귀식 알려줘", "평균이랑 범위 알려줘", "어떤 데이터가 있어?"]
    return out


def default_paths():
    return sorted(p for p in DATA_DIR.iterdir() if p.suffix.lower() in (".csv", ".xlsx"))


def main():
    ap = argparse.ArgumentParser(description="선형회귀·비선형회귀·인공신경망에게 문장으로 질문하기")
    ap.add_argument("--data", nargs="+", help="표 데이터 파일(csv/xlsx). 생략하면 data/ 폴더 전체")
    ap.add_argument("--target", help="기본으로 예측할 열 이름 (--data와 함께, 생략하면 마지막 숫자 열)")
    ap.add_argument("-q", "--question", action="append", help="질문 하나만 하고 종료 (여러 번 주면 이어지는 대화)")
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stdin):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    paths = [Path(p) for p in args.data] if args.data else default_paths()
    datasets = [Dataset(p, args.target if args.data else None) for p in paths]
    brain = Brain(datasets)

    if args.question:
        for q in args.question:
            print(brain.answer(q))
        return

    print("회귀챗: 선형회귀 · 비선형회귀 · 인공신경망을 한 번에 학습했습니다.\n")
    print(brain.data_info())
    print("\n" + brain.help() + "\n")
    while True:
        try:
            q = input("질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q in ("종료", "끝", "exit", "quit", "q"):
            break
        print(brain.answer(q) + "\n")


if __name__ == "__main__":
    main()
