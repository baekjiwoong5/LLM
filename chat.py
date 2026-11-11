# -*- coding: utf-8 -*-
"""
회귀챗: 선형회귀 · 비선형회귀 · 인공신경망에게 문장으로 질문하는 대화형 모델

  python chat.py                              # data/ 폴더의 모든 데이터를 학습하고 대화 시작
  python chat.py -q "키 175면 몸무게는?"       # 질문 하나만 하고 종료
  python chat.py --data 내자료.csv --target 열이름   # 다른 표 데이터로 대화

질문 문장에서 열 이름과 숫자를 찾아(규칙 기반) 세 모델 계열의 예측을 한꺼번에 답합니다.
대형 언어 모델(LLM)처럼 문장을 생성하지는 않으며, 불러온 데이터에 관한 질문에만 답합니다.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from models import fit_all

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

NUM_RE = re.compile(r"(?<![0-9a-z.])-?\d+(?:\.\d+)?")
MAX_GAP = 10          # 열 이름과 숫자 사이에 허용하는 글자 수 ("키는 약 175")

KEYWORDS = {
    "help": ["도움", "사용법", "help", "할 수 있", "기능", "예시", "안녕", "hello"],
    "compare": ["비교", "성능", "정확", "어떤 모델", "어느 모델", "제일 좋", "가장 좋", "최고",
                "r2", "r²", "rmse", "mae", "오차", "결과"],
    "equation": ["수식", "회귀식", "방정식", "계수", "기울기", "구조", "파라미터"],
    "stats": ["평균", "최소", "최대", "최솟값", "최댓값", "범위", "표준편차"],
    "data": ["데이터", "자료", "표본", "몇 명", "몇명", "몇 개", "몇개", "변수", "특성", "목록", "입력"],
}
EQ_WORD_RE = re.compile(r"(?<![가-힣])식(?![가-힣]{2,})")
MODEL_WORDS = {"선형회귀": ["선형"], "비선형(2차 다항)": ["다항", "2차"], "비선형(거듭제곱)": ["거듭제곱"],
               "비선형(지수)": ["지수"], "인공신경망": ["신경망", "ann", "mlp"]}


def _g(v):
    return f"{v:.4g}"


# ── 데이터셋 ──────────────────────────────────────────
class Dataset:
    """표 데이터 하나 + 그 데이터로 한 번에 학습한 모델들"""

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

        target = target or cfg.get("target") or num.columns[-1]
        if target not in num.columns:
            target = self._find_column(target, num.columns)
        const = [c for c in num.columns if c != target and num[c].nunique() < 2]
        self.dropped += const
        self.target = target
        self.features = [c for c in num.columns if c != target and c not in const]
        if not self.features:
            raise ValueError(f"{path.name}: 숫자로 된 입력 열이 없습니다")
        if len(num) < 10:
            raise ValueError(f"{path.name}: 학습하기에는 데이터가 너무 적습니다 ({len(num)}행)")
        self.df = num[self.features + [target]].astype(float)

        # 열 이름 "키 (cm)" → 이름 "키", 단위 "cm"
        self.label, self.unit, self.aliases = {}, {}, []
        for c in self.features + [target]:
            m = re.match(r"^(.*?)\s*\(([^)]*)\)\s*$", c)
            label, unit = (m.group(1), m.group(2)) if m else (c, "")
            extra = cfg.get("columns", {}).get(c, [])
            self.label[c] = extra[0] if extra else label
            self.unit[c] = unit
            for a in {c, label, *extra}:
                self.aliases.append((a.lower(), c))
        self.aliases.sort(key=lambda t: -len(t[0]))      # 긴 이름부터 찾기 ("TC/HDL" 다음 "HDL")

        # 질문에서 빠진 입력을 채울 값: 평균. 값이 두 종류뿐인 열은 더 많은 쪽
        self.fill = {c: float(self.df[c].mode()[0] if self.df[c].nunique() <= 2 else self.df[c].mean())
                     for c in self.features}
        X = self.df[self.features].to_numpy()
        y = self.df[target].to_numpy()
        self.models, self.metrics, self.skipped, self.sizes = fit_all(X, y)
        self.best = min(self.models, key=lambda k: self.metrics[k]["val_rmse"])

    @staticmethod
    def _find_column(name, columns):
        for c in columns:
            if name.lower() in c.lower():
                return c
        raise ValueError(f"목표 열 '{name}'을(를) 찾을 수 없습니다. 숫자 열: {', '.join(columns)}")

    def with_unit(self, col, v, digits=1):
        u = self.unit[col]
        return f"{v:.{digits}f}{' ' + u if u else ''}"

    # 문장에서 열 이름 찾기 → [(시작, 끝, 열)], 열 이름을 공백으로 가린 문장
    def find_columns(self, text):
        masked, hits = text, []
        for alias, col in self.aliases:
            start = 0
            while (i := masked.find(alias, start)) != -1:
                j = i + len(alias)
                start = j
                before, after = masked[i - 1:i] if i else "", masked[j:j + 1]
                if alias.isascii() and (re.match(r"[a-z0-9]", before) or re.match(r"[a-z0-9]", after)):
                    continue                              # "age"가 "average" 안에서 걸리지 않게
                if len(alias) == 1 and re.match(r"[가-힣]", before):
                    continue                              # "키"가 "쿠키" 안에서 걸리지 않게
                hits.append((i, j, col))
                masked = masked[:i] + " " * len(alias) + masked[j:]
        return sorted(hits), masked


# ── 대화 ──────────────────────────────────────────────
class Brain:
    def __init__(self, datasets):
        self.datasets = list(datasets)
        self.last = self.datasets[0]

    # 어떤 데이터에 관한 질문인지 고르기
    def route(self, text):
        best, best_score, parsed = None, 0, {}
        for ds in sorted(self.datasets, key=lambda d: d is not self.last):   # 동점이면 직전 데이터
            hits, masked = ds.find_columns(text)
            parsed[ds.title] = (hits, masked)
            score = len(hits) + sum(k in text for k in ds.keywords)
            if score > best_score:
                best, best_score = ds, score
        ds = best or self.last
        return ds, *parsed[ds.title], best is not None

    def answer(self, raw):
        raw = raw.strip()
        if not raw:
            return self.help()
        if raw.startswith("/load"):
            return self.load(raw.split()[1:])
        text = raw.lower()
        ds, hits, masked, explicit = self.route(text)
        self.last = ds
        for words in MODEL_WORDS.values():               # "2차 다항"의 2를 숫자로 읽지 않게
            for w in words:
                masked = masked.replace(w, " " * len(w))
        numbers = [(m.start(), float(m.group())) for m in NUM_RE.finditer(masked)]

        # 열 이름 바로 뒤의 숫자를 그 열의 값으로
        values, used = {}, set()
        for k, (s, e, col) in enumerate(hits):
            nxt = hits[k + 1][0] if k + 1 < len(hits) else len(masked)
            for pos, v in numbers:
                if e <= pos < nxt and pos - e <= MAX_GAP and pos not in used and col not in values:
                    values[col] = v
                    used.add(pos)
        free = [v for pos, v in numbers if pos not in used]
        mentioned = [c for _, _, c in hits]
        intent = next((name for name, kws in KEYWORDS.items() if any(k in masked for k in kws)), None)
        if intent is None and EQ_WORD_RE.search(masked):
            intent = "equation"

        if ds.target in values and not any(c in values for c in ds.features):
            return (f"[{ds.title}] 이 모델은 입력({', '.join(ds.label[c] for c in ds.features)}) → "
                    f"출력({ds.label[ds.target]}) 방향으로만 학습되어 있어, "
                    f"{ds.label[ds.target]} 값으로 입력을 거꾸로 맞히는 질문에는 답할 수 없습니다.")
        values.pop(ds.target, None)
        if values:
            return self.predict(ds, values)
        if intent == "help":
            return self.help()
        if intent == "compare":
            return self.compare(ds)
        if intent == "equation":
            return self.equation(ds, text)
        if intent == "stats":
            return self.stats(ds, mentioned)
        if intent == "data":
            return self.data_info(ds if explicit else None)
        # 열 이름 없이 숫자만 준 경우: 입력 열이 하나면 그 값, 개수가 맞으면 열 순서대로
        candidates = [ds] if explicit else [ds] + [d for d in self.datasets if d is not ds]
        for d in candidates:
            if len(free) == len(d.features):
                self.last = d
                return self.predict(d, dict(zip(d.features, free)))
        if mentioned:
            return self.stats(ds, mentioned)
        return "질문을 이해하지 못했습니다. " + self.help()

    # ── 답변들 ──
    def predict(self, ds, values):
        x = np.array([[values.get(c, ds.fill[c]) for c in ds.features]])
        given = ", ".join(f"{ds.label[c]} {_g(v)}{ds.unit[c]}" for c, v in values.items())
        t = ds.target
        lines = [f"[{ds.title}] {given} → 예측 {ds.label[t]}"]
        preds = {}
        for name, m in ds.models.items():
            with np.errstate(all="ignore"):
                p = float(np.asarray(m.predict(x)).ravel()[0])
            preds[name] = p
            mark = "  ← 검증 오차 최소" if name == ds.best else ""
            shown = ds.with_unit(t, p) if np.isfinite(p) else "계산 불가"
            lines.append(f"  {name:<12} {shown:>10}{mark}")
        best = preds[ds.best]
        if np.isfinite(best):
            lines.append(f"→ 약 {ds.with_unit(t, best)} "
                         f"(이 모델의 테스트 오차 RMSE {ds.with_unit(t, ds.metrics[ds.best]['rmse'])})")
        missing = [c for c in ds.features if c not in values]
        if missing:
            lines.append("입력하지 않은 값은 데이터 평균(값이 두 종류뿐인 열은 더 많은 쪽)으로 채웠습니다: "
                         + ", ".join(f"{ds.label[c]} {_g(ds.fill[c])}" for c in missing))
        out = [f"{ds.label[c]} {_g(v)} (데이터 범위 {_g(ds.df[c].min())}~{_g(ds.df[c].max())})"
               for c, v in values.items() if not ds.df[c].min() <= v <= ds.df[c].max()]
        if out:
            lines.append("주의: 학습 데이터 범위 밖이라 예측을 믿기 어렵습니다: " + ", ".join(out))
        return "\n".join(lines)

    def compare(self, ds):
        tr, va, te = ds.sizes
        lines = [f"[{ds.title}] 테스트 데이터 {te}{ds.unit_word} 기준 (학습 {tr} / 검증 {va} / 테스트 {te})",
                 f"  {'모델':<14}{'MAE':>9}{'RMSE':>9}{'R²':>8}"]
        for name, m in ds.metrics.items():
            mark = "  ← 검증 오차 최소" if name == ds.best else ""
            lines.append(f"  {name:<14}{m['mae']:9.2f}{m['rmse']:9.2f}{m['r2']:8.3f}{mark}")
        for name, why in ds.skipped.items():
            lines.append(f"  {name}: 건너뜀 ({why})")
        lines.append(f"예측에는 검증 오차가 가장 작은 '{ds.best}'을(를) 대표값으로 씁니다. "
                     "테스트 데이터가 적어 모델 간 작은 차이는 분할에 따라 뒤바뀔 수 있습니다.")
        return "\n".join(lines)

    def equation(self, ds, text):
        names = [ds.label[c] for c in ds.features]
        want = [n for n, ws in MODEL_WORDS.items() if any(w in text for w in ws)
                and not (n == "선형회귀" and "비선형" in text and text.count("선형") == 1)]
        if "비선형" in text:
            want += [n for n in ds.models if n.startswith("비선형")]
        lines = [f"[{ds.title}] y = {ds.label[ds.target]}"]
        for name, m in ds.models.items():
            if not want or name in want:
                lines.append(f"  {name}: {m.equation(names)}")
        return "\n".join(lines)

    def stats(self, ds, cols):
        cols = list(dict.fromkeys(cols)) or ds.features + [ds.target]
        lines = [f"[{ds.title}] {len(ds.df)}{ds.unit_word}"]
        for c in cols:
            s = ds.df[c]
            lines.append(f"  {ds.label[c]}: 평균 {_g(s.mean())}, 표준편차 {_g(s.std())}, "
                         f"범위 {_g(s.min())}~{_g(s.max())}{' ' + ds.unit[c] if ds.unit[c] else ''}")
        return "\n".join(lines)

    def data_info(self, ds=None):
        lines = []
        for d in ([ds] if ds else self.datasets):
            lines.append(f"[{d.title}] {d.path.name}, {len(d.df)}{d.unit_word}")
            lines.append("  입력: " + ", ".join(
                d.label[c] if d.label[c] in c else f"{d.label[c]}({c})" for c in d.features))
            lines.append(f"  목표: {d.label[d.target]}")
            lines.append("  학습된 모델: " + ", ".join(d.models))
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
        return "불러와서 학습했습니다.\n" + self.data_info(ds)

    def help(self):
        ex = []
        for d in self.datasets:
            f = d.features[0]
            ex.append(f"  {d.label[f]} {_g(round(d.df[f].median()))}일 때 {d.label[d.target]} 예측해줘")
        return ("이렇게 물어보세요.\n" + "\n".join(ex) +
                "\n  어떤 모델이 제일 정확해?   /  회귀식 알려줘  /  신경망 구조는?"
                "\n  평균이랑 범위 알려줘       /  어떤 데이터가 있어?"
                "\n  /load 파일.csv [목표열]   (다른 표 데이터 불러오기)   /  종료")


def default_paths():
    return sorted(p for p in DATA_DIR.iterdir() if p.suffix.lower() in (".csv", ".xlsx"))


def main():
    ap = argparse.ArgumentParser(description="선형회귀·비선형회귀·인공신경망에게 문장으로 질문하기")
    ap.add_argument("--data", nargs="+", help="표 데이터 파일(csv/xlsx). 생략하면 data/ 폴더 전체")
    ap.add_argument("--target", help="예측할 열 이름 (--data와 함께, 생략하면 마지막 숫자 열)")
    ap.add_argument("-q", "--question", action="append", help="질문 하나만 하고 종료 (여러 번 가능)")
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
