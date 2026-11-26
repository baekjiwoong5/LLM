# -*- coding: utf-8 -*-
"""
시험 질문으로 챗봇 확인하기

  python tests/run_tests.py

1) tests/questions.json의 질문마다 파이썬 버전이 기대한 대로 이해하는지 (의도, 데이터, 목표 열, 값)
2) Node.js가 있으면 웹 버전(index.html)도 같은 질문에 같은 계획을 세우고 같은 숫자로 답하는지
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chat import Brain, Dataset, default_paths  # noqa: E402
from nlu import resolve, understand  # noqa: E402

QUESTIONS = json.loads((ROOT / "tests" / "questions.json").read_text(encoding="utf-8"))


def close(a, b):
    return abs(a - b) <= 1e-3 * max(1, abs(b))


def problems(plan, exp):
    out = []
    for key in ("intent", "dataset", "target", "pick"):
        if key in exp and plan.get(key) != exp[key]:
            out.append(f"{key}: {plan.get(key)!r} (기대 {exp[key]!r})")
    if "values" in exp:
        got = plan["values"]
        if set(got) != set(exp["values"]) or not all(close(got[k], v) for k, v in exp["values"].items()):
            out.append(f"values: {got} (기대 {exp['values']})")
    if "sweep" in exp:
        sw = plan.get("sweep") or {}
        col, vals = exp["sweep"]
        if sw.get("col") != col or len(sw.get("values", [])) != len(vals) or \
                not all(close(a, b) for a, b in zip(sw["values"], vals)):
            out.append(f"sweep: {sw} (기대 {exp['sweep']})")
    if "effect" in exp:
        ef = plan.get("effect") or {}
        col, d = exp["effect"]
        ok = ef.get("col") == col and (
            (d == "+" and ef.get("delta", 0) > 0) or (d == "-" and ef.get("delta", 0) < 0)
            or (not isinstance(d, str) and close(ef.get("delta", 0), d)))
        if not ok:
            out.append(f"effect: {ef} (기대 {exp['effect']})")
    for key in ("models", "columns"):
        if key in exp and sorted(plan.get(key, [])) != sorted(exp[key]):
            out.append(f"{key}: {plan.get(key)} (기대 {exp[key]})")
    return out


def canon(x):
    """비교용: 숫자는 소수 9자리로, 정수·실수 구분 없이"""
    if isinstance(x, bool) or x is None or isinstance(x, str):
        return x
    if isinstance(x, (int, float)):
        return round(float(x), 9)
    if isinstance(x, dict):
        return {k: canon(v) for k, v in x.items()}
    return [canon(v) for v in x]


def same(a, b):
    return json.dumps(canon(a), sort_keys=True, ensure_ascii=False) == json.dumps(canon(b), sort_keys=True, ensure_ascii=False)


def numbers(text):
    text = re.sub(r"<[^>]+>", " ", text).replace("&gt;", ">").replace("&amp;", "&").replace("−", "-")
    return sorted(set(re.findall(r"[-+]?\d+(?:\.\d+)?", text)))


def run_js(questions):
    node = shutil.which("node")
    html = ROOT / "index.html"
    if not node or "/*DATA_START*/const DATA = [];" in html.read_text(encoding="utf-8"):
        return None
    harness = ROOT / "tests" / "js_harness.js"
    r = subprocess.run([node, str(harness), str(html)], input=json.dumps(questions, ensure_ascii=False),
                       capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise SystemExit("웹 버전 실행 실패:\n" + r.stderr)
    return json.loads(r.stdout)


def main():
    brain = Brain([Dataset(p) for p in default_paths()])
    specs = [d.spec for d in brain.datasets]
    qs = [x["q"] for x in QUESTIONS]
    py = []
    fails = 0
    for exp in QUESTIONS:
        if exp.get("fresh"):
            brain.prev = None                             # "fresh": 앞 질문과 잇지 않고 따로 묻기
        q = resolve(exp["q"], brain.prev) or exp["q"]     # 선택지에 번호로 답하면 그 질문으로
        plan = understand(q, specs, brain.last.title, brain.prev)
        text = brain.answer(exp["q"])                     # 직전 데이터·직전 질문도 함께 갱신
        py.append({"plan": plan, "text": text})
        bad = problems(plan, exp)
        if bad:
            fails += 1
            print(f"✗ {exp['q']}\n    " + "\n    ".join(bad))
    print(f"파이썬: {len(QUESTIONS) - fails}/{len(QUESTIONS)} 통과")

    js = run_js([{"q": x["q"], "fresh": bool(x.get("fresh"))} for x in QUESTIONS])
    if js is None:
        print("웹 버전: 건너뜀 (Node.js가 없거나 index.html에 학습 결과가 없음)")
        return fails
    js_fails = 0
    for q, p, j in zip(qs, py, js):
        diff = []
        if not same(p["plan"], j["plan"]):
            pa, ja = p["plan"], j["plan"] or {}
            keys = sorted(k for k in set(pa) | set(ja) if not same(pa.get(k), ja.get(k)))
            diff.append("계획: " + "; ".join(f"{k} 파이썬={pa.get(k)!r} 웹={ja.get(k)!r}" for k in keys))
        a, b = numbers(p["text"]), numbers(j["html"])
        if a != b:
            diff.append(f"숫자: 파이썬에만 {sorted(set(a) - set(b))[:8]}, 웹에만 {sorted(set(b) - set(a))[:8]}")
        if diff:
            js_fails += 1
            print(f"✗ (웹) {q}\n    " + "\n    ".join(diff))
    print(f"웹 버전: {len(qs) - js_fails}/{len(qs)} 파이썬과 일치")
    return fails + js_fails


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
