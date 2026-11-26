# -*- coding: utf-8 -*-
"""
질문 문장을 이해하는 부분 (규칙 기반)

understand()는 질문을 읽어 "무엇을 해야 하는지"를 담은 계획(plan) dict를 돌려줍니다.
계산과 출력은 chat.py가 합니다. index.html에도 같은 규칙이 JavaScript로 들어 있고,
tests/run_tests.py가 두 쪽의 계획이 같은지 확인합니다.

알아듣는 것
  · 열 이름과 별명, 한두 글자 오타 (몸무계 → 몸무게)
  · 단위: 175cm, 1.75m, 1m 75, 70kg, 150파운드, 59살
  · 한글 숫자: 백칠십오, 마흔다섯 살
  · 어순: "키 175면", "175cm인 사람", "몸무게는? 키가 175일 때"
  · 반대 방향: "몸무게 70이면 키는?" (그 방향으로 따로 학습한 모델로 답함)
  · 여러 값: "160, 170, 180", "160~180", "160부터 180까지"
  · 변화량: "키가 10cm 크면", "1cm당", "BMI가 높으면"
  · 영향: "진행도에 뭐가 제일 영향이 커?"
  · 검산: "키 175에 몸무게 80이면?" (예측과 비교)
"""
import json
import math
import re
import unicodedata
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

NLU = json.loads((Path(__file__).resolve().parent / "nlu.json").read_text(encoding="utf-8"))
MAX_GAP = NLU["max_gap"]
NUM_MASK, MENTION_MASK, CONCEPT_MASK = "#", "@", "%"


# ── 작은 도구들 ───────────────────────────────────────
def round_sig(v, digits):
    """유효숫자 digits자리로 반올림 (JavaScript toPrecision과 같은 사사오입)"""
    if v == 0 or not math.isfinite(v):
        return float(v)
    d = Decimal(v)
    return float(d.quantize(Decimal(1).scaleb(d.adjusted() - digits + 1), rounding=ROUND_HALF_UP))


def fixed(v, digits):
    """소수점 digits자리 문자열 (JavaScript toFixed와 같은 결과)"""
    q = Decimal(abs(v)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
    return ("-" if v < 0 else "") + f"{q:f}"


def fmt(v, digits=6):
    """숫자를 짧게: 175.0 → '175', 68.03886... → '68.0389' (JavaScript와 같은 결과)"""
    if v is None or not math.isfinite(v):
        return "–"
    x = round_sig(v, digits)
    if x != 0 and (abs(x) < 1e-6 or abs(x) >= 1e21):       # JavaScript처럼 2e-7, 1e+21
        return re.sub(r"e([+-])0*(\d)", r"e\1\2", repr(x)).replace(".0e", "e")
    s = format(Decimal(repr(x)), "f")
    return s[:-2] if s.endswith(".0") else s


def nice2(v):
    """유효숫자 두 자리: 9.47 → 9.5"""
    return round_sig(v, 2)


def is_latin(s):
    return re.fullmatch(r"[a-z0-9 /._()\-]+", s) is not None


def has_word(text, kw):
    """키워드가 문장에 있는지. 영문은 단어 중간에서 걸리지 않게 경계를 본다"""
    if is_latin(kw):
        pat = r"(?<![a-z])" + re.escape(kw) + (r"(?![a-z])" if len(kw) <= 3 else "")
        return re.search(pat, text) is not None
    return kw in text


def bigrams(s):
    s = re.sub(r"\s+", "", s)
    return [s[i:i + 2] for i in range(len(s) - 1)]


def dice(a, b):
    """두 문장이 얼마나 비슷한지 (글자 2개씩 묶음이 겹치는 정도, 0~1)"""
    A, B = bigrams(a), bigrams(b)
    if not A or not B:
        return 0.0
    rest, hit = list(B), 0
    for g in A:
        if g in rest:
            rest.remove(g)
            hit += 1
    return 2 * hit / (len(A) + len(B))


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def mask(text, s, e, ch):
    return text[:s] + ch * (e - s) + text[e:]


def jamo(word):
    """'혈당' → 초성·중성·종성 조각 목록 (오타를 자모 단위로 비교하기 위해)"""
    out = []
    for ch in word:
        c = ord(ch) - 0xAC00
        if 0 <= c < 11172:
            out += [f"c{c // 588}", f"v{(c % 588) // 28}"] + ([f"t{c % 28}"] if c % 28 else [])
        else:
            out.append("x" + ch)
    return out


def flex_pattern(alias):
    """세 글자 이상 한글 이름은 글자 사이 띄어쓰기를 무시: '중성지방' ↔ '중성 지방'"""
    if is_latin(alias) or len(alias.replace(" ", "")) < 3:
        return None
    return re.compile(r"\s?".join(re.escape(ch) for ch in alias.replace(" ", "")))


# ── 1. 문장 정리: 한글 숫자, 1m 75 ─────────────────────
KN = NLU["korean_numbers"]
SINO_RE = re.compile("(?<![가-힣])[" + "".join(KN["digits"]) + "".join(KN["units"]) + "]+")
NATIVE_RE = re.compile(
    "(?<![가-힣])(" + "|".join(sorted(KN["native_tens"], key=len, reverse=True)) + ")("
    + "|".join(sorted(KN["native_ones"], key=len, reverse=True)) + ")?(?=\\s*(?:살|세)|[?.!,]|$)")
METER_RE = re.compile(r"(?<![\d.])(\d)\s*(?:m|미터)\s*(\d{2})(?:\s*(?:cm|센치|센티))?(?![\d.a-z])")
FTIN_RE = re.compile(r"(?<![\d.])(\d)\s*(?:피트|feet|foot|ft)\s*(\d{1,2})\s*(?:인치|inches|inch)?(?![\d.a-z])")
DEC = NLU["decade"]
DECADE_RE = re.compile(r"(?<![\d.])([1-9]\d*0)\s*대\s*(" + "|".join(DEC["words"]) + r")?(?!\s*\d)(?!"
                       + "|".join(map(re.escape, DEC["not_followed_by"])) + ")")


def parse_sino(s):
    total, cur, last, has_unit = 0, None, 10 ** 9, False
    for ch in s:
        if ch in KN["digits"]:
            if cur is not None:
                return None
            cur = KN["digits"][ch]
        else:
            u = KN["units"][ch]
            if u >= last:
                return None
            total += (1 if cur is None else cur) * u
            cur, last, has_unit = None, u, True
    return total + (cur or 0) if has_unit else None


def normalize(text):
    notes = []
    t = unicodedata.normalize("NFKC", text).lower()
    t = re.sub(r"[〜∼]", "~", t).replace("–", "-").replace("—", "-")

    out, pos = [], 0
    for m in SINO_RE.finditer(t):                       # 백칠십오 → 175
        s = m.group()
        for end in range(len(s), 0, -1):
            v = parse_sino(s[:end])
            nxt = t[m.start() + end:m.start() + end + 1]
            if end > 1 and s[end - 1] == "이" and nxt and nxt in KN["particle_i_before"]:
                continue                                  # 백육십이면 = 160 + 이면 (162는 백육십이이면)
            if v is not None and (not re.match("[가-힣]", nxt) or nxt in KN["follow_ok"]):
                out.append(t[pos:m.start()] + str(v))
                pos = m.start() + end
                notes.append(f"'{s[:end]}' → {v}")
                break
    t = "".join(out) + t[pos:]

    def native(m):                                       # 마흔다섯 → 45
        v = KN["native_tens"][m.group(1)] + (KN["native_ones"][m.group(2)] if m.group(2) else 0)
        notes.append(f"'{m.group()}' → {v}")
        return str(v)
    t = NATIVE_RE.sub(native, t)

    def meter(m):                                        # 1m 75 → 175cm
        v = int(m.group(1)) * 100 + int(m.group(2))
        notes.append(f"'{m.group().strip()}' → {v}cm")
        return f"{v}cm"
    t = METER_RE.sub(meter, t)

    def ftin(m):                                         # 5피트 9인치 → 175.26cm
        v = int(m.group(1)) * 30.48 + int(m.group(2)) * 2.54
        notes.append(f"'{m.group().strip()}' → {fmt(v)}cm")
        return f"{fmt(v)}cm"
    t = FTIN_RE.sub(ftin, t)

    def decade(m):                                       # 50대 → 55세, 170대 중반 → 175
        base = int(m.group(1))
        v = base + (DEC["words"][m.group(2)] if m.group(2) else DEC["default"])
        out = f"{v}세" if base <= DEC["age_max"] else str(v)
        notes.append(f"'{m.group().strip()}' → {out}")
        return out
    t = DECADE_RE.sub(decade, t)
    return t, notes


# ── 2. 숫자와 단위 찾기 ───────────────────────────────
UNIT_OF = {w: canon for canon, words in NLU["units"].items() for w in words}


def _words_alt(words):
    return "|".join(re.escape(w) + ("(?![a-z])" if is_latin(w) else "")
                    for w in sorted(words, key=len, reverse=True))


# 단위는 띄어 써도 되고(175 cm), '2차' '1년' 같은 무시할 단위는 붙어 있을 때만
NUM_RE = re.compile(r"(?<![0-9a-z.])(\d+(?:\.\d+)?)(?:\s*(" + _words_alt(UNIT_OF) + ")|("
                    + _words_alt(NLU["ignore_units"]) + "))?")


def find_numbers(t):
    toks = []
    for m in NUM_RE.finditer(t):
        toks.append({"s": m.start(), "e": m.end(), "v": float(m.group(1)), "raw": m.group().strip(),
                     "unit": UNIT_OF[m.group(2)] if m.group(2) else None, "ignore": m.group(3) is not None})
    return toks


# ── 3. 열 이름 찾기 ───────────────────────────────────
def find_mentions(spec, text):
    """질문에서 열 이름(별명, '몇 kg' 같은 물음 표현, 오타 포함)과 개념어(비만, 남자/여자)를 찾는다"""
    masked, hits, notes = text, [], []
    concepts = {name: None for name in NLU["concepts"]}
    for name, words in NLU["concepts"].items():
        for w in sorted(words, key=len, reverse=True):
            start = 0
            while (i := masked.find(w, start)) != -1:
                start = i + len(w)
                if is_latin(w) and (re.match("[a-z]", masked[i - 1:i]) or re.match("[a-z]", masked[i + len(w):i + len(w) + 1])):
                    continue
                inside = any(len(a) > len(w) and w in a and any(
                    masked.find(a, k) == k for k in range(max(0, i - len(a) + len(w)), i + 1)) for a, _ in spec["aliases"])
                if inside:
                    continue
                concepts[name] = concepts[name] or w
                masked = mask(masked, i, i + len(w), CONCEPT_MASK)
    for phrase, col in spec["asks"]:
        i = masked.find(phrase)
        while i != -1:
            hits.append({"s": i, "e": i + len(phrase), "col": col, "how": "ask"})
            masked = mask(masked, i, i + len(phrase), MENTION_MASK)
            i = masked.find(phrase, i + len(phrase))
    for alias, col in spec["aliases"]:
        start, pat = 0, flex_pattern(alias)
        while True:
            if pat:
                m = pat.search(masked, start)
                if not m:
                    break
                i, j = m.start(), m.end()
            else:
                i = masked.find(alias, start)
                if i == -1:
                    break
                j = i + len(alias)
            start = j
            before, after = masked[i - 1:i], masked[j:j + 1]
            if is_latin(alias) and (re.match("[a-z0-9]", before) or re.match("[a-z0-9]", after)):
                continue                                  # 'age'가 'average' 안에서 걸리지 않게
            if len(alias) == 1 and re.match("[가-힣]", before):
                continue                                  # '키'가 '쿠키' 안에서 걸리지 않게
            hits.append({"s": i, "e": j, "col": col, "how": "name"})
            masked = mask(masked, i, j, MENTION_MASK)
    # 오타: 한글은 자모 단위로 한 조각(긴 이름은 두 조각)까지, 영어는 한두 글자까지 허용
    for m in re.finditer(r"[가-힣a-z/()]{2,}", masked):
        word = m.group()
        for p in sorted(NLU["particles"], key=len, reverse=True):
            if word.endswith(p) and len(word) - len(p) >= 2:
                word = word[:-len(p)]
                break
        if word in FUZZY_STOP:
            continue
        hangul = re.fullmatch("[가-힣]+", word) is not None
        best = None
        for alias, col in spec["aliases"]:
            a = alias.replace(" ", "")
            if hangul:
                if not re.fullmatch("[가-힣]+", a) or len(a) < 2 or abs(len(a) - len(word)) > 1:
                    continue
                ja, jw = jamo(a), jamo(word)
                if ja[0] != jw[0]:
                    continue                              # 첫 자음이 다르면 오타로 보지 않음 ('차이' ≠ '나이')
                d, limit = edit_distance(jw, ja), (1 if len(ja) <= 9 else 2)
            else:
                if not is_latin(word) or len(a) < 3 or abs(len(a) - len(word)) > 2:
                    continue
                d, limit = edit_distance(word, a), (2 if len(a) >= 6 else 1)
            if 0 < d <= limit and (best is None or d < best[0]):
                best = (d, alias, col)
        if best:
            s, e = m.start(), m.start() + len(word)
            hits.append({"s": s, "e": e, "col": best[2], "how": "fuzzy"})
            masked = mask(masked, s, e, MENTION_MASK)
            notes.append(f"'{word}' → {spec['label'][best[2]]}")
    hits.sort(key=lambda h: h["s"])
    return hits, masked, notes, concepts


FUZZY_STOP = set(NLU["fuzzy_stopwords"]) | set(NLU["predict_words"]) | set(NLU["context_words"]) \
    | set(NLU["particles"]) | {w for ws in NLU["concepts"].values() for w in ws} \
    | {w for it in NLU["intents"].values() for w in it["keywords"]} \
    | {w for ws in NLU["model_words"].values() for w in ws} | {w for ws in NLU["units"].values() for w in ws}


# ── 4. 숫자 묶기: 범위·목록·변화량 ─────────────────────
def _alt(words):
    return "|".join(re.escape(w) + ("(?![a-z])" if is_latin(w) and w.isalpha() else "")
                    for w in sorted(words, key=len, reverse=True))


UP_RE, DOWN_RE = _alt(NLU["up_words"]), _alt(NLU["down_words"])
FILLER = _alt(NLU["change_fillers"])
CHANGE_AFTER_NUM = re.compile(
    rf"^\s*(?P<fill>{FILLER})?\s*(?P<part>이|가|을|를|은|는|도)?\s*(?P<more>더\s*)?"
    rf"(?:(?P<up>{UP_RE})|(?P<down>{DOWN_RE}))")
STRONG_CHANGE = NLU["strong_change_words"]
PER_AFTER_NUM = re.compile(rf"^\s*(?:{_alt(NLU['per_words'])})")
CHANGE_AFTER_COL = re.compile(
    rf"^\s*(?:이|가|은|는|을|를|도|값이|수치가)?\s*(?:더\s*)?(?:(?P<up>{UP_RE})|(?P<down>{DOWN_RE}))")
RANGE_GAP = re.compile(rf"\s*(?:{_alt(NLU['range_words'])})\s*")
RANGE_END = re.compile(rf"^\s*(?:{_alt(NLU['range_end_words'])})")
STEP_AFTER = re.compile(rf"^\s*(?:{_alt(NLU['step_words'])})")
NUMBER_FIRST = re.compile(rf"^(?:{_alt(NLU['number_first_after'])})")
LIST_GAP = re.compile(rf"\s*(?:{_alt(NLU['list_words'])})\s*")
BY_AFTER = re.compile(rf"^\s*(?:{_alt(NLU['by_words'])})")


def change_of(text_after, after_col=False):
    m = (CHANGE_AFTER_COL if after_col else CHANGE_AFTER_NUM).match(text_after)
    if m and not after_col:
        word = m.group("up") or m.group("down")
        if not m.group("more") and not m.group("fill") and (
                word == "큰" or (m.group("part") and word not in STRONG_CHANGE)):
            m = None                                      # "180이 큰 편이야?"는 변화량이 아님
    if m:
        return -1 if m.group("down") else 1
    if not after_col and PER_AFTER_NUM.match(text_after):
        return 1
    return 0


def group_numbers(toks, masked):
    groups = []
    for tk in toks:
        if tk["ignore"]:
            continue
        tk["change"] = change_of(masked[tk["e"]:].replace(NUM_MASK, " "))
        prev = groups[-1] if groups else None
        if prev and not tk["change"] and not prev["change"]:
            gap = masked[prev["e"]:tk["s"]]
            ok_unit = not (prev["unit"] and tk["unit"] and prev["unit"] != tk["unit"])
            if ok_unit and NUM_MASK not in gap and MENTION_MASK not in gap:
                if (prev["kind"] == "single" and RANGE_GAP.fullmatch(gap)
                        and tk["v"] > prev["values"][0]):
                    prev.update(kind="range", e=tk["e"], unit=prev["unit"] or tk["unit"],
                                raws=prev["raws"] + [tk["raw"]])
                    prev["values"].append(tk["v"])
                    m = RANGE_END.match(masked[tk["e"]:])
                    if m:
                        prev["e"] = tk["e"] + m.end()
                    continue
                if prev["kind"] in ("single", "list") and LIST_GAP.fullmatch(gap):
                    prev.update(kind="list", e=tk["e"], unit=prev["unit"] or tk["unit"],
                                raws=prev["raws"] + [tk["raw"]])
                    prev["values"].append(tk["v"])
                    continue
        if (prev and prev["kind"] == "range" and "step" not in prev and not tk["change"]
                and re.fullmatch(r"[\s,]*", masked[prev["e"]:tk["s"]]) and STEP_AFTER.match(masked[tk["e"]:])):
            prev["step"] = tk["v"]                         # "160부터 180까지 5씩" / "10단위로"
            prev["e"] = tk["e"] + STEP_AFTER.match(masked[tk["e"]:]).end()
            continue
        groups.append({"s": tk["s"], "e": tk["e"], "values": [tk["v"]], "raws": [tk["raw"]],
                       "unit": tk["unit"], "kind": "single", "change": tk["change"]})
    return groups


def nice_steps(a, b, step=None):
    """160~180 → 160, 165, 170, 175, 180 (8칸 이하가 되는 깔끔한 간격, step을 주면 그 간격)"""
    span = b - a
    if not step or span / step > 40:
        k = math.floor(math.log10(span / 8))
        for m in (1, 2, 5, 10):
            step = m * 10 ** k
            if span / step <= 8 + 1e-9:
                break
    vals, i = [], 0
    while a + i * step <= b + 1e-9:
        vals.append(round(a + i * step, 10))
        i += 1
    if abs(vals[-1] - b) > 1e-9:
        vals.append(b)
    return vals, step


def data_steps(spec, col):
    """'키별' → 데이터 범위(148~197)를 깔끔한 간격으로: 150, 160, …, 190"""
    if col in spec["binary"]:
        return list(spec["binary"][col]), None
    lo, hi = spec["stats"][col]["min"], spec["stats"][col]["max"]
    span = hi - lo
    k = math.floor(math.log10(span / 8))
    for m in (1, 2, 5, 10):
        step = m * 10 ** k
        if span / step <= 8 + 1e-9:
            break
    start = math.ceil(lo / step - 1e-9) * step
    vals, i = [], 0
    while start + i * step <= hi + 1e-9:
        vals.append(round(start + i * step, 10))
        i += 1
    return vals, step


# ── 5. 데이터 고르기 ──────────────────────────────────
def col_for_unit(spec, unit):
    """단위로 열 찾기. 1.75m처럼 바꿔야 하면 배율도 돌려준다"""
    if not unit:
        return []
    out = [(c, 1.0) for c in spec["columns"] if spec["match_unit"].get(c) == unit]
    conv = NLU["unit_convert"].get(unit)
    if conv:
        out += [(c, conv["factor"]) for c in spec["columns"] if spec["match_unit"].get(c) == conv["to"]]
    return out


def concept_words(text):
    """질문에 나온 개념어: 비만, 남자/여자"""
    found = {name: next((w for w in words if has_word(text, w)), None) for name, words in NLU["concepts"].items()}
    return found


def bmi_class(bmi):
    for lim, name in NLU["bmi_levels"]:
        if lim is None or bmi < lim:
            return name


def score_dataset(spec, text, mentions, toks):
    s = 2 * len(mentions)
    cw = concept_words(text)
    roles = spec["roles"]
    if cw["obesity"] and ("bmi" in roles or ("height" in roles and "weight" in roles)):
        s += 1
    if (cw["male"] or cw["female"]) and "sex" in roles:
        s += 1
    s += sum(has_word(text, k) for k in spec["keywords"])
    s += sum(1 for tk in toks if col_for_unit(spec, tk["unit"]))
    strong = s
    plain = [tk for tk in toks if not tk["ignore"]]
    if not mentions and plain and len(plain) == len(spec["columns"]) - 1:
        s += 1
    return s, strong


def detect_intent(masked):
    scores = {}
    for name in NLU["intent_order"]:
        n = sum(has_word(masked, k) for k in NLU["intents"][name]["keywords"])
        if n:
            scores[name] = n
    if not scores and re.search(r"(?<![가-힣])식(?![가-힣]{2,})", masked):
        scores["equation"] = 1
    if scores:
        return max(NLU["intent_order"], key=lambda n: (scores.get(n, 0), -NLU["intent_order"].index(n))), False
    clean = re.sub(r"[#@]", "", masked).strip()
    best, best_s = None, 0.0
    for name in NLU["intent_order"]:
        for ex in NLU["intents"][name]["examples"]:
            s = dice(clean, ex)
            if s > best_s:
                best, best_s = name, s
    if best_s >= NLU["fuzzy_intent_threshold"]:
        return best, True
    return None, False


def model_filter(masked):
    want = []
    for name, words in NLU["model_words"].items():
        for w in words:
            if not has_word(masked, w):
                continue
            if name == "선형회귀" and w == "선형" and masked.count("선형") == masked.count("비선형"):
                continue                                  # '비선형' 안의 '선형'
            want.append(name)
            break
    if "비선형" in masked:
        want += [n for n in NLU["model_words"] if n.startswith("비선형") and n not in want]
    return want


# ── 6. 이해하기 ───────────────────────────────────────
CARRY = ("predict", "check", "sweep", "effect", "obesity")


def understand(text, specs, last=None, prev=None):
    """질문 → 계획(dict).
    specs: 데이터별 설명(chat.Dataset.spec), last: 직전에 쓴 데이터 제목, prev: 직전 질문의 계획"""
    stay = []
    if prev and prev.get("intent") in CARRY and prev.get("dataset") == last:
        stay = list(prev["values"]) + ([prev["sweep"]["col"]] if prev.get("sweep") else []) \
            + ([prev["effect"]["col"]] if prev.get("effect") else [])
    plan = _core(text, specs, last, stay)
    ctx = plan.pop("_ctx")
    spec = next(sp for sp in specs if sp["title"] == plan["dataset"])
    if prev and prev.get("dataset") == plan["dataset"] and prev.get("intent") in CARRY:
        _carry(plan, prev, spec, ctx)
    _concepts(plan, spec, ctx)
    if plan["unused"] and not plan["values"] and not plan["sweep"] and not plan["effect"]:
        _guess_numbers(plan, specs, ctx)
    if plan["intent"] == "unknown":
        _clarify(plan, specs, ctx)
    return plan


def resolve(text, prev):
    """짐작한 선택지에 번호로 답하면('2', '두 번째') 그 질문 문장을 돌려준다. 아니면 None"""
    if not prev or prev.get("intent") != "clarify" or not prev.get("options"):
        return None
    t = unicodedata.normalize("NFKC", text).strip().lower().rstrip(".!?요")
    for i, words in enumerate(NLU["choice_words"]):
        if i < len(prev["options"]) and t in words:
            return prev["options"][i]
    return None


def half_up(v):
    return math.floor(v + 0.5)


def _question_for(spec, values):
    """값 dict → 시스템이 그대로 다시 알아듣는 질문 문장"""
    cols = spec["columns"]
    if len(cols) == 2 and len(values) == 1:
        target = next(c for c in cols if c not in values)
    else:
        target = spec["target"]
    if target in values:
        return f"{_given(spec, values)}이면?"
    return f"{_given(spec, values)}일 때 {spec['label'][target]} 예측해줘"


def _guess_numbers(plan, specs, ctx):
    """열 이름 없는 숫자: 각 열의 분포에 얼마나 잘 맞는지(-z²/2 합)로 어느 열의 값인지 짐작.
    가장 그럴듯한 해석이 확실히 앞서면 그대로 답하고, 비슷하면 상위 3개를 선택지로"""
    nums = [v for gr in ctx["free"] for v in gr["values"]]
    if not nums or len(nums) > NLU["guess_max_numbers"] or any(gr["kind"] == "range" for gr in ctx["free"]):
        return
    cands = []
    for si, sp in enumerate(specs):
        cols = sp["columns"]

        def fit(c, v):
            st = sp["stats"][c]
            if c in sp["binary"]:                       # 1·2 같은 열은 숫자가 여럿일 때만 ('1'만 쓰면 짐작 안 함)
                return 0.0 if v in sp["binary"][c] and len(nums) > 1 else None
            span = st["max"] - st["min"]
            if not st["min"] - span / 10 <= v <= st["max"] + span / 10:
                return None
            z = (v - st["mean"]) / st["std"]
            return -z * z / 2

        def walk(i, used, score, picked):
            if i == len(nums):
                if len(cols) > 2 and picked == [sp["target"]]:
                    return                                # 예측 목표 하나만 채우는 해석은 답할 게 없어서 뺌
                bonus = NLU["guess_complete_bonus"] if len(nums) >= len(cols) - 1 else 0.0
                cands.append((score + bonus, si, [cols.index(c) for c in picked], sp, list(picked)))
                return
            for c in cols:
                if c in used:
                    continue
                f = fit(c, nums[i])
                if f is not None:
                    walk(i + 1, used | {c}, score + f, picked + [c])
        walk(0, set(), 0.0, [])
    if not cands:
        return
    cands.sort(key=lambda x: (-x[0], x[1], x[2]))
    best = cands[0]
    sp = best[3]
    raws = ", ".join(fmt(v, 4) for v in nums)
    if len(cands) == 1 or best[0] - cands[1][0] >= NLU["guess_gap"]:
        values = dict(zip(best[4], nums))
        cols = sp["columns"]
        target = next(c for c in cols if c not in values) if len(cols) == 2 and len(values) == 1 else sp["target"]
        plan.update(dataset=sp["title"], values=values, target=target, unused=[], sweep=None, effect=None,
                    intent="check" if target in values else "predict", focus=best[4][-1])
        plan["notes"].append(f"숫자만 보고 데이터 범위로 짐작: {_given(sp, values)}")
        return
    options = []
    for c in cands:
        q = _question_for(c[3], dict(zip(c[4], nums)))
        if q not in options:
            options.append(q)
        if len(options) == 3:
            break
    plan.update(dataset=sp["title"], intent="clarify", options=options, reason="numbers", unused=[], values={})
    plan["notes"].append(f"숫자 {raws}만으로는 어느 값인지 정할 수 없어 데이터 범위로 짐작")


def _clarify(plan, specs, ctx):
    """알아듣지 못한 질문: 지금 데이터로 가장 그럴듯한 질문 3개를 짐작해 고르게 한다"""
    sp = next(s for s in specs if s["title"] == plan["dataset"])
    t = sp["target"]
    f0 = next(c for c in sp["columns"] if c != t)
    lt = sp["label"][t]
    predict_q = f"{sp['label'][f0]} {fmt(half_up(sp['stats'][f0]['median']), 4)}일 때 {lt} 예측해줘"
    roles = sp["roles"]
    obese_ok = "bmi" in roles or ("height" in roles and "weight" in roles)
    if ctx["eval"]:
        options = (["비만이야?"] if obese_ok else []) + [f"{lt} 평균이랑 범위 알려줘", predict_q]
    else:
        options = [predict_q, f"{lt}에 영향이 큰 건?" if len(sp["columns"]) > 2 else "어떤 모델이 제일 정확해?",
                   "어떤 데이터가 있어?"]
    plan.update(intent="clarify", options=options[:3], reason="vague")


def _given(spec, values):
    return ", ".join(f"{spec['label'][c]} {fmt(v, 4)}{spec['unit'][c]}" for c, v in values.items())


def _in_range(spec, col, v):
    st = spec["stats"][col]
    span = st["max"] - st["min"]
    return st["min"] - span / 4 <= v <= st["max"] + span / 4


def _carry(plan, prev, spec, ctx):
    """앞 질문 잇기: '그럼 35면?', '그 환자의 혈당은?', '신경망으로는?', '그럼 혈압 120이면?'"""
    carry = dict(prev["values"])
    notes = plan["notes"]
    focus = prev.get("focus")
    no_values = not plan["values"] and not plan["sweep"] and not plan["effect"]

    # (a) 열 이름 없는 숫자 하나(또는 범위·목록 하나) → 앞 질문에서 쓴 열 중 값이 맞는 열로 (방금 바꾼 열 먼저)
    if len(ctx["free"]) == 1 and not plan["values"] and \
            (ctx["free"][0]["kind"] != "single" or len(ctx["free"][0]["values"]) == 1):
        cands = list(dict.fromkeys(([focus] if focus else []) + list(carry)
                                   + ([prev["sweep"]["col"]] if prev.get("sweep") else [])
                                   + ([prev["effect"]["col"]] if prev.get("effect") else [])))
        new, sweep, used = {}, None, []
        for gr in ctx["free"]:
            v0 = gr["values"][0]
            col = next((c for c in cands if c not in new and c != plan["target"] and _in_range(spec, c, v0)), None)
            if col is None:
                continue
            used += gr["raws"]
            if gr["kind"] == "single":
                new[col] = v0
            elif sweep is None:
                vals = nice_steps(gr["values"][0], gr["values"][1], gr.get("step"))[0] if gr["kind"] == "range" else gr["values"][:20]
                sweep = {"col": col, "values": vals}
                new[col] = None
        if new:
            plan["unused"] = [r for r in plan["unused"] if r not in used]
            kept = {c: v for c, v in carry.items() if c not in new}
            target = ctx["ask_target"] or prev["target"]
            kept.pop(target, None) if prev["intent"] != "check" else None
            vals = dict(kept, **{c: v for c, v in new.items() if v is not None})
            plan.update(values=vals, sweep=sweep, effect=None, target=target,
                        focus=next(iter(new)))
            plan["intent"] = "sweep" if sweep else "check" if target in vals else "predict"
            if kept:
                notes.append("앞 질문에 이어서: " + _given(spec, kept))
            return
    # (b) 값 없이 열 이름만: "그 환자의 혈당은?", "그럼 비만이야?"
    concept = ctx["obesity"] or ctx["sex_word"]
    if no_values and (ctx["asked"] or concept) and ctx["kw_intent"] is None and carry:
        target = ctx["asked"][-1] if ctx["asked"] else prev["target"]
        vals = {c: v for c, v in carry.items() if c != target or not ctx["asked"]}
        if vals:
            plan.update(values=vals, target=target, intent="check" if target in vals else "predict",
                        focus=prev.get("focus"))
            notes.append("앞 질문에 이어서: " + _given(spec, vals))
        return
    # (c) 값이 있고 '그럼' 같은 말이 있거나 열 이름 없이 단위로만 준 값이면("50살이면?")
    #     앞 질문의 나머지 값과 합치기: "그럼 혈압 120이면?"
    if (ctx["ctxword"] or ctx["mentions"] == 0) and plan["intent"] in CARRY and \
            (plan["values"] or plan["sweep"] or plan["effect"]):
        newcols = set(plan["values"]) | ({plan["sweep"]["col"]} if plan["sweep"] else set()) \
            | ({plan["effect"]["col"]} if plan["effect"] else set())
        keep_target = prev["intent"] == "check" and prev["target"] == plan["target"]
        kept = {c: v for c, v in carry.items() if c not in newcols and (keep_target or c != plan["target"])}
        if kept:
            plan["values"] = dict(kept, **plan["values"])
            if keep_target and plan["target"] in plan["values"] and plan["intent"] == "predict":
                plan["intent"] = "check"
            notes.append("앞 질문에 이어서: " + _given(spec, kept))
        return
    # (d) 모델 이름만: "신경망으로는?", "선형회귀로 하면?"
    if no_values and plan["intent"] == "unknown" and plan["models"] and not ctx["asked"]:
        for k in ("intent", "target", "values", "sweep", "effect", "focus"):
            plan[k] = prev[k]
        plan["pick"] = plan["models"][0]
        notes.append(f"앞 질문을 {plan['pick']} 기준으로 다시 계산")


def _concepts(plan, spec, ctx):
    roles, notes = spec["roles"], plan["notes"]
    word = ctx["sex_word"]
    if word:
        sc = roles.get("sex")
        if not sc:
            notes.append(f"이 데이터에는 성별 정보가 없어 '{word}'은(는) 반영하지 못함")
        elif sc not in plan["values"] and not (plan["sweep"] and plan["sweep"]["col"] == sc):
            lo, hi = spec["binary"][sc]
            notes.append(f"{spec['label'][sc]}이 {fmt(lo)}·{fmt(hi)}로만 기록돼 있어 어느 쪽이 '{word}'인지 알 수 없어 둘 다 계산함")
            if plan["intent"] in ("predict", "check", "need_input", "stats", "unknown", "need_target"):
                tgt = plan["target"] if plan["target"] and plan["target"] != sc else spec["target"]
                plan["values"].pop(tgt, None)
                plan.update(intent="sweep", target=tgt, sweep={"col": sc, "values": [lo, hi]})
    if ctx["obesity"] and ("bmi" in roles or ("height" in roles and "weight" in roles)) and \
            plan["intent"] in ("predict", "check", "stats", "unknown", "need_input", "need_target", "help", "data"):
        plan["intent"] = "obesity"


def _core(text, specs, last=None, stay=()):
    t, notes = normalize(text)
    toks = find_numbers(t)
    masked0 = t
    for tk in toks:
        if not tk["ignore"]:
            masked0 = mask(masked0, tk["s"], tk["e"], NUM_MASK)

    # 데이터 고르기: 열 이름·키워드·단위가 많이 맞는 쪽, 같으면 직전 데이터
    order = sorted(specs, key=lambda sp: sp["title"] != last)
    found = []
    for sp in order:
        mentions, masked, fz, cw = find_mentions(sp, masked0)
        sc, strong = score_dataset(sp, t, mentions, toks)
        found.append((sc, strong, sp, mentions, masked, fz, cw))
    # 앞 질문에 이어지는 짧은 질문("그럼 35면?")은 열 이름·단위·키워드가 없고
    # 숫자가 앞 질문에서 쓴 열의 범위에 맞으면 직전 데이터에 머묾
    plain = [tk["v"] for tk in toks if not tk["ignore"] and not tk["unit"]]
    if stay and all(f[1] == 0 for f in found) and \
            all(any(_in_range(found[0][2], c, v) for c in stay) for v in plain):
        best = found[0]
    else:
        best = found[0]
        for f in found[1:]:
            if f[0] > best[0]:
                best = f
    score, _, spec, mentions, masked, fz, cw = best
    notes += fz
    explicit = score > 0
    plan = {"intent": None, "dataset": spec["title"], "target": None, "values": {}, "sweep": None,
            "effect": None, "columns": [], "models": [], "notes": notes, "unused": [],
            "explicit": explicit, "focus": None, "pick": None, "options": [], "reason": None}

    groups = group_numbers(toks, masked)
    assigned, effect = {}, None
    label = spec["label"]

    def put(col, g, factor, how):
        vals = [v * factor for v in g["values"]]
        if factor != 1:
            notes.append(f"{', '.join(g['raws'])} → {', '.join(fmt(v) + spec['unit'][col] for v in vals)}")
        if how == "number-first":
            notes.append(f"{', '.join(g['raws'])} → {label[col]}")
        assigned[col] = dict(g, values=vals)

    # (1) 단위로 열이 하나로 정해지는 숫자
    def plausible(col, factor, v):
        st = spec["stats"][col]
        span = st["max"] - st["min"]
        inside = lambda x: st["min"] - span <= x <= st["max"] + span
        return factor if factor == 1 or inside(v * factor) or not inside(v) else 1.0

    for g in groups:
        if g["change"]:
            continue
        cands = col_for_unit(spec, g["unit"])
        if len(cands) == 1 and cands[0][0] not in assigned:
            col = cands[0][0]
            put(col, g, plausible(col, cands[0][1], g["values"][0]), "unit")
            g["used"] = True
    # (2) 열 이름 바로 뒤의 숫자: "키 175", "혈당 수치가 87"
    names = [m for m in mentions if m["how"] != "ask"]
    for g in groups:
        if g.get("used") or g["change"]:
            continue
        prev = [m for m in names if m["e"] <= g["s"]]
        if prev:
            m = prev[-1]
            gap = masked[m["e"]:g["s"]]
            if len(gap) <= MAX_GAP and NUM_MASK not in gap and m["col"] not in assigned:
                put(m["col"], g, 1, "after")
                g["used"] = True
            elif (len(gap) <= MAX_GAP and NUM_MASK not in gap and m["col"] in assigned
                  and assigned[m["col"]]["e"] <= m["s"] and assigned[m["col"]]["kind"] in ("single", "list")):
                a = assigned[m["col"]]
                a.update(kind="list", values=a["values"] + g["values"], raws=a["raws"] + g["raws"], e=g["e"])
                g["used"] = True
    # (3) 숫자 바로 뒤의 열 이름: "175 키인 사람"
    for g in groups:
        if g.get("used") or g["change"]:
            continue
        nxt = [m for m in names if m["s"] >= g["e"]]
        if (nxt and re.fullmatch(r"\s*", masked[g["e"]:nxt[0]["s"]]) and nxt[0]["col"] not in assigned
                and NUMBER_FIRST.match(masked[nxt[0]["e"]:])):
            put(nxt[0]["col"], g, 1, "number-first")
            g["used"] = True

    # (3-2) "키 160인 사람과 180인 사람": 앞 숫자와 같은 열로 묶기
    for g in groups:
        if g.get("used") or g["change"]:
            continue
        before = [(c, a) for c, a in assigned.items() if a["e"] <= g["s"] and a["kind"] in ("single", "list")]
        if before:
            c, a = max(before, key=lambda ca: ca[1]["e"])
            gap = masked[a["e"]:g["s"]]
            unit_ok = not g["unit"] or any(cc == c for cc, _ in col_for_unit(spec, g["unit"]))
            if len(gap) <= 12 and NUM_MASK not in gap and MENTION_MASK not in gap and unit_ok:
                a["kind"] = "list"
                a["values"] = a["values"] + g["values"]
                a["raws"] = a["raws"] + g["raws"]
                a["e"] = g["e"]
                g["used"] = True

    # 변화량: "키가 10cm 크면", "1cm당", "BMI가 높으면"
    for g in groups:
        if not g["change"] or effect:
            continue
        cands = col_for_unit(spec, g["unit"])
        col, factor = (cands[0] if len(cands) == 1 else (None, 1.0))
        if col is not None:
            factor = plausible(col, factor, g["values"][0])
        if col is None:
            prev = [m for m in names if m["e"] <= g["s"]]
            if prev and len(masked[prev[-1]["e"]:g["s"]]) <= MAX_GAP:
                col = prev[-1]["col"]
        if col is not None:
            effect = {"col": col, "delta": g["change"] * g["values"][0] * factor, "auto": False}
            g["used"] = True
    if not effect:
        for m in names:
            sign = change_of(masked[m["e"]:], after_col=True)
            if sign and m["col"] not in assigned:
                std = spec["stats"][m["col"]]["std"]
                effect = {"col": m["col"], "delta": sign * nice2(std), "auto": True}
                break

    # "키별 몸무게", "BMI에 따른 진행도" → 그 열을 데이터 범위 전체로 나눠 표로
    by = None
    if not effect and all(a["kind"] == "single" for a in assigned.values()):
        by = next((m["col"] for m in names if m["col"] not in assigned and BY_AFTER.match(masked[m["e"]:])), None)

    # 무엇을 물었나: 값이 없는 열 이름 ('몇 kg' 같은 물음 표현이 있으면 그쪽 우선)
    effect_col = effect["col"] if effect else by
    asked = []
    for m in mentions:
        if m["col"] not in assigned and m["col"] != effect_col and m["col"] not in asked:
            asked.append(m["col"])
    ask_cols = [m["col"] for m in mentions if m["how"] == "ask" and m["col"] not in assigned and m["col"] != effect_col]
    target_hint = [m["col"] for m in mentions
                   if any(masked[m["e"]:].startswith(p) for p in NLU["target_particles"])]

    default = spec["target"]
    if ask_cols:
        target = ask_cols[-1]
    elif asked:
        target = asked[-1]
    elif default not in assigned and default != effect_col:
        target = default
    elif len(spec["columns"]) == 2:
        target = next((c for c in spec["columns"] if c not in assigned and c != effect_col), None)
    else:
        target = None

    # (4) 열이 안 정해진 숫자: 남은 입력 열이 하나뿐이거나, 개수가 딱 맞으면 열 순서대로
    free = [g for g in groups if not g.get("used")]
    remaining = [c for c in spec["columns"] if c != target and c not in assigned and c != effect_col]
    def near(col, vals):
        st = spec["stats"][col]
        span = st["max"] - st["min"]
        return all(st["min"] - span / 2 <= v <= st["max"] + span / 2 for v in vals)

    if free and target:
        if len(free) == 1 and len(remaining) == 1 and near(remaining[0], free[0]["values"]):
            put(remaining[0], free[0], 1, "only")
            notes.append(f"{', '.join(free[0]['raws'])} → {label[remaining[0]]}")
            free = []
        elif not assigned and len(remaining) > 1:
            flat = [g for g in free if g["kind"] == "single"]
            seq = flat if len(flat) == len(free) else (free if len(free) == 1 and free[0]["kind"] == "list" else [])
            vals = [v for g in seq for v in g["values"]]
            if len(vals) == len(remaining) and all(near(c, [v]) for c, v in zip(remaining, vals)):
                for c, v in zip(remaining, vals):
                    assigned[c] = {"values": [v], "kind": "single", "raws": [fmt(v)]}
                notes.append(f"숫자 {len(vals)}개를 열 순서({', '.join(label[c] for c in remaining)})대로 읽음")
                free = []
    plan["unused"] = [r for g in free for r in g["raws"]]

    plan["values"] = {c: g["values"][0] for c, g in assigned.items() if g["kind"] == "single"}
    multi = [(c, g) for c, g in assigned.items() if g["kind"] != "single"]
    if multi:
        c, g = multi[0]
        if g["kind"] == "range":
            vals, step = nice_steps(g["values"][0], g["values"][1], g.get("step"))
            notes.append(f"{fmt(g['values'][0])}~{fmt(g['values'][1])}를 {fmt(step)} 간격으로 나눔")
        else:
            vals = g["values"][:20]
        plan["sweep"] = {"col": c, "values": vals}
        for c2, g2 in multi[1:]:
            plan["values"][c2] = g2["values"][0]
    elif by:
        vals, step = data_steps(spec, by)
        plan["sweep"] = {"col": by, "values": vals}
        if step:
            notes.append(f"{label[by]} 데이터 범위 {fmt(spec['stats'][by]['min'])}~{fmt(spec['stats'][by]['max'])}를 "
                         f"{fmt(step)} 간격으로 나눔")
    plan["effect"] = effect
    plan["models"] = model_filter(masked)
    placed = [(a["e"], c) for c, a in assigned.items() if "e" in a]
    plan["focus"] = (max(placed)[1] if placed else plan["sweep"]["col"] if plan["sweep"]
                     else effect["col"] if effect else None)

    intent, _ = detect_intent(masked)
    has_values = bool(assigned) or effect is not None or by is not None
    plan["_ctx"] = {"free": [{"values": g["values"], "kind": g["kind"], "raws": g["raws"], "step": g.get("step")}
                             for g in free],
                    "ctxword": any(has_word(masked, w) for w in NLU["context_words"]),
                    "kw_intent": intent, "asked": asked, "ask_target": ask_cols[-1] if ask_cols else None,
                    "mentions": len(mentions),
                    "obesity": cw["obesity"] is not None, "sex_word": cw["male"] or cw["female"],
                    "eval": any(has_word(masked, w) for w in NLU["eval_words"])}

    # 영향: 목표 열은 '에' 같은 조사가 붙은 열 → 기본 목표 순서로 정함
    if intent == "importance" and not has_values:
        tgt = (target_hint[-1] if target_hint else
               asked[-1] if len(asked) >= 2 else
               asked[0] if asked and asked[0] == default else default)
        feat = [c for c in asked if c != tgt]
        plan["target"] = tgt
        if feat:
            std = spec["stats"][feat[0]]["std"]
            plan["intent"], plan["effect"] = "effect", {"col": feat[0], "delta": nice2(std), "auto": True}
        else:
            plan["intent"] = "importance"
        return plan

    if has_values:
        if target is None and default in plan["values"] and len(plan["values"]) > 1:
            target = default
        if target is None:
            plan["intent"], plan["target"] = "need_target", None
        elif plan["sweep"]:
            plan["intent"], plan["target"] = "sweep", target
        elif effect:
            plan["intent"], plan["target"] = "effect", target
        elif target in plan["values"]:
            plan["intent"], plan["target"] = "check", target
        else:
            plan["intent"], plan["target"] = "predict", target
        return plan

    plan["target"] = target or default
    if intent in ("compare", "equation", "stats", "data", "help", "about", "thanks", "greet"):
        plan["intent"] = intent
        plan["columns"] = asked
        return plan
    predict_word = any(has_word(masked, w) for w in NLU["predict_words"])
    if ask_cols or (asked and predict_word) or (not mentions and not groups and predict_word):
        plan["intent"] = "need_input"
        return plan
    if asked:
        plan["intent"], plan["columns"] = "stats", asked
        return plan
    plan["intent"] = "unknown"
    return plan
