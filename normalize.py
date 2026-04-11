"""Korean text normalizer for ASR evaluation (CER comparison).

Handles number→Korean conversion, special symbol mapping, hanja→hangul,
and punctuation removal for fair CER comparison.

Reference: CoreaSpeech N2gk (https://github.com/CoreaSpeech/CoreaSpeech)
"""
from __future__ import annotations

import re


# ── Number → Korean conversion ──────────────────────────────────────────────

NUM_KOR = ["", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]
UNIT_SMALL = ["", "십", "백", "천"]
UNIT_LARGE = ["", "만", "억", "조", "경"]
NEVER_SKIP_ONE = {"억", "조", "경"}

GOOYO_SIP = {
    10: "열", 20: "스물", 30: "서른", 40: "마흔",
    50: "쉰", 60: "예순", 70: "일흔", 80: "여든", 90: "아흔",
}
BASIC_NATIVE = {
    1: ("하나", "한"), 2: ("둘", "두"), 3: ("셋", "세"), 4: ("넷", "네"),
    5: ("다섯", "다섯"), 6: ("여섯", "여섯"), 7: ("일곱", "일곱"),
    8: ("여덟", "여덟"), 9: ("아홉", "아홉"),
}
GOOYO_PREFIX_TENS = {20: "스무"}

# 고유어 수사를 쓰는 단위
NATIVE_UNITS = {
    "명", "사람", "마리", "번째", "시", "배", "방", "가구", "게임", "건", "세트",
    "개", "가지", "개비", "잔", "번", "장", "병", "권", "벌", "곳", "시간",
    "척", "차례", "바퀴", "경기", "골", "살", "연세", "춘추",
    "달", "글자",
}

# 한자어 수사를 쓰는 단위
HANJA_UNITS = {
    "초", "분", "일", "주", "개월", "월", "년",
    "점", "포인트", "퍼센트", "레벨", "점수", "등급", "등", "개국", "볼트",
    "원", "달러", "유로", "엔", "페소", "배럴",
    "회", "차", "기", "호", "페이지",
    "도",
    # 교육/서열
    "학년", "학기", "학점", "학번", "교시", "반",
    "급", "단계", "위", "형",
    # 군사/조직
    "사단",
    # 건물/공간
    "층",
    # 세대/시대
    "세기", "대", "세대",
    # 경쟁/스포츠
    "라운드",
    # 금액 큰 단위
    "만원", "만명", "만",
    # 외래 단위
    "피트", "파운드", "마일", "인치", "헥타르",
    # 인승/인극
    "인승", "인극",
}

# 단위명 변환 (영문 단위 → 한글)
UNIT_NAME_MAP = {
    "kg": "킬로그램", "Kg": "킬로그램", "g": "그램", "mg": "밀리그램",
    "t": "톤", "T": "톤", "l": "리터", "L": "리터", "ml": "밀리리터",
    "cm": "센티미터", "mm": "밀리미터", "m": "미터", "km": "킬로미터",
    "mi": "마일",
}

# 특수기호 → 한글
SPECIAL_SYMBOL_MAP = {
    "％": "퍼센트", "%": "퍼센트",
    "%p": "퍼센트포인트", "% p": "퍼센트포인트",
    "&": "앤", "#": "샵", "@": "앳",
    "+": "플러스", "±": "플러스마이너스",
    "㎝": "센티미터", "㎜": "밀리미터", "㎏": "킬로그램",
    "㎖": "밀리리터", "℃": "도", "㎞": "킬로미터", "㎎": "밀리그램",
    "㎡": "제곱미터", "㎥": "세제곱미터",
    "～": "~", "ｍ": "미터",
    "°C": "도", "°c": "도",
}

# 예외 케이스
EXCEPTION_CASES = {
    r"\b20\s?살\b": "스무 살",
    r"\b1\s?등\b": "일 등",
    r"(?<!\d)(0?6)\s*월": "유월",
    r"(?<!\d)(10)\s*월": "시월",
}


def _to_gooyo(num: int, prefix: bool = False) -> str:
    """고유어 수사 변환 (1~99)."""
    if num <= 0:
        return "영"
    if num <= 9:
        base = BASIC_NATIVE.get(num)
        return (base[1] if prefix else base[0]) if base else "영"
    if num == 10:
        return "열"
    if num < 100:
        tens = (num // 10) * 10
        ones = num % 10
        if prefix and ones == 0 and tens in GOOYO_PREFIX_TENS:
            return GOOYO_PREFIX_TENS[tens]
        tens_str = GOOYO_SIP.get(tens, "")
        return tens_str + (_to_gooyo(ones, prefix=prefix) if ones else "")
    # 100 이상은 한자어로 fallback
    return _to_hanja(num)


def _convert_small_unit(chunk: str) -> str:
    """4자리 이하 숫자 청크를 한자어 수사로 변환."""
    result = ""
    length = len(chunk)
    for i, ch in enumerate(chunk):
        digit = int(ch)
        if digit == 0:
            continue
        unit = UNIT_SMALL[length - i - 1]
        if digit == 1 and unit:
            result += unit
        else:
            result += NUM_KOR[digit] + unit
    return result


def _to_hanja(num, natural: bool = True) -> str:
    """한자어 수사 변환."""
    if isinstance(num, float):
        int_part = int(num)
        frac_str = str(num).split(".")[1]
        int_kor = _to_hanja(int_part, natural)
        frac_kor = "".join(
            NUM_KOR[int(ch)] if ch != "0" else "영" for ch in frac_str
        )
        return f"{int_kor}점{frac_kor}"

    if isinstance(num, str):
        try:
            num = float(num) if "." in num else int(num)
            return _to_hanja(num, natural)
        except ValueError:
            return num

    if num == 0:
        return "영"
    if num < 0:
        return "마이너스 " + _to_hanja(-num, natural)

    s = str(num)
    chunks = [s[max(i - 4, 0):i] for i in range(len(s), 0, -4)][::-1]
    if len(chunks) > 5:
        return str(num)

    result = ""
    for i, chunk in enumerate(chunks):
        if int(chunk) == 0:
            continue
        part = _convert_small_unit(chunk.zfill(4))
        unit = UNIT_LARGE[len(chunks) - i - 1]
        if part == "일" and unit:
            if natural and unit not in NEVER_SKIP_ONE:
                part = ""
        result += part + unit
    return result


def _n2gk_with_unit(num: int, unit: str) -> str:
    """숫자+단위를 인식하여 적절한 수사(고유어/한자어)로 변환."""
    # 단위명 변환 (영문 → 한글)
    display_unit = UNIT_NAME_MAP.get(unit, unit)

    if unit in NATIVE_UNITS and 1 <= num <= 99:
        return _to_gooyo(num, prefix=True) + display_unit
    else:
        return _to_hanja(num) + display_unit


# ── All unit strings sorted by length (longest first for regex matching) ──
_ALL_UNITS = sorted(
    list(NATIVE_UNITS) + list(HANJA_UNITS) + list(UNIT_NAME_MAP.keys()),
    key=len, reverse=True,
)
# 영문 단위는 뒤에 word boundary 추가 (m이 mi를 매칭하는 것 방지)
_UNITS_PATTERN = "|".join(
    re.escape(u) + r"(?![a-zA-Z])" if re.match(r"^[a-zA-Z]+$", u) else re.escape(u)
    for u in _ALL_UNITS
)


# 통화 기호 (숫자 앞에 오는 prefix 통화 → 숫자 뒤로 이동)
CURRENCY_PREFIX_MAP = {
    "$": "달러",
    "€": "유로",
    "£": "파운드",
    "¥": "엔",
    "₩": "원",
}
_CURRENCY_SYMBOLS = "|".join(re.escape(s) for s in CURRENCY_PREFIX_MAP)


def _convert_currency_prefix(text: str) -> str:
    """통화 기호+숫자 → 숫자+통화명 변환 ($5 → 5달러, €100 → 100유로)."""
    pattern = rf"({_CURRENCY_SYMBOLS})\s*(\d+(?:[.,]\d+)*)"

    def replacer(m):
        currency = CURRENCY_PREFIX_MAP[m.group(1)]
        num = m.group(2)
        return f"{num}{currency}"

    return re.sub(pattern, replacer, text)


def _convert_phone_numbers(text: str) -> str:
    """전화번호 패턴 변환 (010-1234-5678 → 공일공-일이삼사-오육칠팔)."""
    DIGIT_KOR = ["공", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]

    def _digits_to_kr(s: str) -> str:
        return "".join(DIGIT_KOR[int(d)] for d in s)

    # 하이픈 포함
    text = re.sub(
        r"(?<!\d)(\d{2,3})-(\d{3,4})-(\d{4})(?!\d)",
        lambda m: f"{_digits_to_kr(m[1])}-{_digits_to_kr(m[2])}-{_digits_to_kr(m[3])}",
        text,
    )
    # 11자리 연속
    text = re.sub(
        r"(?<!\d)(\d{11})(?!\d)",
        lambda m: f"{_digits_to_kr(m[1][:3])}-{_digits_to_kr(m[1][3:7])}-{_digits_to_kr(m[1][7:])}",
        text,
    )
    return text


def _convert_range_with_units(text: str) -> str:
    """범위 패턴 변환 (1~3마리 → 한마리에서세마리, 10–60분 → 십분에서육십분)."""
    range_sep = r"[~\u2013\u2014]"  # ~, –(en-dash), —(em-dash)
    pattern = rf"(\d+(?:\.\d+)?)\s*{range_sep}\s*(\d+(?:\.\d+)?)\s*({_UNITS_PATTERN})"

    def replacer(m):
        try:
            left = float(m.group(1)) if "." in m.group(1) else int(m.group(1))
            right = float(m.group(2)) if "." in m.group(2) else int(m.group(2))
            unit = m.group(3)
            l_str = _n2gk_with_unit(left, unit)
            r_str = _n2gk_with_unit(right, unit)
            return f"{l_str}에서{r_str}"
        except (ValueError, OverflowError):
            return m.group(0)

    return re.sub(pattern, replacer, text)


def _convert_numbers_with_units(text: str) -> str:
    """숫자+단위 패턴 변환 (5개 → 다섯개, 100원 → 백원)."""
    pattern = rf"(\d{{1,3}}(?:,\d{{3}})*|\d+(?:\.\d+)?)\s?({_UNITS_PATTERN})"

    def replacer(m):
        raw = m.group(1).replace(",", "")
        word = m.group(2)
        try:
            num = float(raw) if "." in raw else int(raw)
            return _n2gk_with_unit(num, word)
        except (ValueError, OverflowError):
            return m.group(0)

    return re.sub(pattern, replacer, text)


def _convert_float_numbers(text: str) -> str:
    """소수점 숫자 변환 (3.14 → 삼점일사)."""
    def replacer(m):
        try:
            num = float(m.group(1))
            return _to_hanja(num)
        except (ValueError, OverflowError):
            return m.group(1)

    return re.sub(r"(\d+\.\d+)", replacer, text)


def _convert_pure_numbers(text: str) -> str:
    """남은 모든 숫자 변환 (1234 → 천이백삼십사).

    _convert_numbers_with_units 이후 호출되므로 단위 붙은 숫자는 이미 변환됨.
    한글 인접 숫자도 변환 (예: 200은 → 이백은, 1학년 → 일학년).
    """
    pattern = r"(\d{1,3}(?:,\d{3})+|\d+)"

    def replacer(m):
        try:
            num = int(m.group(1).replace(",", ""))
            return _to_hanja(num)
        except (ValueError, OverflowError):
            return m.group(0)

    return re.sub(pattern, replacer, text)


def _apply_exceptions(text: str) -> str:
    """예외 케이스 처리 (스무살, 유월, 시월 등)."""
    for pattern, replacement in EXCEPTION_CASES.items():
        text = re.sub(pattern, replacement, text)
    return text


def _apply_special_symbols(text: str) -> str:
    """특수기호 → 한글 변환."""
    # 긴 패턴 먼저 매칭하기 위해 길이 역순 정렬
    for symbol in sorted(SPECIAL_SYMBOL_MAP, key=len, reverse=True):
        text = text.replace(symbol, SPECIAL_SYMBOL_MAP[symbol])
    return text


def _remove_punctuation(text: str) -> str:
    """구두점/괄호/기타 기호 제거."""
    # 괄호 안 내용 제거
    text = re.sub(r"\([^)]*\)", "", text)
    # 한글(완성형+자모), 라틴문자(악센트 포함), 숫자, 공백만 유지
    # \u00C0-\u024F: Latin Extended (à, ó, ñ, ü 등)
    text = re.sub(r"[^\uAC00-\uD7A3\u3131-\u3163a-zA-Z\u00C0-\u024F0-9\s]", "", text)
    return text


# ── Main entry point ─────────────────────────────────────────────────────────

def normalize_korean(text: str, kss=None) -> str:
    """한국어 ASR 평가용 텍스트 정규화.

    Pipeline:
    1. 한자 → 한글 변환 (kss)
    2. 특수기호 → 한글 (%, ℃ 등)
    3. 예외 케이스 (유월, 시월, 스무살)
    4. 전화번호 변환
    5. 숫자+단위 변환 (5개→다섯개, 100원→백원)
    6. 소수점 숫자 변환 (3.14→삼점일사)
    7. 순수 숫자 변환 (1234→천이백삼십사)
    8. 구두점 제거
    9. 소문자 변환
    10. 공백 제거 (CER 비교용)
    """
    if not text:
        return ""

    # 1. 한자 → 한글
    if kss:
        try:
            text = kss.hanja2hangul(text)
        except Exception:
            pass

    # 2. 통화 기호+숫자 → 숫자+통화명 ($5 → 5달러)
    text = _convert_currency_prefix(text)

    # 3. 특수기호 → 한글
    text = _apply_special_symbols(text)

    # 3. 예외 케이스
    text = _apply_exceptions(text)

    # 4. 전화번호
    text = _convert_phone_numbers(text)

    # 5. 범위+단위 (숫자+단위보다 먼저)
    text = _convert_range_with_units(text)

    # 6. 숫자+단위
    text = _convert_numbers_with_units(text)

    # 6. 소수점 숫자
    text = _convert_float_numbers(text)

    # 7. 순수 숫자
    text = _convert_pure_numbers(text)

    # 8. 구두점 제거
    text = _remove_punctuation(text)

    # 9. 소문자
    text = text.lower()

    text = re.sub(r"\s+", "", text) 

    return text
