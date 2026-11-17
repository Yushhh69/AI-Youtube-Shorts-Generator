"""Zero-cost transcript heuristics; no external or metered LLM calls."""
import json
import re
from statistics import median
from typing import Dict, List, Tuple

STAMP = re.compile(r"^\[(\d+(?:\.\d+)?)s\]\s*(.*)$")
SIGNALS = {
    "secret": 3, "nobody": 3, "mistake": 3, "wrong": 2,
    "surprising": 3, "surprise": 3, "never": 2, "important": 2,
    "danger": 2, "dangerous": 2, "problem": 2, "solution": 2,
    "truth": 2, "actually": 2, "instead": 2, "because": 1,
    "discover": 2, "discovered": 2, "stop": 2, "avoid": 2,
    "best": 1, "worst": 2, "impossible": 2, "key": 1,
}
ACTION_PHRASES = (
    "how to", "you need to", "you should", "the reason",
    "here is", "here's", "what happens", "the problem", "the answer",
)


def _timestamped_lines(prompt: str) -> List[Tuple[float, str]]:
    transcript = prompt.rsplit("Transcript:\n", 1)[-1]
    result = []
    for line in transcript.splitlines():
        match = STAMP.match(line.strip())
        if match and match.group(2).strip():
            result.append((float(match.group(1)), match.group(2).strip()))
    return result


def _classify(prompt: str) -> Dict[str, str]:
    sample = prompt.rsplit("Transcript sample:", 1)[-1].lower()
    categories = {
        "podcast": ("podcast", "episode", "guest"),
        "interview": ("interview", "question", "answer"),
        "tutorial": ("tutorial", "step by step", "let's build"),
        "lecture": ("in this lecture", "chapter", "theorem"),
        "debate": ("i disagree", "you're wrong", "on the other hand"),
        "commentary": ("my opinion", "i think", "in my view"),
        "vlog": ("come with me", "this morning", "today we're"),
    }
    content_type = "other"
    for name, markers in categories.items():
        if sum(marker in sample for marker in markers) >= 2:
            content_type = name
            break
    word_count = len(re.findall(r"\b\w+\b", sample))
    density = "high" if word_count > 260 else "medium" if word_count > 100 else "low"
    return {"content_type": content_type, "density": density}


def _score(text: str) -> Tuple[int, str, str]:
    lower = text.lower()
    words = re.findall(r"\b[\w'-]+\b", lower)
    signal_hits = [
        (word, weight) for word, weight in SIGNALS.items()
        if re.search(r"\b" + re.escape(word) + r"\b", lower)
    ]
    number_count = len(re.findall(r"\b\d+(?:\.\d+)?%?\b", lower))
    question_count = text.count("?")
    emphasis_count = text.count("!")
    action_count = sum(phrase in lower for phrase in ACTION_PHRASES)
    filler_count = len(re.findall(r"\b(um|uh|erm|you know|sort of|kind of)\b", lower))
    score = (
        35 + min(24, sum(weight for _, weight in signal_hits) * 3)
        + min(10, number_count * 2)
        + min(8, question_count * 2 + emphasis_count)
        + min(8, action_count * 3)
        + min(18, len(words) // 8)
        - min(10, filler_count * 2)
    )
    score = max(25, min(96, int(score)))

    reasons = []
    if signal_hits:
        reasons.append("hook or contrast wording")
    if number_count:
        reasons.append("specific numbers or facts")
    if action_count:
        reasons.append("actionable phrasing")
    if question_count or emphasis_count:
        reasons.append("spoken emphasis")
    if not reasons:
        reasons.append("speech density")
    reason = "Transcript heuristic: " + ", ".join(reasons) + "."

    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    def sentence_score(sentence: str) -> int:
        value = 0
        lower_sentence = sentence.lower()
        for word, weight in SIGNALS.items():
            if re.search(r"\b" + re.escape(word) + r"\b", lower_sentence):
                value += weight
        if "?" in sentence or "!" in sentence:
            value += 2
        return value + min(4, len(sentence.split()) // 5)
    hook = max(sentences, key=sentence_score) if sentences else text[:180]
    if len(hook) > 220:
        hook = hook[:217].rsplit(" ", 1)[0] + "..."
    return score, hook, reason


def _highlights(prompt: str) -> Dict[str, List[Dict]]:
    rows = sorted(_timestamped_lines(prompt), key=lambda row: row[0])
    if len(rows) < 2:
        return {"highlights": []}

    gaps = [b[0] - a[0] for a, b in zip(rows[-11:-1], rows[-10:]) if b[0] > a[0]]
    tail = max(2.0, min(8.0, median(gaps) if gaps else 4.0))
    duration = rows[-1][0] + tail
    window = min(65.0, duration)
    if duration < 25.0:
        window = duration
    step = max(15.0, window * 0.65)
    proposed_starts = [0.0]
    point = step
    while point < max(0.1, duration - min(20.0, window * 0.25)):
        proposed_starts.append(point)
        point += step

    for timestamp, line in rows:
        lower = line.lower()
        if any(term in lower for term in ("nobody", "the secret", "the mistake", "the truth", "you need to", "what if")):
            proposed_starts.append(max(0.0, timestamp - 2.0))

    candidates = []
    selected_starts = []
    for proposed in sorted(proposed_starts):
        row = next((item for item in rows if item[0] >= proposed), None)
        if row is None:
            continue
        start = row[0]
        if any(abs(start - previous) < 18.0 for previous in selected_starts):
            continue
        end = min(duration, start + window)
        if end - start < min(15.0, window * 0.6):
            continue
        transcript_window = " ".join(
            line for timestamp, line in rows if start <= timestamp <= end
        ).strip()
        if len(transcript_window.split()) < 8:
            continue
        score, hook, reason = _score(transcript_window)
        title_words = hook.split()
        title = " ".join(title_words[:9]).strip(" ,.:;!?")
        if len(title_words) > 9:
            title += "..."
        candidates.append({
            "title": title or "Selected highlight",
            "start_time": round(start, 2),
            "end_time": round(end, 2),
            "score": score,
            "hook_sentence": hook,
            "virality_reason": reason,
        })
        selected_starts.append(start)

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return {"highlights": candidates[:20]}


def call_local_llm(prompt: str) -> str:
    """Compatibility function for the existing pipeline; uses local heuristics."""
    lower = prompt.lower()
    if "transcript sample:" in lower and "content type" in lower:
        result = _classify(prompt)
    else:
        result = _highlights(prompt)
    return json.dumps(result, ensure_ascii=False)
