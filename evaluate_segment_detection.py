import re
from datetime import timedelta
from collections import defaultdict


def parse_label_blocks(text: str) -> list[tuple[str, int, int]]:
    """
    Parses lines like:
    Commercial [00:00:05 - 00:01:05]
    Returns: list of (label, start_seconds, end_seconds)
    """
    pattern: str = r"(\w+)\s*\[\s*([\d:]+)\s*-\s*([\d:]+)\s*\]"
    segments: list[tuple[str, int, int]] = []
    for match in re.finditer(pattern, text):
        label: str = match.group(1).capitalize()
        start: int = parse_timestamp_to_seconds(match.group(2))
        end: int = parse_timestamp_to_seconds(match.group(3))
        segments.append((label, start, end))
    return segments


def parse_timestamp_to_seconds(ts: str) -> int:
    parts: list[int] = list(map(int, ts.split(":")))
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    return 0


def format_seconds(s: int) -> str:
    return str(timedelta(seconds=int(s)))


def iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """
    Calculates intersection over union (IoU) for two time ranges.
    """
    inter_start: int = max(a_start, b_start)
    inter_end: int = min(a_end, b_end)
    inter: int = max(0, inter_end - inter_start)
    union: int = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


def match_segments(
    gt_segments: list[tuple[str, int, int]],
    pred_segments: list[tuple[str, int, int]],
    iou_threshold: float = 0.5,
) -> dict[str, dict[str, int]]:
    """
    Matches predicted segments to ground truth by label and overlap.
    Returns precision, recall, f1, and mismatched segments.
    """
    label_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )
    matched: set[int] = set()
    matched_gt: set[int] = set()

    for i, (plabel, pstart, pend) in enumerate(pred_segments):
        best_match: int | None = None
        best_iou: float = 0.0

        for j, (glabel, gstart, gend) in enumerate(gt_segments):
            if j in matched_gt or glabel != plabel:
                continue
            score: float = iou(pstart, pend, gstart, gend)
            if score >= iou_threshold and score > best_iou:
                best_iou = score
                best_match = j

        if best_match is not None:
            label_stats[plabel]["tp"] += 1
            matched.add(i)
            matched_gt.add(best_match)
        else:
            label_stats[plabel]["fp"] += 1

    for j, (glabel, _, _) in enumerate(gt_segments):
        if j not in matched_gt:
            label_stats[glabel]["fn"] += 1

    return label_stats


def compute_metrics(
    stats: dict[str, dict[str, int]],
) -> dict[str, dict[str, float | int]]:
    metrics: dict[str, dict[str, float | int]] = {}
    for label, vals in stats.items():
        tp: int = vals["tp"]
        fp: int = vals["fp"]
        fn: int = vals["fn"]
        precision: float = tp / (tp + fp) if (tp + fp) else 0.0
        recall: float = tp / (tp + fn) if (tp + fn) else 0.0
        f1: float = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        metrics[label] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return metrics


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print(
            "Usage: python evaluate_segment_detection.py <ground_truth.txt> <predicted.txt>"
        )
        sys.exit(1)

    with open(sys.argv[1], "r") as f:
        gt_text: str = f.read()

    with open(sys.argv[2], "r") as f:
        pred_text: str = f.read()

    gt_segments: list[tuple[str, int, int]] = parse_label_blocks(gt_text)
    pred_segments: list[tuple[str, int, int]] = parse_label_blocks(pred_text)

    stats: dict[str, dict[str, int]] = match_segments(
        gt_segments, pred_segments, iou_threshold=0.5
    )
    metrics: dict[str, dict[str, float | int]] = compute_metrics(stats)

    print("\n=== Evaluation Results ===")
    for label, m in metrics.items():
        print(f"\nLabel: {label}")
        print(f"  Precision: {m['precision']}")
        print(f"  Recall:    {m['recall']}")
        print(f"  F1 Score:  {m['f1']}")
        print(f"  TP: {m['tp']} | FP: {m['fp']} | FN: {m['fn']}")
