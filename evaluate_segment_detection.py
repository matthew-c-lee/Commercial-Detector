import re
from datetime import timedelta
from collections import defaultdict

# --- Parsing utilities ---


def parse_label_blocks(text):
    """
    Parses lines like:
    Commercial [00:00:05 - 00:01:05]
    Returns: list of (label, start_seconds, end_seconds)
    """
    pattern = r"(\w+)\s*\[\s*([\d:]+)\s*-\s*([\d:]+)\s*\]"
    segments = []
    for match in re.finditer(pattern, text):
        label = match.group(1).capitalize()
        start = parse_timestamp_to_seconds(match.group(2))
        end = parse_timestamp_to_seconds(match.group(3))
        segments.append((label, start, end))
    return segments


def parse_timestamp_to_seconds(ts):
    parts = list(map(int, ts.split(":")))
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    return 0


def format_seconds(s):
    return str(timedelta(seconds=int(s)))


# --- Evaluation utilities ---


def iou(a_start, a_end, b_start, b_end):
    """
    Calculates intersection over union (IoU) for two time ranges.
    """
    inter_start = max(a_start, b_start)
    inter_end = min(a_end, b_end)
    inter = max(0, inter_end - inter_start)
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0


def match_segments(gt_segments, pred_segments, iou_threshold=0.5):
    """
    Matches predicted segments to ground truth by label and overlap.
    Returns precision, recall, f1, and mismatched segments.
    """
    label_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    matched = set()
    matched_gt = set()

    for i, (plabel, pstart, pend) in enumerate(pred_segments):
        best_match = None
        best_iou = 0

        for j, (glabel, gstart, gend) in enumerate(gt_segments):
            if j in matched_gt or glabel != plabel:
                continue
            score = iou(pstart, pend, gstart, gend)
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


def compute_metrics(stats):
    metrics = {}
    for label, vals in stats.items():
        tp = vals["tp"]
        fp = vals["fp"]
        fn = vals["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = (
            2 * precision * recall / (precision + recall) if (precision + recall) else 0
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


# --- Main

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print(
            "Usage: python evaluate_segment_detection.py <ground_truth.txt> <predicted.txt>"
        )
        sys.exit(1)

    with open(sys.argv[1], "r") as f:
        gt_text = f.read()

    with open(sys.argv[2], "r") as f:
        pred_text = f.read()

    gt_segments = parse_label_blocks(gt_text)
    pred_segments = parse_label_blocks(pred_text)

    stats = match_segments(gt_segments, pred_segments, iou_threshold=0.5)
    metrics = compute_metrics(stats)

    print("\n=== Evaluation Results ===")
    for label, m in metrics.items():
        print(f"\nLabel: {label}")
        print(f"  Precision: {m['precision']}")
        print(f"  Recall:    {m['recall']}")
        print(f"  F1 Score:  {m['f1']}")
        print(f"  TP: {m['tp']} | FP: {m['fp']} | FN: {m['fn']}")
