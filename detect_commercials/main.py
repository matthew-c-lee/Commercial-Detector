import whisper
import subprocess
import re
import textwrap
from datetime import timedelta
from pyannote.audio import Pipeline
import argparse
import os
import cv2
import numpy as np
from dotenv import load_dotenv, set_key
import tiktoken
from openai import OpenAI
import sys
from dataclasses import dataclass
from pathlib import Path
import hashlib
import sqlite3

DOTENV_PATH = Path.home() / ".env"
load_dotenv(dotenv_path=DOTENV_PATH)

REQUIRED_KEYS = ["HUGGINGFACE_TOKEN", "OPENAI_API_KEY"]

TOKENS_PER_MINUTE = 527
PRICE_PER_1K_TOKENS_IN_DOLLARS = 0.005
PROMPT = """
You are analyzing a transcript from a video that contains:
- Commercials (paid advertisements)
- Bumps (brief channel-branded clips or transitions)
- Show content, cartoons, or dead air

Each line includes a timestamp and speaker ID.

Your job is to segment the transcript and return ONLY the portions that are clearly:
- Commercial
- Bump

BUMPS ARE AT LEAST 5 SECONDS LONG

All other content should be ignored.

For each **Commercial**, infer what product, brand, or service is being advertised.
Use that as the label. If it’s unclear, use a short descriptive label.
Use **lowercase letters** and **underscores instead of spaces**.

For example:
- A cereal ad → `honey_nut_cheerios`
- A DVD ad → `shrek_2_dvd`
- A generic promo → `nickelodeon_promo`

Use speaker changes and **shifts in tone or intent**, as well as visual fade-to-black cues, to find transitions.

If there are two different commercials one after the other, list them separately!

Return your output in this format:
honey_nut_cheerios [00:00:05 - 00:01:05]
Bump [00:01:05 - 00:01:15]

(DO NOT INCLUDE OTHER TEXT)

Transcript:
"""


@dataclass
class Segment:
    start: float
    text: str
    speaker: str


@dataclass
class SpeakerSpan:
    start: float
    end: float
    speaker: str


@dataclass
class BlackFrame:
    start: float


@dataclass(frozen=True)
class LabeledSegment:
    label: str
    start: str
    end: str


def format_timestamp(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    milliseconds = int((td.total_seconds() - total_seconds) * 1000)
    return str(td).split(".")[0] + f".{milliseconds:03}"


def parse_timestamp_to_seconds(ts: str) -> float:
    parts = ts.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return 0


def detect_black_frames(
    video_path: Path, threshold: int = 10, min_duration: float = 0.3
) -> list[float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0
    black_start = None
    markers = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        current_time = frame_idx / fps

        if avg_brightness < threshold:
            if black_start is None:
                black_start = current_time
        else:
            if black_start is not None:
                duration = current_time - black_start
                if duration >= min_duration:
                    markers.append(black_start)
                black_start = None

        frame_idx += 1

    cap.release()
    return markers


def adjust_segments_to_black_frames(
    labeled_segments: list[LabeledSegment], black_frame_times: list[float]
) -> list[LabeledSegment]:
    adjusted = []
    black_seconds = sorted(bf for bf in black_frame_times)

    def find_closest(time: float) -> float:
        return (
            min(black_seconds, key=lambda x: abs(x - time)) if black_seconds else time
        )

    for segment in labeled_segments:
        start_sec = parse_timestamp_to_seconds(segment.start)
        end_sec = parse_timestamp_to_seconds(segment.end)
        new_start = find_closest(start_sec)
        new_end = find_closest(end_sec)
        if new_end > new_start:
            adjusted.append(
                LabeledSegment(
                    label=segment.label,
                    start=format_timestamp(new_start),
                    end=format_timestamp(new_end),
                )
            )

    return adjusted


def extract_audio(video_path: Path, wav_path: Path) -> Path:
    print("Extracting audio...")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(wav_path),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return wav_path


def transcribe_with_whisper(
    audio_path: Path, model_size: str = "base"
) -> list[Segment]:
    print("Transcribing with Whisper...")
    model = whisper.load_model(model_size)
    result = model.transcribe(str(audio_path))

    assert isinstance(result["segments"], list)
    segments: list[dict] = result["segments"]

    return [
        Segment(
            start=seg["start"],
            text=seg["text"].strip(),
            speaker="Unknown",  # Speaker is added later via diarization
        )
        for seg in segments
    ]


def run_diarization(wav_path: Path, hf_token: str) -> list[SpeakerSpan]:
    print("Running speaker diarization...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization", use_auth_token=hf_token
    )
    diarization = pipeline(wav_path)
    speaker_list = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_list.append(
            SpeakerSpan(start=turn.start, end=turn.end, speaker=speaker)
        )
    return speaker_list


def assign_speakers_to_segments(
    segments: list[Segment], speaker_spans: list[SpeakerSpan]
) -> list[Segment]:
    print("Assigning speakers to segments...")
    annotated = []
    for segment in segments:
        speaker = "Unknown"
        for speaker_span in speaker_spans:
            if speaker_span.start <= segment.start <= speaker_span.end:
                speaker = speaker_span.speaker
                break
        annotated.append(
            Segment(start=segment.start, text=segment.text.strip(), speaker=speaker)
        )
    return annotated


def chunk_segments(
    segments: list[Segment], max_duration: float = 1000, overlap: float = 50
) -> list[list[Segment]]:
    chunks = []
    start_idx = 0
    while start_idx < len(segments):
        chunk = []
        start_time = segments[start_idx].start
        end_time = start_time + max_duration

        i = start_idx
        while i < len(segments) and segments[i].start <= end_time:
            chunk.append(segments[i])
            i += 1

        if chunk:
            chunks.append(chunk)

        next_start_time = start_time + (max_duration - overlap)
        next_idx = None
        for j in range(i, len(segments)):
            if segments[j].start >= next_start_time:
                next_idx = j
                break

        if next_idx is None:
            break
        start_idx = next_idx
    return chunks


def call_ollama(prompt: str, model: str = "mistral-deterministic") -> str | None:
    print("Calling local LLM...")
    result = subprocess.run(
        ["ollama", "run", model], input=prompt, text=True, capture_output=True
    )
    return result.stdout.strip()


def call_openai(prompt: str, model: str = "gpt-4o") -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("Calling OpenAI API...")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    answer = response.choices[0].message.content
    assert answer is not None
    return answer.strip()


def build_prompt(segments: list[Segment]) -> str:
    transcript_text = "\n".join(
        f"[{format_timestamp(seg.start)}] (Speaker: {seg.speaker}) {seg.text}"
        for seg in segments
    )

    print(f"{transcript_text=}")

    prompt = f"{PROMPT}\n{textwrap.indent(transcript_text, '  ')}"
    enc = tiktoken.encoding_for_model("gpt-4o")
    tokens = len(enc.encode(prompt))
    print(f"The prompt is {tokens} tokens.")
    return prompt


def extract_labeled_segments(response: str) -> list[LabeledSegment]:
    pattern = r"([a-zA-Z0-9_]+)\s*\[\s*([\d:]+)\s*-\s*([\d:]+)\s*\]"
    results: list[LabeledSegment] = []
    for match in re.finditer(pattern, response, flags=re.IGNORECASE):
        label = match.group(1).capitalize()
        start = match.group(2)
        end = match.group(3)
        results.append(LabeledSegment(label=label, start=start, end=end))
    return results


def hash_file(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def init_db(db_path: str = "cache.db") -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            hash TEXT PRIMARY KEY,
            path TEXT,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS transcripts (
            video_hash TEXT,
            start REAL,
            text TEXT,
            speaker TEXT,
            FOREIGN KEY(video_hash) REFERENCES videos(hash)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS speaker_spans (
            video_hash TEXT,
            start REAL,
            end REAL,
            speaker TEXT,
            FOREIGN KEY(video_hash) REFERENCES videos(hash)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS llm_responses (
            video_hash TEXT,
            chunk_index INTEGER,
            prompt TEXT,
            response TEXT,
            PRIMARY KEY (video_hash, chunk_index)
        )
    """)
    conn.commit()
    return conn


def is_video_cached(conn: sqlite3.Connection, video_hash: str) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM videos WHERE hash = ?", (video_hash,))
    return cur.fetchone() is not None


def store_video_data(
    conn: sqlite3.Connection,
    video_hash: str,
    video_path: Path,
    segments: list[Segment],
    spans: list[SpeakerSpan],
) -> None:
    cur = conn.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO videos (hash, path) VALUES (?, ?)",
        (video_hash, str(video_path)),
    )

    for seg in segments:
        cur.execute(
            "INSERT INTO transcripts (video_hash, start, text, speaker) VALUES (?, ?, ?, ?)",
            (video_hash, seg.start, seg.text, seg.speaker),
        )

    for span in spans:
        cur.execute(
            "INSERT INTO speaker_spans (video_hash, start, end, speaker) VALUES (?, ?, ?, ?)",
            (video_hash, span.start, span.end, span.speaker),
        )

    conn.commit()


def load_video_data(
    conn: sqlite3.Connection, video_hash: str
) -> tuple[list[Segment], list[SpeakerSpan]]:
    cur = conn.cursor()

    # Load segments from transcript table
    cur.execute(
        "SELECT start, text, speaker FROM transcripts WHERE video_hash = ?",
        (video_hash,),
    )
    segments = [
        Segment(start=row[0], text=row[1], speaker=row[2]) for row in cur.fetchall()
    ]

    # Load speaker spans
    cur.execute(
        "SELECT start, end, speaker FROM speaker_spans WHERE video_hash = ?",
        (video_hash,),
    )
    spans = [
        SpeakerSpan(start=row[0], end=row[1], speaker=row[2]) for row in cur.fetchall()
    ]

    return segments, spans


def store_llm_response(
    conn: sqlite3.Connection,
    video_hash: str,
    chunk_index: int,
    prompt: str,
    response: str,
) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO llm_responses (video_hash, chunk_index, prompt, response)
        VALUES (?, ?, ?, ?)
        """,
        (video_hash, chunk_index, prompt, response),
    )
    conn.commit()


def load_llm_response(
    conn: sqlite3.Connection, video_hash: str, chunk_index: int
) -> str | None:
    cur = conn.cursor()
    cur.execute(
        "SELECT response FROM llm_responses WHERE video_hash = ? AND chunk_index = ?",
        (video_hash, chunk_index),
    )
    row = cur.fetchone()
    return row[0] if row else None


def get_llm_response(
    conn: sqlite3.Connection,
    video_hash: str,
    idx: int,
    chunk: list[Segment],
    use_cache: bool,
) -> str:
    prompt = build_prompt(chunk)
    if use_cache:
        cached = load_llm_response(conn, video_hash, idx)
        if cached:
            print(f"\nLoaded cached LLM response for chunk {idx}")
            return cached
    response = call_openai(prompt)
    store_llm_response(conn, video_hash, idx, prompt, response)
    print(f"\nLLM Response (new) for chunk {idx}:\n", response)
    return response


def get_video_duration_seconds(path: Path) -> float:
    """Use ffprobe to get the duration of the video in seconds."""
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    return float(probe.stdout.strip())


def get_unique_filename(output_dir, base_name):
    base_pattern = re.compile(rf"^{re.escape(base_name)}(?:_(\d+))?\.mp4$")
    existing = [fname for fname in os.listdir(output_dir) if base_pattern.match(fname)]

    existing_indices = set()
    for fname in existing:
        match = base_pattern.match(fname)
        if match:
            idx = match.group(1)
            existing_indices.add(int(idx) if idx else 0)

    count = 0
    while count in existing_indices:
        count += 1

    suffix = f"_{count}" if count > 0 else ""
    return os.path.join(output_dir, f"{base_name}{suffix}.mp4")


def detect_commercials(
    conn: sqlite3.Connection,
    video_path: Path,
    output_dir: Path,
    should_reprocess: bool,
    override_cache: bool,
    use_cached_llm_response: bool,
) -> None:
    video_hash = hash_file(file_path=video_path)

    os.makedirs(output_dir, exist_ok=True)

    if not override_cache and is_video_cached(conn=conn, video_hash=video_hash):
        if not should_reprocess:
            print("Video already processed. Skipping...")
            sys.exit(0)
        else:
            print("Reprocessing using cached transcription and diarization data...")
            annotated_segments, speaker_spans = load_video_data(
                conn=conn, video_hash=video_hash
            )
    else:
        print("Processing new video...")
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            print("HUGGINGFACE_TOKEN needs to be an environment variable.")
            sys.exit(1)

        audio_path: Path = extract_audio(
            video_path=video_path, wav_path=Path("runtime/audio.wav")
        )
        speaker_spans: list[SpeakerSpan] = run_diarization(audio_path, hf_token)
        annotated_segments: list[Segment] = transcribe_with_whisper(audio_path)
        store_video_data(
            conn=conn,
            video_hash=video_hash,
            video_path=video_path,
            segments=annotated_segments,
            spans=speaker_spans,
        )

    black_frame_starts: list[float] = detect_black_frames(
        video_path, threshold=10, min_duration=0.3
    )
    for black_frame_time in black_frame_starts:
        annotated_segments.append(
            Segment(
                start=black_frame_time, text="(Visual: fade to black)", speaker="Visual"
            )
        )

    annotated_segments.sort(key=lambda x: x.start)
    chunks: list[list[Segment]] = chunk_segments(
        annotated_segments, max_duration=1000, overlap=30
    )

    all_segments: list[LabeledSegment] = []
    for i, chunk in enumerate(chunks):
        response: str = get_llm_response(
            conn=conn,
            video_hash=video_hash,
            idx=i,
            chunk=chunk,
            use_cache=use_cached_llm_response,
        )
        labels: list[LabeledSegment] = extract_labeled_segments(response=response)
        all_segments.extend(labels)

    seen: set[LabeledSegment] = set()
    deduped = []
    for labeled_segment in all_segments:
        if labeled_segment not in seen:
            seen.add(labeled_segment)
            deduped.append(labeled_segment)

    deduped.sort(
        key=lambda labeled_segment: parse_timestamp_to_seconds(labeled_segment.start)
    )

    sensitive_markers: list[float] = detect_black_frames(
        video_path, threshold=20, min_duration=0.03
    )
    adjusted_segments: list[LabeledSegment] = adjust_segments_to_black_frames(
        deduped, sensitive_markers
    )

    print("\n=== Final Labeled Segments ===")
    for segment in adjusted_segments:
        line = f"{segment.label} [{segment.start} - {segment.end}]"
        print(line)

    print("\nExporting individual video segments...")
    for segment in adjusted_segments:
        base_name = segment.label.lower()
        filename = get_unique_filename(output_dir, base_name)

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-ss",
            segment.start,
            "-to",
            segment.end,
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            filename,
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Saved: {filename}")


def ensure_env_vars():
    missing = [key for key in REQUIRED_KEYS if not os.getenv(key)]
    if not missing:
        return

    print("\nMissing environment variables:")

    for key in missing:
        value = input(f"Enter value for {key}: ").strip()
        if not value:
            print(f"Aborted: {key} is required.")
            sys.exit(1)
        set_key(str(DOTENV_PATH), key, value)

    print(f"\nSaved to {DOTENV_PATH}")


def main() -> None:
    ensure_env_vars()

    parser = argparse.ArgumentParser()
    parser.add_argument("video_file", help="Path to video file")
    parser.add_argument(
        "--output_dir", default="clips", help="Directory to save video segments"
    )
    parser.add_argument(
        "--reprocess", action="store_true", help="Reprocess even if video is cached"
    )
    parser.add_argument(
        "--override_cache", action="store_true", help="Do not use the cache"
    )
    parser.add_argument(
        "--use_cached_llm_response",
        action="store_true",
        help="Use the cached LLM response",
    )
    args = parser.parse_args()

    input_path: Path = Path(args.video_file)
    output_dir: Path = Path(args.output_dir)
    should_reprocess: bool = args.reprocess
    override_cache: bool = args.override_cache
    use_cached_llm_response: bool = args.use_cached_llm_response

    video_files = []
    if input_path.is_file():
        video_files = [input_path]
    elif input_path.is_dir():
        video_files = sorted(
            [
                f
                for f in input_path.iterdir()
                if f.suffix.lower() in (".mp4", ".mkv", ".mov", ".avi") and f.is_file()
            ]
        )
    else:
        print(f"Error: {input_path} is not a valid file or directory.")
        sys.exit(1)

    # Estimate total cost
    total_minutes = sum(get_video_duration_seconds(f) for f in video_files) / 60
    estimated_tokens = total_minutes * TOKENS_PER_MINUTE
    price = (estimated_tokens / 1000) * PRICE_PER_1K_TOKENS_IN_DOLLARS

    print("\n=== Batch Summary ===")
    print(f"Total videos: {len(video_files)}")
    print(f"Total video duration: {total_minutes:.2f} minutes")
    print(f"Estimated total token usage: {int(estimated_tokens)} tokens")
    print(f"Estimated total API cost: ${price:.2f}")

    confirm = input("Proceed with LLM calls for all videos? (y/n): ").strip().lower()
    if confirm not in ("y", "yes"):
        print("Aborted by user.")
        sys.exit(0)

    db = init_db()

    for video_path in video_files:
        detect_commercials(
            conn=db,
            video_path=video_path,
            output_dir=output_dir,
            should_reprocess=should_reprocess,
            override_cache=override_cache,
            use_cached_llm_response=use_cached_llm_response,
        )


if __name__ == "__main__":
    main()
