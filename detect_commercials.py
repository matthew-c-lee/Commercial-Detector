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
from dotenv import load_dotenv
import tiktoken
from openai import OpenAI
import sys
from dataclasses import dataclass
from pathlib import Path

load_dotenv()

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

@dataclass
class LabeledSegment:
    label: str
    start: str
    end: str

# --- Utility functions ---

def format_timestamp(seconds: float):
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    milliseconds = int((td.total_seconds() - total_seconds) * 1000)
    return str(td).split(".")[0] + f".{milliseconds:03}"

def parse_timestamp_to_seconds(ts):
    parts = ts.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return 0

# --- Visual Analysis: Black Frame Detection ---

def detect_black_frames(video_path: Path, threshold: int=10, min_duration: float=0.3) -> list[float]:
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

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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

def adjust_segments_to_black_frames(labeled_segments: list[LabeledSegment], black_frame_times) -> list[LabeledSegment]:
    adjusted = []
    black_seconds = sorted(bf for bf in black_frame_times)

    def find_closest(time):
        return min(black_seconds, key=lambda x: abs(x - time)) if black_seconds else time

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
                    end=format_timestamp(new_end))
                )

    return adjusted

# --- Audio + Transcription + Diarization ---

def extract_audio(video_path: Path, wav_path: Path):
    print("Extracting audio...")
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-ac", "1", "-ar", "16000", str(wav_path)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return wav_path

def transcribe_with_whisper(audio_path, model_size="base") -> list[Segment]:
    print("Transcribing with Whisper...")
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)

    assert isinstance(result["segments"], list)
    segments: list[dict] = result["segments"]

    return [
        Segment(
            start=seg["start"],
            text=seg["text"].strip(),
            speaker="Unknown"  # Speaker is added later via diarization
        )
        for seg in segments
    ]

def run_diarization(wav_path: Path, hf_token: str) -> list[SpeakerSpan]:
    print("Running speaker diarization...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)
    diarization = pipeline(wav_path)
    speaker_list = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_list.append(
            SpeakerSpan(
                start=turn.start,
                end=turn.end,
                speaker=speaker
            )
        )
    return speaker_list

def assign_speakers_to_segments(segments: list[Segment], speaker_spans: list[SpeakerSpan]) -> list[Segment]:
    print("Assigning speakers to segments...")
    annotated = []
    for segment in segments:
        speaker = "Unknown"
        for speaker_span in speaker_spans:
            if speaker_span.start <= segment.start <= speaker_span.end:
                speaker = speaker_span.speaker
                break
        annotated.append(
            Segment(
                start=segment.start,
                text=segment.text.strip(),
                speaker=speaker
            )
        )
    return annotated

# --- Chunking ---

def chunk_segments(segments: list[Segment], max_duration: float=1000, overlap: float=50) -> list[list[Segment]]:
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

# --- LLM Interaction ---

def call_ollama(prompt, model='mistral-deterministic'):
    print("Calling local LLM...")
    result = subprocess.run(
        ['ollama', 'run', model],
        input=prompt,
        text=True,
        capture_output=True
    )
    return result.stdout.strip()

def call_openai(prompt, model='gpt-4o'):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("Calling OpenAI API...")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    answer = response.choices[0].message.content
    assert answer is not None
    return answer.strip()

def build_prompt(segments: list[Segment]):
    transcript_text = "\n".join(
        f"[{format_timestamp(seg.start)}] (Speaker: {seg.speaker}) {seg.text}"
        for seg in segments
    )

    print(f"{transcript_text=}")

    prompt = f"""
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

Use speaker changes and **shifts in tone or intent**, as well as visual fade-to-black cues, to find transitions.

If there are two different commercials one after the other, list them separately!

Return your output in this format:
Commercial [00:00:05 - 00:01:05]
Bump [00:01:05 - 00:01:15]

(DO NOT INCLUDE OTHER TEXT)

Transcript:
{textwrap.indent(transcript_text, '  ')}
"""
    enc = tiktoken.encoding_for_model("gpt-4o")
    tokens = len(enc.encode(prompt))
    print(f"The prompt is {tokens} tokens.")
    return prompt

def extract_labeled_segments(response) -> list[LabeledSegment]:
    pattern = r'(Commercial|Bump)\s*\[\s*([\d:]+)\s*-\s*([\d:]+)\s*\]'
    results: list[LabeledSegment] = []
    for match in re.finditer(pattern, response, flags=re.IGNORECASE):
        label = match.group(1).capitalize()
        start = match.group(2)
        end = match.group(3)
        results.append(
            LabeledSegment(
                label=label, 
                start=start, 
                end=end
            )
        )
    return results

# --- Main ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_file", help="Path to video file")
    parser.add_argument("--output_dir", default="clips", help="Directory to save video segments")
    args = parser.parse_args()

    video_file = Path(args.video_file)
    output_dir = Path(args.output_dir)

    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("HUGGINGFACE_TOKEN needs to be an environment variable.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "prediction.txt")

    audio_path: Path = extract_audio(video_path=video_file, wav_path=Path("runtime/audio.wav"))
    speaker_spans: list[SpeakerSpan] = run_diarization(audio_path, hf_token)
    segments: list[Segment] = transcribe_with_whisper(audio_path)
    annotated_segments: list[Segment] = assign_speakers_to_segments(segments, speaker_spans)

    black_frame_starts: list[float] = detect_black_frames(video_file, threshold=10, min_duration=0.3)
    for black_frame_time in black_frame_starts:
        annotated_segments.append(
            Segment(
                start=black_frame_time,
                text="(Visual: fade to black)",
                speaker="Visual"
            )
        )

    annotated_segments.sort(key=lambda x: x.start)
    chunks: list[list[Segment]] = chunk_segments(annotated_segments, max_duration=1000, overlap=30)

    all_segments: list[LabeledSegment] = []
    for chunk in chunks:
        prompt = build_prompt(chunk)
        response = """
Commercial [00:00:06 - 00:01:03]
Commercial [00:01:07 - 00:01:35]
Commercial [00:01:37 - 00:02:30]
Commercial [00:03:23 - 00:03:49]
Commercial [00:04:10 - 00:05:07]
Bump [00:05:08 - 00:05:09]
Commercial [00:05:09 - 00:05:52]
Bump [00:05:54 - 00:05:57]
"""
        print("\nLLM Response:\n", response)
        labels: list[LabeledSegment] = extract_labeled_segments(response)
        all_segments.extend(labels)

    seen: set[LabeledSegment] = set()
    deduped = []
    for labeled_segment in all_segments:
        if labeled_segment not in seen:
            seen.add(labeled_segment)
            deduped.append(labeled_segment)

    deduped.sort(key=lambda labeled_segment: parse_timestamp_to_seconds(labeled_segment.start))

    sensitive_markers: list[float] = detect_black_frames(video_file, threshold=20, min_duration=0.03)
    adjusted_segments: list[LabeledSegment] = adjust_segments_to_black_frames(deduped, sensitive_markers)

    print("\n=== Final Labeled Segments ===")
    with open(output_file, "w") as f:
        for segment in adjusted_segments:
            line = f"{segment.label} [{segment.start} - {segment.end}]"
            print(line)
            f.write(line + "\n")

    print(f"\nResults saved to: {output_file}")

    print("\nExporting individual video segments...")
    counters = {}

    for segment in adjusted_segments:
        base_name = segment.label.lower()
        count = counters.get(base_name, 0)
        filename = os.path.join(output_dir, f"{base_name}{'' if count == 0 else f'_{count}'}.mp4")
        counters[base_name] = count + 1

        cmd = [
            "ffmpeg", "-y", "-i", video_file,
            "-ss", segment.start, "-to", segment.end,
            "-c:v", "libx264", "-c:a", "aac",
            filename
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Saved: {filename}")
