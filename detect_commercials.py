import whisper
import subprocess
import re
import textwrap
from datetime import timedelta
from pyannote.audio import Pipeline
import sys
import os
import cv2
import numpy as np
from dotenv import load_dotenv
import tiktoken
from openai import OpenAI

load_dotenv()

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

def detect_black_frames(video_path, threshold=10, min_duration=0.3):
    cap = cv2.VideoCapture(video_path)
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
                    markers.append({"start": black_start})
                black_start = None

        frame_idx += 1

    cap.release()
    return markers

def adjust_segments_to_black_frames(segments, black_frames):
    adjusted = []
    black_seconds = sorted(bf["start"] for bf in black_frames)

    def find_closest(time):
        return min(black_seconds, key=lambda x: abs(x - time)) if black_seconds else time

    for label, start, end in segments:
        start_sec = parse_timestamp_to_seconds(start)
        end_sec = parse_timestamp_to_seconds(end)
        new_start = find_closest(start_sec)
        new_end = find_closest(end_sec)
        if new_end > new_start:
            adjusted.append((label, format_timestamp(new_start), format_timestamp(new_end)))

    return adjusted

# --- Audio + Transcription + Diarization ---

def extract_audio(video_path, wav_path="runtime/audio.wav"):
    print("Extracting audio...")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "1", "-ar", "16000", wav_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return wav_path

def transcribe_with_whisper(audio_path, model_size="base"):
    print("Transcribing with Whisper...")
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)
    return result["segments"]

def run_diarization(wav_path, hf_token):
    print("Running speaker diarization...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)
    diarization = pipeline(wav_path)
    speaker_map = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_map.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })
    return speaker_map

def assign_speakers_to_segments(segments, speaker_map):
    print("Assigning speakers to segments...")
    annotated = []
    for seg in segments:
        speaker = "Unknown"
        for entry in speaker_map:
            if entry["start"] <= seg["start"] <= entry["end"]:
                speaker = entry["speaker"]
                break
        annotated.append({
            "start": seg["start"],
            "text": seg["text"].strip(),
            "speaker": speaker
        })
    return annotated

# --- Chunking ---

def chunk_segments(segments, max_duration=1000, overlap=50):
    chunks = []
    start_idx = 0
    while start_idx < len(segments):
        chunk = []
        start_time = segments[start_idx]['start']
        end_time = start_time + max_duration

        i = start_idx
        while i < len(segments) and segments[i]['start'] <= end_time:
            chunk.append(segments[i])
            i += 1

        if chunk:
            chunks.append(chunk)

        next_start_time = start_time + (max_duration - overlap)
        next_idx = None
        for j in range(i, len(segments)):
            if segments[j]['start'] >= next_start_time:
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
        temperature=0.0  # deterministic output
    )
    answer = response.choices[0].message.content
    assert answer is not None
    return answer.strip()

def build_prompt(segments):
    transcript_text = "\n".join(
        f"[{format_timestamp(seg['start'])}] (Speaker: {seg['speaker']}) {seg['text']}"
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
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = len(enc.encode(prompt))
    print(f"The prompt is {tokens} tokens.")
    return prompt

def extract_labeled_segments(response):
    pattern = r'(Commercial|Bump)\s*\[\s*([\d:]+)\s*-\s*([\d:]+)\s*\]'
    results = []
    for match in re.finditer(pattern, response, flags=re.IGNORECASE):
        label = match.group(1).capitalize()
        start = match.group(2)
        end = match.group(3)
        results.append((label, start, end))
    return results

# --- Main ---

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect_commercials_with_speakers.py <video_file>")
        sys.exit(1)

    video_file = sys.argv[1]
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("HUGGINGFACE_TOKEN not found in .env")
        sys.exit(1)

    output_file = "training/prediction.txt"

    audio_path = extract_audio(video_file)
    speaker_map = run_diarization(audio_path, hf_token)
    segments = transcribe_with_whisper(audio_path)
    annotated = assign_speakers_to_segments(segments, speaker_map)

    visual_markers = detect_black_frames(video_file, threshold=10, min_duration=0.3)
    for marker in visual_markers:
        annotated.append({
            "start": marker["start"],
            "text": "(Visual: fade to black)",
            "speaker": "Visual"
        })

    annotated.sort(key=lambda x: x['start'])
    chunks = chunk_segments(annotated, max_duration=1000, overlap=30)

    all_segments = []
    for chunk in chunks:
        prompt = build_prompt(chunk)
        # response = call_openai(prompt)
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
        labels = extract_labeled_segments(response)
        all_segments.extend(labels)

    seen = set()
    deduped = []
    for label, start, end in all_segments:
        key = (label, start, end)
        if key not in seen:
            seen.add(key)
            deduped.append((label, start, end))

    deduped.sort(key=lambda x: parse_timestamp_to_seconds(x[1]))

    sensitive_markers = detect_black_frames(video_file, threshold=20, min_duration=0.03)

    adjusted = adjust_segments_to_black_frames(deduped, sensitive_markers)

    print("\n=== Final Labeled Segments ===")
    with open(output_file, "w") as f:
        for label, start, end in adjusted:
            line = f"{label} [{start} - {end}]"
            print(line)
            f.write(line + "\n")

    print(f"\nResults saved to: {output_file}")
