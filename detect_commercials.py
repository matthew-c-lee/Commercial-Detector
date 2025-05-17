import whisper
import subprocess
import re
import textwrap
from datetime import timedelta
from pyannote.audio import Pipeline
import sys
import os
from dotenv import load_dotenv
import tiktoken

load_dotenv()

# --- Utility functions ---

def format_timestamp(seconds: float):
    return str(timedelta(seconds=int(seconds)))

def parse_timestamp_to_seconds(ts):
    parts = list(map(int, ts.split(":")))
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    return 0

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

def chunk_segments(segments, max_duration=250, overlap=50):
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

Each line includes a timestamp and speaker label.
Speaker changes may indicate a shift from one ad to another.

Your task: segment the transcript into clearly labeled parts, and only output segments that are:
- "Commercial"
- "Bump"

Do NOT include show content, dead air, or unknown sections.

Return your output in this format:
Commercial [00:00:05 - 00:01:05]
Bump [00:01:05 - 00:01:15]

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
    chunks = chunk_segments(annotated, max_duration=150, overlap=30)

    all_segments = []
    for chunk in chunks:
        prompt = build_prompt(chunk)
        response = call_ollama(prompt)
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

    print("\n=== Final Labeled Segments ===")
    with open(output_file, "w") as f:
        for label, start, end in deduped:
            line = f"{label} [{start} - {end}]"
            print(line)
            f.write(line + "\n")

    print(f"\nResults saved to: {output_file}")
