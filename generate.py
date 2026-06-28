#!/usr/bin/env python3
import struct
import zipfile
import json
import os
import random
import argparse
from pathlib import Path
from collections import defaultdict

import librosa
import numpy as np

AUDIO_DIR = "/home/hormxn/Downloads/yt_music_recaps/all_songs"
OUTPUT_DIR = "/home/hormxn/Downloads/yt_music_recaps/all_maps"

MIN_GAP_MS = 40
NOTES_PER_MINUTE = 0
RESERVE_DURATION = 2
FIXED_DIFFICULTY = 3
FIXED_DIFFICULTY_NAME = "Hard"

PATTERNS = {"snake": True, "spiral": True, "island": True, "plus": True, "gap": True}
PATTERN_CHANCES = {"snake": 0.14, "spiral": 0.08, "island": 0.14, "plus": 0.06, "gap": 0.04}

GRID = [(-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 0), (0, 1),
        (1, -1), (1, 0), (1, 1)]
ALL_CELLS = set(GRID)

SNAKE_PATTERNS = {
    "right":  [(-1, 0), (0, 0), (1, 0)],
    "left":   [(1, 0), (0, 0), (-1, 0)],
    "up":     [(0, -1), (0, 0), (0, 1)],
    "down":   [(0, 1), (0, 0), (0, -1)],
    "diag_r": [(-1, -1), (0, 0), (1, 1)],
    "diag_l": [(1, -1), (0, 0), (-1, 1)],
    "diag_ru": [(-1, 1), (0, 0), (1, -1)],
    "diag_lu": [(1, 1), (0, 0), (-1, -1)],
}

SPIRAL_PATTERNS = [
    [(0,0),(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)],
    [(0,0),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)],
    [(-1,-1),(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0),(0,0)],
    [(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(0,0)],
    [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0)],
    [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1),(1,0)],
    [(0,0),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1),(1,1)],
    [(0,0),(-1,0),(-1,-1),(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1)],
]

ISLAND_PATTERNS = [
    [(-1,-1),(1,1),(0,0),(-1,1),(1,-1),(0,0),(-1,-1),(1,1),(0,0)],
    [(1,-1),(-1,1),(0,0),(1,1),(-1,-1),(0,0),(1,-1),(-1,1),(0,0)],
    [(-1,-1),(1,1),(-1,1),(1,-1),(-1,-1),(1,1),(-1,1),(1,-1),(-1,-1),(1,1)],
    [(1,-1),(-1,1),(1,1),(-1,-1),(1,-1),(-1,1),(1,1),(-1,-1),(1,-1),(-1,1)],
    [(-1,0),(1,0),(0,0),(-1,0),(1,0),(0,0),(-1,0),(1,0),(0,0),(-1,0)],
    [(0,-1),(0,1),(0,0),(0,-1),(0,1),(0,0),(0,-1),(0,1),(0,0),(0,-1)],
    [(-1,-1),(1,1),(0,0),(-1,1),(1,-1),(0,0),(-1,-1),(1,1),(0,0),(-1,1),(1,-1)],
    [(1,-1),(-1,1),(0,0),(1,1),(-1,-1),(0,0),(1,-1),(-1,1),(0,0),(1,1),(-1,-1)],
    [(-1,-1),(1,1),(-1,1),(1,-1),(-1,-1),(1,1),(-1,1),(1,-1),(-1,-1),(1,1),(-1,1),(1,-1)],
    [(-1,0),(0,-1),(1,0),(0,1),(-1,0),(0,-1),(1,0),(0,1),(-1,0),(0,-1),(1,0),(0,1)],
    [(-1,-1),(0,0),(1,1),(0,0),(-1,1),(0,0),(1,-1),(0,0),(-1,-1),(0,0),(1,1)],
    [(1,-1),(0,0),(-1,1),(0,0),(1,1),(0,0),(-1,-1),(0,0),(1,-1),(0,0),(-1,1)],
    [(-1,-1),(1,1),(0,0),(-1,1),(1,-1),(0,0),(-1,-1),(1,1),(0,0),(-1,1),(1,-1),(0,0)],
    [(1,-1),(-1,1),(0,0),(1,1),(-1,-1),(0,0),(1,-1),(-1,1),(0,0),(1,1),(-1,-1),(0,0)],
    [(-1,-1),(1,1),(-1,1),(1,-1),(-1,-1),(1,1),(-1,1),(1,-1),(-1,-1),(1,1),(-1,1),(1,-1),(-1,-1)],
]

PATTERN_COUNTER = defaultdict(int)


def analyze_audio(y, sr):
    hop = 512

    harmonic, percussive = librosa.effects.hpss(y)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_p = librosa.feature.rms(y=percussive, hop_length=hop)[0]
    rms_h = librosa.feature.rms(y=harmonic, hop_length=hop)[0]
    frame_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)

    diff = np.zeros(len(rms))
    for i in range(1, len(rms)):
        diff[i] = max(0, rms[i] - rms[i-1])

    window = 8
    onsets = []
    for i in range(window, len(rms)):
        local_avg = np.mean(rms[i-window:i])
        if local_avg < 1e-6:
            continue
        spike = diff[i] / local_avg
        if spike > 0.15:
            onsets.append((i, spike))

    filtered = []
    for idx, spike in onsets:
        if not filtered or idx - filtered[-1][0] >= 2:
            filtered.append((idx, spike))

    onset_times = np.array([frame_times[idx] for idx, _ in filtered])
    onset_frame_indices = [idx for idx, _ in filtered]
    onset_spikes = [spike for _, spike in filtered]

    if len(onset_times) == 0:
        onset_times = np.array([0.5])
        return onset_times, [0.0], [0.5], [0.0]

    pitches, magnitudes = librosa.piptrack(y=harmonic, sr=sr, fmin=50, fmax=2000)

    onset_pitches = []
    onset_volumes = []
    for frame_idx in onset_frame_indices:
        pitch_frame = pitches[:, frame_idx]
        mag_frame = magnitudes[:, frame_idx]
        if mag_frame.sum() > 0:
            avg_pitch = np.average(pitch_frame, weights=mag_frame)
        else:
            avg_pitch = 0.0
        onset_pitches.append(avg_pitch)
        onset_volumes.append(rms[frame_idx])

    return onset_times, onset_pitches, onset_volumes, onset_spikes


def normalize_to_grid(values):
    vmin = min(values) if values else 0
    vmax = max(values) if values else 1
    if vmax - vmin < 1e-6:
        return [0.0] * len(values)
    return [2.0 * ((v - vmin) / (vmax - vmin)) - 1.0 for v in values]


def pitch_to_x(pitch):
    if pitch <= 0:
        return 0.0
    log_pitch = np.log2(max(pitch, 1))
    x = (log_pitch - 5.0) / 5.0
    return max(-1.0, min(1.0, 2.0 * x - 1.0))


def thin_onsets(onset_times, onset_pitches, onset_volumes, min_gap_ms, npm, onset_spikes=None):
    if onset_spikes is None:
        onset_spikes = [0.0] * len(onset_times)

    if npm > 0:
        target_gap = 60.0 / npm
        thinned = [0]
        for i in range(1, len(onset_times)):
            if onset_times[i] - onset_times[thinned[-1]] >= target_gap:
                thinned.append(i)
        onset_times = [onset_times[i] for i in thinned]
        onset_pitches = [onset_pitches[i] for i in thinned]
        onset_volumes = [onset_volumes[i] for i in thinned]
        onset_spikes = [onset_spikes[i] for i in thinned]

    if min_gap_ms > 0:
        filtered = [0]
        for i in range(1, len(onset_times)):
            dt = (onset_times[i] - onset_times[filtered[-1]]) * 1000
            if dt >= min_gap_ms:
                filtered.append(i)
        onset_times = [onset_times[i] for i in filtered]
        onset_pitches = [onset_pitches[i] for i in filtered]
        onset_volumes = [onset_volumes[i] for i in filtered]
        onset_spikes = [onset_spikes[i] for i in filtered]

    return onset_times, onset_pitches, onset_volumes, onset_spikes


def detect_gaps(onset_times, onset_volumes):
    vol_norm = normalize_to_grid(onset_volumes)
    gaps = set()
    for i in range(1, len(onset_times)):
        dt = onset_times[i] - onset_times[i-1]
        vol_drop = vol_norm[i-1] - vol_norm[i]
        if dt > 0.4:
            gaps.add(i)
        if vol_drop > 1.2 and dt > 0.2:
            gaps.add(i)
    return gaps


def make_plus(cx, cy):
    cells = set()
    notes = []
    for dx, dy in [(0, 1), (-1, 0), (1, 0), (0, -1)]:
        nx, ny = cx + dx, cy + dy
        if -1 <= nx <= 1 and -1 <= ny <= 1:
            cells.add((nx, ny))
            notes.append((nx, ny))
    cells.add((cx, cy))
    return notes, cells


def generate_notes(onset_times, onset_pitches, onset_volumes, onset_spikes):
    global PATTERN_COUNTER
    PATTERN_COUNTER.clear()

    n = len(onset_times)
    notes = [None] * n
    in_pattern = [False] * n

    vol_norm = normalize_to_grid(onset_volumes)
    gaps = detect_gaps(onset_times, onset_volumes)
    occupied_ranges = defaultdict(list)

    def is_available(x, y, idx):
        for end_idx in occupied_ranges[(x, y)]:
            if end_idx >= idx:
                return False
        return True

    def reserve_cell(x, y, idx, duration=1):
        occupied_ranges[(x, y)].append(idx + duration)

    def place_pattern(pattern, name, idx):
        length = min(len(pattern), n - idx)
        for j in range(length):
            x, y = pattern[j]
            if not is_available(x, y, idx + j):
                return False
        for j in range(length):
            x, y = pattern[j]
            ms = int(onset_times[idx + j] * 1000)
            notes[idx + j] = (x, y, ms)
            in_pattern[idx + j] = True
            reserve_cell(x, y, idx + j, duration=RESERVE_DURATION)
        PATTERN_COUNTER[name] += 1
        return True

    def place_pattern_fast(pattern, name, idx, base_ms, spacing_ms):
        length = min(len(pattern), n - idx)
        for j in range(length):
            x, y = pattern[j]
            if not is_available(x, y, idx + j):
                return False
        for j in range(length):
            x, y = pattern[j]
            ms = base_ms + j * spacing_ms
            notes[idx + j] = (x, y, ms)
            in_pattern[idx + j] = True
            reserve_cell(x, y, idx + j, duration=RESERVE_DURATION)
        PATTERN_COUNTER[name] += 1
        return True

    def local_tempo(idx, window=5):
        start = max(0, idx - window)
        end = min(n, idx + window + 1)
        dt = onset_times[end - 1] - onset_times[start]
        count = end - start - 1
        if count <= 0 or dt <= 0:
            return 500
        return (dt / count) * 1000

    def local_pitch_direction(idx, window=4):
        start = max(0, idx - window)
        end = min(n, idx + window + 1)
        pitches_slice = onset_pitches[start:end]
        if len(pitches_slice) < 2:
            return 0
        return pitches_slice[-1] - pitches_slice[0]

    i = 0
    while i < n:
        if PATTERNS["gap"] and (i in gaps or random.random() < PATTERN_CHANCES["gap"] * 0.5):
            PATTERN_COUNTER["gap"] += 1
            i += 1
            continue

        tempo_ms = local_tempo(i)
        pitch_dir = local_pitch_direction(i)
        vol = vol_norm[i]
        high_energy = vol > 0.6
        fast_section = tempo_ms < 200
        slow_section = tempo_ms > 400
        rising = pitch_dir > 0.3
        falling = pitch_dir < -0.3

        placed = False

        if PATTERNS["island"] and high_energy and slow_section and i + 8 < n:
            if random.random() < 0.3:
                pat = random.choice(ISLAND_PATTERNS)
                if place_pattern(pat, "island", i):
                    i += len(pat)
                    placed = True

        if not placed and PATTERNS["spiral"] and fast_section and i + 8 < n:
            if random.random() < 0.25:
                spiral = random.choice(SPIRAL_PATTERNS)
                if place_pattern_fast(spiral, "spiral_fast", i, int(onset_times[i] * 1000), 90):
                    i += len(spiral)
                    placed = True

        if not placed and PATTERNS["spiral"] and high_energy and i + 8 < n:
            if random.random() < 0.15:
                spiral = random.choice(SPIRAL_PATTERNS)
                if place_pattern(spiral, "spiral", i):
                    i += len(spiral)
                    placed = True

        if not placed and PATTERNS["snake"] and i + 2 < n:
            if (rising or falling) and random.random() < 0.2:
                if rising:
                    candidates = ["up", "diag_ru", "diag_r"]
                else:
                    candidates = ["down", "diag_l", "diag_lu"]
                snake = SNAKE_PATTERNS[random.choice(candidates)]
                if place_pattern(snake, "snake", i):
                    i += len(snake)
                    placed = True

        if not placed and PATTERNS["plus"] and high_energy and i + 3 < n:
            if random.random() < 0.08:
                cx = random.randint(-1, 1)
                cy = random.randint(-1, 1)
                plus_notes, _ = make_plus(cx, cy)
                if len(plus_notes) == 4:
                    if place_pattern(plus_notes, "plus", i):
                        i += 4
                        placed = True

        if not placed:
            target_x = pitch_to_x(onset_pitches[i])
            target_y = vol_norm[i]
            tx = max(-1, min(1, round(target_x)))
            ty = max(-1, min(1, round(target_y)))

            if not is_available(tx, ty, i):
                candidates = []
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = tx + dx, ty + dy
                        if -1 <= nx <= 1 and -1 <= ny <= 1 and is_available(nx, ny, i):
                            candidates.append((nx, ny))
                if candidates:
                    tx, ty = random.choice(candidates)
                else:
                    for cell in ALL_CELLS:
                        if is_available(cell[0], cell[1], i):
                            tx, ty = cell
                            break

            ms = int(onset_times[i] * 1000)
            notes[i] = (tx, ty, ms)
            reserve_cell(tx, ty, i, duration=1)
            i += 1

    final = [note for note in notes if note is not None]

    extras = []
    for i in range(n):
        if notes[i] is None or in_pattern[i]:
            continue
        spike = onset_spikes[i] if i < len(onset_spikes) else 0
        x, y, ms = notes[i]
        if spike > 0.6:
            extras.append((x, y, ms + 5))
            extras.append((x, y, ms + 10))
        elif spike > 0.35:
            extras.append((x, y, ms + 7))

    final.extend(extras)
    final.sort(key=lambda n: n[2])
    return final


def estimate_difficulty(notes):
    if len(notes) < 2:
        return 0, "NA"

    total_ms = notes[-1][2] - notes[0][2]
    duration_min = max(total_ms / 60000, 0.01)
    npm = len(notes) / duration_min

    jumps = []
    speeds = []
    for i in range(1, len(notes)):
        dx = abs(notes[i][0] - notes[i-1][0])
        dy = abs(notes[i][1] - notes[i-1][1])
        jumps.append(dx + dy)
        dt = notes[i][2] - notes[i-1][2]
        if 0 < dt < 10000:
            speeds.append(dt)

    avg_jump = sum(jumps) / len(jumps)
    long_pct = sum(1 for j in jumps if j >= 3) / len(jumps) * 100

    avg_ms = sum(speeds) / len(speeds) if speeds else 999
    fast_pct = sum(1 for s in speeds if s < 120) / len(speeds) * 100 if speeds else 0

    score = 0

    # npm thresholds from discord reference maps (median values)
    # Easy=228, Normal=292, Hard=348, Insane=488, Illogical=929
    if npm < 250: score += 1
    elif npm < 350: score += 2
    elif npm < 500: score += 3
    elif npm < 750: score += 4
    else: score += 5

    # avg_ms thresholds (inverse — lower = harder)
    # Easy=274, Normal=212, Hard=180, Insane=126, Illogical=67
    if avg_ms > 250: score += 1
    elif avg_ms > 200: score += 2
    elif avg_ms > 150: score += 3
    elif avg_ms > 100: score += 4
    else: score += 5

    # fast_pct thresholds
    # Easy=22, Normal=18, Hard=32, Insane=64, Illogical=91
    if fast_pct < 20: score += 1
    elif fast_pct < 35: score += 2
    elif fast_pct < 60: score += 3
    elif fast_pct < 85: score += 4
    else: score += 5

    # long_pct thresholds
    # Easy=13, Normal=14, Hard=15, Insane=24, Illogical=20
    if long_pct < 15: score += 1
    elif long_pct < 25: score += 2
    elif long_pct < 40: score += 3
    else: score += 4

    if score <= 4:   return 1, "Easy"
    elif score <= 6:  return 2, "Normal"
    elif score <= 9:  return 3, "Hard"
    elif score <= 12: return 4, "Insane"
    else:             return 5, "Illogical"


def encode_phxmo(notes):
    buf = bytearray()
    buf += struct.pack("<I", 12)
    buf += struct.pack("<I", len(notes))

    for x, y, ms in notes:
        buf += struct.pack("<I", ms)
        quantum = (int(x) != x or int(y) != y or x < -1 or x > 1 or y < -1 or y > 1)
        buf += struct.pack("B", 1 if quantum else 0)
        if quantum:
            buf += struct.pack("<f", float(x))
            buf += struct.pack("<f", float(y))
        else:
            buf += struct.pack("B", int(x + 1))
            buf += struct.pack("B", int(y + 1))

    for _ in range(11):
        buf += struct.pack("<I", 0)

    return bytes(buf)


def make_metadata(song_name, note_count, audio_len_ms, difficulty, difficulty_name, map_id):
    return json.dumps({
        "Artist": "",
        "ArtistLink": "",
        "ArtistPlatform": "",
        "AudioExt": "mp3",
        "Difficulty": difficulty,
        "DifficultyName": difficulty_name,
        "HasAudio": True,
        "HasCover": False,
        "HasVideo": False,
        "ID": map_id,
        "Length": audio_len_ms,
        "Mappers": ["MapGenerator"],
        "Rating": 0.0,
        "Title": song_name
    }, indent="\t")


def process_file(mp3_path, output_dir):
    name = Path(mp3_path).stem
    map_id = name.replace(" ", "_").replace("-", "_").lower()

    print(f"  Loading {name}...", end=" ", flush=True)
    try:
        y, sr = librosa.load(mp3_path, sr=22050, mono=True)
    except Exception as e:
        print(f"ERROR loading: {e}")
        return False

    duration_ms = int(len(y) / sr * 1000)

    print("analyzing...", end=" ", flush=True)
    onset_times, onset_pitches, onset_volumes, onset_spikes = analyze_audio(y, sr)
    if len(onset_times) == 0:
        print("no onsets found, using uniform spacing")
        duration_sec = len(y) / sr
        onset_times = np.arange(0.5, duration_sec, 0.5)
        onset_pitches = [0.0] * len(onset_times)
        onset_volumes = [0.5] * len(onset_times)
        onset_spikes = [0.0] * len(onset_times)

    onset_times, onset_pitches, onset_volumes, onset_spikes = thin_onsets(
        onset_times, onset_pitches, onset_volumes, MIN_GAP_MS, NOTES_PER_MINUTE, onset_spikes
    )

    print(f"{len(onset_times)} onsets...", end=" ", flush=True)
    notes = generate_notes(onset_times, onset_pitches, onset_volumes, onset_spikes)

    diff_num = FIXED_DIFFICULTY
    diff_name = FIXED_DIFFICULTY_NAME

    pattern_str = " ".join(f"{k}:{v}" for k, v in sorted(PATTERN_COUNTER.items()))
    print(f"OK ({len(notes)} notes) [{pattern_str}] [{diff_name}]")

    phxmo_data = encode_phxmo(notes)
    metadata_str = make_metadata(
        name, len(notes), duration_ms, diff_num, diff_name, map_id
    )

    map_dir = os.path.join(output_dir, map_id)
    os.makedirs(map_dir, exist_ok=True)

    phxm_path = os.path.join(map_dir, f"{map_id}.phxm")
    with zipfile.ZipFile(phxm_path, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("metadata.json", metadata_str)
        zf.writestr("objects.phxmo", phxmo_data)
        zf.write(mp3_path, "audio.mp3")

    return True


def main():
    parser = argparse.ArgumentParser(description="Rhythia map generator")
    parser.add_argument("-i", "--input", help="Single MP3 file to process")
    parser.add_argument("-o", "--output", help="Output directory override")
    parser.add_argument("-d", "--difficulty", type=int, help="Override difficulty (0-5)")
    parser.add_argument("-n", "--npm", type=int, help="Notes per minute override")
    args = parser.parse_args()

    output_dir = args.output or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    if args.input:
        mp3s = [Path(args.input)]
    else:
        mp3s = sorted(Path(AUDIO_DIR).glob("*.mp3"))

    print(f"Found {len(mp3s)} MP3 files")
    print(f"Patterns: {', '.join(k for k, v in PATTERNS.items() if v)}")
    print(f"Min gap: {MIN_GAP_MS}ms, NPM: {'auto' if NOTES_PER_MINUTE == 0 else NOTES_PER_MINUTE}")

    success = 0
    failed = 0
    for i, mp3 in enumerate(mp3s):
        print(f"[{i+1}/{len(mp3s)}] {mp3.name}")
        if process_file(str(mp3), output_dir):
            success += 1
        else:
            failed += 1

    print(f"\nDone! {success} maps generated, {failed} failed")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
