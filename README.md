# Rhythia Map Generator

A Python-based map generator for [Rhythia](https://store.steampowered.com/app/1878910/Rhythia/) that creates `.phxm` map files from MP3 audio.

Made by [opencode](https://opencode.ai) — an AI coding assistant.

## What it does

Takes audio files and generates rhythm game maps by:

- **Transient detection** — finds major audio energy changes (beat hits, vocal attacks, drops)
- **Vocal separation** — uses HPSS to isolate harmonic (vocal) and percussive layers, prioritizing vocal rhythm
- **Context-aware patterns** — snakes on pitch direction, spirals on fast sections, islands on sustained high-energy sections
- **Double/triple notes** — rapid-fire stacked notes on big spikes for emphasis
- **Gaps** — natural rests where energy drops

## Usage

### Batch mode (all MP3s in a folder)

Point `AUDIO_DIR` and `OUTPUT_DIR` at the top of `generate.py` to your folders, then:

```bash
python3 generate.py
```

It will process every `.mp3` in `AUDIO_DIR` and create a `.phxm` map for each one in `OUTPUT_DIR`. Drop hundreds of songs in and let it run.

### Single file

```bash
python3 generate.py -i /path/to/song.mp3 -o /path/to/output
```

## Settings

All settings are hardcoded at the top of `generate.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `AUDIO_DIR` | `~/Downloads/yt_music_recaps/all_songs` | Input folder of MP3s |
| `OUTPUT_DIR` | `~/Downloads/yt_music_recaps/all_maps` | Output folder for `.phxm` maps |
| `MIN_GAP_MS` | `40` | Minimum gap between notes in ms |
| `FIXED_DIFFICULTY` | `3` | Map difficulty (0=NA, 1=Easy, 2=Normal, 3=Hard, 4=Insane, 5=Illogical) |

## Requirements

```bash
pip install librosa numpy
```

## Map format

`.phxm` is the Rhythia map format — a ZIP containing:
- `metadata.json` — map info (title, difficulty, length, etc.)
- `objects.phxmo` — binary note data (timestamps + grid positions)
- `audio.mp3` — the song

Coordinates range from -1 to 1 on both axes (3x3 grid).
