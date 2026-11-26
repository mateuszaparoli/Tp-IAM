from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from symusic import Note, Score, Track

ROOT = Path(__file__).resolve().parents[1]
RAW_JSON = ROOT / "data" / "raw" / "Jsb16thSeparated.json"
MIDI_DIR = ROOT / "data" / "midi"
SPLITS = ("train", "valid", "test")

TPQ = 480                 # ticks por semínima dos MIDIs gerados
STEP = TPQ // 4           # uma semicolcheia
STEPS_PER_BAR = 16        # todos os corais tratados como 4/4
VELOCITY = 80
TEMPO_BPM = 80


def load_split(split: str) -> list[np.ndarray]:
    data = json.loads(RAW_JSON.read_text())
    return [np.asarray(piece, dtype=np.int16) for piece in data[split]]


def pickup_steps(grid: np.ndarray) -> int:
    rem = len(grid) % STEPS_PER_BAR
    return (STEPS_PER_BAR - rem) % STEPS_PER_BAR


def grid_to_notes(grid: np.ndarray, offset_steps: int = 0) -> list[tuple[int, int, int]]:
    notes = []
    T = len(grid)
    for v in range(grid.shape[1]):
        col = grid[:, v]
        t = 0
        while t < T:
            p = int(col[t])
            u = t + 1
            while u < T and col[u] == p:
                u += 1
            if p >= 0:
                notes.append((t + offset_steps, u - t, p))
            t = u
    notes.sort()
    return notes


def notes_to_score(notes, program: int = 0) -> Score:
    score = Score(TPQ)
    track = Track(program=program, is_drum=False, name="SATB")
    for start, dur, pitch in notes:
        track.notes.append(Note(start * STEP, dur * STEP, pitch, VELOCITY))
    score.tracks.append(track)
    return score


def grid_to_score(grid: np.ndarray, transpose: int = 0) -> Score:
    g = grid.copy()
    if transpose:
        g[g >= 0] += transpose
    return notes_to_score(grid_to_notes(g, pickup_steps(grid)))


def write_midi(score: Score, path: Path, tempo_bpm: int = TEMPO_BPM) -> None:
    """Grava o MIDI com andamento fixo (o REMI usado não codifica andamento)."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(resolution=TPQ, initial_tempo=tempo_bpm)
    inst = pretty_midi.Instrument(program=19, name="SATB")  # órgão de igreja (GM 20)
    sec_per_tick = 60.0 / (tempo_bpm * score.ticks_per_quarter)
    for track in score.tracks:
        for n in track.notes:
            inst.notes.append(pretty_midi.Note(
                velocity=int(n.velocity) or VELOCITY, pitch=int(n.pitch),
                start=n.time * sec_per_tick, end=(n.time + n.duration) * sec_per_tick))
    pm.instruments.append(inst)
    path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(path))


def pitch_bounds(grids) -> tuple[int, int]:
    vals = np.concatenate([g[g >= 0] for g in grids])
    return int(vals.min()), int(vals.max())
