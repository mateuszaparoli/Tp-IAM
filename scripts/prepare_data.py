"""Pré-processamento: JSON -> MIDI -> tokens REMI (+ aumento por transposição no treino)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chorales.data import ROOT, SPLITS, grid_to_score, load_split, pitch_bounds, write_midi
from chorales.tokenizer import (PITCH_RANGE, TOKENIZER_PATH, build_tokenizer, decode_ids,
                                encode_score, token_types)

TRANSPOSITIONS = range(-5, 7)   # 12 tonalidades


def main():
    tokenizer = build_tokenizer()
    TOKENIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(TOKENIZER_PATH)
    stats = {"vocab_size": len(tokenizer), "pitch_range_tokens": PITCH_RANGE, "splits": {}}
    tok_dir = ROOT / "data" / "tokens"
    tok_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        grids = load_split(split)
        sequences, transposed_count = [], 0
        for i, g in enumerate(grids):
            write_midi(grid_to_score(g), ROOT / "data" / "midi" / split / f"{split}_{i:03d}.mid")
            shifts = TRANSPOSITIONS if split == "train" else [0]
            for s in shifts:
                lo, hi = g[g >= 0].min() + s, g[g >= 0].max() + s
                if lo < PITCH_RANGE[0] or hi >= PITCH_RANGE[1]:
                    continue
                ids = encode_score(tokenizer, grid_to_score(g, transpose=s))
                sequences.append(ids)
                transposed_count += s != 0
                if split == "test" and s == 0:
                    write_midi(decode_ids(tokenizer, ids),
                               ROOT / "data" / "midi" / "test_ref" / f"test_{i:03d}.mid")
        (tok_dir / f"{split}.json").write_text(json.dumps(sequences))

        lengths = np.array([len(s) for s in sequences])
        original_lengths = np.array([len(encode_score(tokenizer, grid_to_score(g))) for g in grids])
        n_notes = [int(((np.diff(g, axis=0, prepend=-2) != 0) & (g >= 0)).sum()) for g in grids]
        types = {}
        for s in sequences[: len(grids)]:
            for t in token_types(tokenizer, s):
                types[t] = types.get(t, 0) + 1
        stats["splits"][split] = {
            "pieces": len(grids),
            "sequences_after_aug": len(sequences),
            "transposed_copies": transposed_count,
            "steps16_mean": float(np.mean([len(g) for g in grids])),
            "bars_mean": float(np.mean([len(g) for g in grids]) / 16),
            "notes_mean": float(np.mean(n_notes)),
            "tokens_mean_original": float(original_lengths.mean()),
            "tokens_std_original": float(original_lengths.std()),
            "tokens_min_original": int(original_lengths.min()),
            "tokens_max_original": int(original_lengths.max()),
            "tokens_total_after_aug": int(lengths.sum()),
            "pitch_min_max": pitch_bounds(grids),
            "token_type_share": {k: v / sum(types.values()) for k, v in types.items()},
        }
        print(f"{split}: {len(grids)} peças, {len(sequences)} sequências")

    stats["vocab_types"] = sorted({k.split("_")[0] for k in tokenizer.vocab})
    out = ROOT / "results" / "dataset_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2))
    print("vocab:", len(tokenizer))


if __name__ == "__main__":
    main()
