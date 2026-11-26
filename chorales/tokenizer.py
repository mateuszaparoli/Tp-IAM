from __future__ import annotations


from miditok import REMI, TokenizerConfig, TokSequence

from .data import ROOT

TOKENIZER_PATH = ROOT / "data" / "tokenizer.json"
PITCH_RANGE = (33, 86)


def build_tokenizer() -> REMI:
    cfg = TokenizerConfig(
        pitch_range=PITCH_RANGE,
        beat_res={(0, 4): 4, (4, 12): 2},
        num_velocities=1,
        use_velocities=False,
        use_chords=False, use_rests=False, use_tempos=False,
        use_time_signatures=False, use_programs=False,
        use_pitchdrum_tokens=False,
        special_tokens=["PAD", "BOS", "EOS"],
    )
    return REMI(cfg)


def load_tokenizer() -> REMI:
    if TOKENIZER_PATH.exists():
        return REMI(params=TOKENIZER_PATH)
    return build_tokenizer()


def special_ids(tok: REMI) -> dict[str, int]:
    return {name: tok.vocab[f"{name}_None"] for name in ("PAD", "BOS", "EOS")}


def encode_score(tok: REMI, score) -> list[int]:
    res = tok.encode(score)
    seq = res[0] if isinstance(res, list) else res
    sp = special_ids(tok)
    return [sp["BOS"], *seq.ids, sp["EOS"]]


def decode_ids(tok: REMI, ids: list[int]):
    sp = special_ids(tok)
    ids = [i for i in ids if i not in sp.values()]
    return tok.decode([TokSequence(ids=ids)])


def token_types(tok: REMI, ids: list[int]) -> list[str]:
    inv = {v: k for k, v in tok.vocab.items()}
    return [inv[i].split("_")[0] for i in ids]
