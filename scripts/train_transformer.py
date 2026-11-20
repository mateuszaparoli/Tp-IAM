"""Treino do Transformer autoregressivo."""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chorales.data import ROOT
from chorales.model import ModelConfig, MusicTransformer, sequence_nll
from chorales.tokenizer import load_tokenizer, special_ids


def batches(seqs, batch_size, max_len, pad_id, rng):
    """Uma época: cada sequência uma vez, com recorte aleatório se exceder o contexto."""
    order = rng.permutation(len(seqs))
    for b in range(0, len(order), batch_size):
        chunk = []
        for i in order[b:b + batch_size]:
            s = seqs[i]
            if len(s) > max_len + 1:
                st = int(rng.integers(0, len(s) - max_len))
                s = s[st:st + max_len + 1]
            chunk.append(s)
        T = max(len(s) for s in chunk)
        arr = np.full((len(chunk), T), pad_id, dtype=np.int64)
        for r, s in enumerate(chunk):
            arr[r, :len(s)] = s
        t = torch.from_numpy(arr)
        yield t[:, :-1], t[:, 1:]


@torch.no_grad()
def evaluate(model, seqs, device):
    model.eval()
    total_loss, n = 0.0, 0
    for s in seqs:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            a, b = sequence_nll(model, s, device)
        total_loss += a
        n += b
    model.train()
    return total_loss / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="checkpoints/transformer.pt")
    parser.add_argument("--history", default="results/transformer_history.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = load_tokenizer()
    sp = special_ids(tok)
    train = json.loads((ROOT / "data/tokens/train.json").read_text())
    valid = json.loads((ROOT / "data/tokens/valid.json").read_text())

    config = ModelConfig(vocab_size=len(tok), max_len=args.max_len, d_model=args.d_model,
                      n_layers=args.layers, n_heads=args.heads, d_ff=args.ff,
                      dropout=args.dropout, pad_id=sp["PAD"])
    model = MusicTransformer(config).to(device)
    print(f"Treinando em {device}")

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.98))
    steps_per_epoch = math.ceil(len(train) / args.batch)
    total = steps_per_epoch * args.epochs
    warmup = max(1, min(500, total // 10))
    def learning_rate_factor(step):
        warmup_factor = min(1.0, (step + 1) / warmup)
        progress = min(step, total) / total
        cosine_factor = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        return warmup_factor * cosine_factor

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    history, best, epochs_without_improvement = [], float("inf"), 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_tokens = 0.0, 0
        for x, y in batches(train, args.batch, args.max_len, sp["PAD"], rng):
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                loss = model.loss(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            token_count = int((y != sp["PAD"]).sum())
            train_loss += loss.item() * token_count
            train_tokens += token_count
        valid_loss = evaluate(model, valid, device)
        record = {"epoch": epoch, "train_nll": train_loss / train_tokens, "valid_nll": valid_loss, "valid_ppl": math.exp(valid_loss),
               "lr": scheduler.get_last_lr()[0], "time_s": time.time() - t0}
        history.append(record)
        print(f"Época {epoch}: treino {train_loss / train_tokens:.3f}, validação {valid_loss:.3f}")
        if valid_loss < best - 1e-4:
            best, epochs_without_improvement = valid_loss, 0
            torch.save({"config": model.to_dict(), "state_dict": model.state_dict(),
                        "epoch": epoch, "valid_nll": valid_loss, "args": vars(args)}, out)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("parada antecipada")
                break

    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    (ROOT / args.history).write_text(json.dumps(
        {"history": history, "best_valid_nll": best, "n_params": model.n_params(),
         "args": vars(args), "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"},
        indent=2))
    print(f"melhor NLL de validação: {best:.4f} (ppl {math.exp(best):.3f})")


if __name__ == "__main__":
    main()
