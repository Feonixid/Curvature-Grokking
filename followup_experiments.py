import torch, torch.nn as nn, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import time, warnings, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
warnings.filterwarnings("ignore")

N_GPUS = torch.cuda.device_count()
print("CUDA available:", torch.cuda.is_available(), "| GPU count:", N_GPUS)
DEVICES = [f"cuda:{i}" for i in range(N_GPUS)] if N_GPUS > 0 else ["cpu"]
print("Using devices:", DEVICES)

P = 97
TRAIN_FRAC = 0.5
GROK_THRESHOLD = 0.90
EVAL_EVERY = 200
EVAL_SUBSAMPLE = 1500
MAX_STEPS = 8000

init_lock = threading.Lock()


def make_data(seed):
    rng = np.random.default_rng(seed)
    pairs = [(a, b) for a in range(P) for b in range(P)]
    rng.shuffle(pairs)
    n_train = int(len(pairs) * TRAIN_FRAC)
    train_pairs, test_pairs = pairs[:n_train], pairs[n_train:]
    def to_tensors(pairs):
        a = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        b = torch.tensor([p[1] for p in pairs], dtype=torch.long)
        y = torch.tensor([(p[0] + p[1]) % P for p in pairs], dtype=torch.long)
        return a, b, y
    return to_tensors(train_pairs), to_tensors(test_pairs)


D_MODEL = 128
N_HEADS = 4

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(D_MODEL, N_HEADS, batch_first=True)
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.mlp = nn.Sequential(nn.Linear(D_MODEL, 4*D_MODEL), nn.ReLU(), nn.Linear(4*D_MODEL, D_MODEL))
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.out_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
    def forward(self, x, mask=None):
        a, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        x = self.ln1(x + a)
        x = self.ln2(x + self.mlp(x))
        return self.out_proj(x)

class GrokTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(P + 1, D_MODEL)
        self.pos_emb = nn.Embedding(3, D_MODEL)
        self.block1 = Block()
        self.block2 = Block()
        self.head = nn.Linear(D_MODEL, P)
    def forward(self, a, b):
        eq_tok = torch.full_like(a, P)
        toks = torch.stack([a, b, eq_tok], dim=1)
        pos = torch.arange(3, device=toks.device).unsqueeze(0).expand(toks.shape[0], -1)
        x = self.tok_emb(toks) + self.pos_emb(pos)
        x = self.block1(x)
        x = self.block2(x)
        return self.head(x[:, -1, :])
    def commutator_defect(self):
        W1, W2 = self.block1.out_proj.weight, self.block2.out_proj.weight
        comm = W1 @ W2 - W2 @ W1
        return (comm ** 2).sum()


def train_one(condition, seed, device_str, max_steps=MAX_STEPS, lr=1e-3, weight_decay=1.0, curv_weight=0.1):
    device = torch.device(device_str)
    use_amp = device.type == "cuda"
    (tr_a, tr_b, tr_y), (te_a, te_b, te_y) = make_data(seed)
    tr_a, tr_b, tr_y = tr_a.to(device), tr_b.to(device), tr_y.to(device)
    te_a, te_b, te_y = te_a.to(device), te_b.to(device), te_y.to(device)

    rng = np.random.default_rng(seed + 777)
    train_sub = torch.tensor(rng.choice(len(tr_a), size=min(EVAL_SUBSAMPLE, len(tr_a)), replace=False), device=device)
    test_sub = torch.tensor(rng.choice(len(te_a), size=min(EVAL_SUBSAMPLE, len(te_a)), replace=False), device=device)

    with init_lock:
        torch.manual_seed(seed)
        model = GrokTransformer().to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    actx = (lambda: torch.autocast(device_type="cuda", dtype=torch.float16)) if use_amp else (lambda: nullcontext())

    grok_step = None
    max_test_acc = 0.0
    test_acc = 0.0

    for step in range(max_steps):
        model.train()
        opt.zero_grad(set_to_none=True)
        with actx():
            logits = model(tr_a, tr_b)
            loss = ce(logits, tr_y)
            curv = model.commutator_defect()
            if condition == "suppress":
                loss = loss + curv_weight * curv
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        is_final = (step == max_steps - 1)
        if step % EVAL_EVERY == 0 or is_final:
            model.eval()
            with torch.no_grad(), actx():
                if is_final:
                    test_acc = (model(te_a, te_b).argmax(-1) == te_y).float().mean().item()
                else:
                    test_acc = (model(te_a[test_sub], te_b[test_sub]).argmax(-1) == te_y[test_sub]).float().mean().item()
            max_test_acc = max(max_test_acc, test_acc)
            if grok_step is None and test_acc >= GROK_THRESHOLD:
                grok_step = step

    final_test_acc = test_acc
    return dict(condition=condition, seed=seed, device=device_str, weight_decay=weight_decay,
                grok_step=grok_step if grok_step is not None else max_steps,
                ever_grokked=grok_step is not None,
                final_test_acc=final_test_acc, max_test_acc=max_test_acc,
                durable_grok=final_test_acc >= GROK_THRESHOLD)


def run_jobs(jobs, label):
    print(f"\n{'='*70}\n{label}: {len(jobs)} jobs across {len(DEVICES)} device(s)\n{'='*70}")
    results_list = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, len(DEVICES))) as executor:
        futures = {}
        for i, (cond, seed, wd) in enumerate(jobs):
            dev = DEVICES[i % len(DEVICES)]
            fut = executor.submit(train_one, cond, seed, dev, MAX_STEPS, 1e-3, wd)
            futures[fut] = (cond, seed, wd, dev)
        for n, fut in enumerate(as_completed(futures), 1):
            cond, seed, wd, dev = futures[fut]
            r = fut.result()
            results_list.append(r)
            status = f"grokked@{r['grok_step']}" if r["ever_grokked"] else "did not grok"
            dur = "durable" if r["durable_grok"] else ("LOST IT" if r["ever_grokked"] else "n/a")
            print(f"[{n}/{len(jobs)}] {cond:10s} seed {seed:2d} wd={wd} ({dev}) | {status:16s} | "
                  f"{dur:8s} | final {r['final_test_acc']:.3f} | {time.time()-t0:.0f}s")
    print(f"{label} done in {time.time()-t0:.1f}s")
    return results_list

EXTRA_SEEDS = range(10, 30)  # new seeds
followup1_jobs = []
for seed in EXTRA_SEEDS:
    followup1_jobs.append(("baseline", seed, 1.0))
    followup1_jobs.append(("suppress", seed, 1.0))
followup1_results = run_jobs(followup1_jobs, "FOLLOW-UP 1")

WD_ABLATION_SEEDS = range(15)
followup2_jobs = []
for seed in WD_ABLATION_SEEDS:
    followup2_jobs.append(("baseline", seed, 0.1))
    followup2_jobs.append(("suppress", seed, 0.1))
followup2_results = run_jobs(followup2_jobs, "FOLLOW-UP 2")

original_wd1 = [
    dict(condition="baseline", seed=0, weight_decay=1.0, grok_step=1000, ever_grokked=True, final_test_acc=1.000, durable_grok=True),
    dict(condition="baseline", seed=1, weight_decay=1.0, grok_step=1200, ever_grokked=True, final_test_acc=0.998, durable_grok=True),
    dict(condition="baseline", seed=2, weight_decay=1.0, grok_step=1200, ever_grokked=True, final_test_acc=1.000, durable_grok=True),
    dict(condition="baseline", seed=3, weight_decay=1.0, grok_step=1200, ever_grokked=True, final_test_acc=0.008, durable_grok=False),
    dict(condition="baseline", seed=4, weight_decay=1.0, grok_step=1000, ever_grokked=True, final_test_acc=0.513, durable_grok=False),
    dict(condition="baseline", seed=5, weight_decay=1.0, grok_step=1200, ever_grokked=True, final_test_acc=0.087, durable_grok=False),
    dict(condition="baseline", seed=6, weight_decay=1.0, grok_step=800,  ever_grokked=True, final_test_acc=1.000, durable_grok=True),
    dict(condition="baseline", seed=7, weight_decay=1.0, grok_step=1600, ever_grokked=True, final_test_acc=1.000, durable_grok=True),
    dict(condition="baseline", seed=8, weight_decay=1.0, grok_step=1200, ever_grokked=True, final_test_acc=1.000, durable_grok=True),
    dict(condition="baseline", seed=9, weight_decay=1.0, grok_step=800,  ever_grokked=True, final_test_acc=1.000, durable_grok=True),
    dict(condition="suppress", seed=0, weight_decay=1.0, grok_step=1000, ever_grokked=True, final_test_acc=0.049, durable_grok=False),
    dict(condition="suppress", seed=1, weight_decay=1.0, grok_step=1000, ever_grokked=True, final_test_acc=1.000, durable_grok=True),
    dict(condition="suppress", seed=2, weight_decay=1.0, grok_step=1200, ever_grokked=True, final_test_acc=1.000, durable_grok=True),
    dict(condition="suppress", seed=3, weight_decay=1.0, grok_step=1000, ever_grokked=True, final_test_acc=0.986, durable_grok=True),
    dict(condition="suppress", seed=4, weight_decay=1.0, grok_step=1200, ever_grokked=True, final_test_acc=1.000, durable_grok=True),
    dict(condition="suppress", seed=5, weight_decay=1.0, grok_step=1200, ever_grokked=True, final_test_acc=1.000, durable_grok=True),
    dict(condition="suppress", seed=6, weight_decay=1.0, grok_step=1200, ever_grokked=True, final_test_acc=1.000, durable_grok=True),
    dict(condition="suppress", seed=7, weight_decay=1.0, grok_step=1400, ever_grokked=True, final_test_acc=1.000, durable_grok=True),
    dict(condition="suppress", seed=8, weight_decay=1.0, grok_step=1400, ever_grokked=True, final_test_acc=1.000, durable_grok=True),
    dict(condition="suppress", seed=9, weight_decay=1.0, grok_step=1000, ever_grokked=True, final_test_acc=0.996, durable_grok=True),
]

all_wd1 = original_wd1 + [{k: v for k, v in r.items() if k != "device"} for r in followup1_results]
df_wd1 = pd.DataFrame(all_wd1)

print("\n" + "="*70)
print("n=30 each weight_decay=1.0")
print("="*70)
summary = df_wd1.groupby("condition").agg(
    n=("seed", "count"),
    durable_frac=("durable_grok", "mean"),
    ever_grokked_frac=("ever_grokked", "mean"),
    final_acc_mean=("final_test_acc", "mean"),
).reset_index()
print(summary.to_string(index=False))

table = pd.crosstab(df_wd1["condition"], df_wd1["durable_grok"])
print("\nContingency table (durable_grok):")
print(table)
odds_ratio, p_val = stats.fisher_exact(table.values)
print(f"\nFisher's exact test on durable_grok, baseline vs suppress (n=30 each): "
      f"odds ratio={odds_ratio:.3f}, p={p_val:.4f}")

df_wd01 = pd.DataFrame([{k: v for k, v in r.items() if k != "device"} for r in followup2_results])

print("\n" + "="*70)
print("="*70)
summary_wd01 = df_wd01.groupby("condition").agg(
    n=("seed", "count"),
    durable_frac=("durable_grok", "mean"),
    ever_grokked_frac=("ever_grokked", "mean"),
    final_acc_mean=("final_test_acc", "mean"),
).reset_index()
print("wd=0.1:")
print(summary_wd01.to_string(index=False))
print("\nwd=1.0 (from above, n=30):")
print(summary.to_string(index=False))
df_wd1.to_csv("/kaggle/working/extended_n30_wd1.csv", index=False)
df_wd01.to_csv("/kaggle/working/wd_ablation_wd01.csv", index=False)