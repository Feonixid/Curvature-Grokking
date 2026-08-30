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

N_SEEDS_CONFIRMED = 10    
N_SEEDS_ENCOURAGE = 24    
MAX_STEPS_CONFIRMED = 8000
MAX_STEPS_ENCOURAGE = 16000  

init_lock = threading.Lock()

def make_data(seed):
    rng = np.random.default_rng(seed)
    pairs = [(a, b) for a in range(P) for b in range(P)]
    rng.shuffle(pairs)
    n_train = int(len(pairs) * TRAIN_FRAC)
    train_pairs = pairs[:n_train]
    test_pairs = pairs[n_train:]

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
        x = self.out_proj(x)
        return x


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
        W1 = self.block1.out_proj.weight
        W2 = self.block2.out_proj.weight
        comm = W1 @ W2 - W2 @ W1
        return (comm ** 2).sum()


def train_one(condition, seed, device_str, max_steps, lr=1e-3, weight_decay=1.0, curv_weight=0.1):
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
    autocast_ctx = (lambda: torch.autocast(device_type="cuda", dtype=torch.float16)) if use_amp else (lambda: nullcontext())

    grok_step = None
    max_test_acc = 0.0
    history = {"step": [], "train_acc": [], "test_acc": [], "curvature": []}

    for step in range(max_steps):
        model.train()
        opt.zero_grad(set_to_none=True)
        with autocast_ctx():
            logits = model(tr_a, tr_b)
            loss = ce(logits, tr_y)
            curv = model.commutator_defect()
            if condition == "suppress":
                loss = loss + curv_weight * curv
            elif condition == "encourage":
                loss = loss - curv_weight * curv
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        is_final = (step == max_steps - 1)
        if step % EVAL_EVERY == 0 or is_final:
            model.eval()
            with torch.no_grad(), autocast_ctx():
                if is_final:
                    train_acc = (model(tr_a, tr_b).argmax(-1) == tr_y).float().mean().item()
                    test_acc = (model(te_a, te_b).argmax(-1) == te_y).float().mean().item()
                else:
                    train_acc = (model(tr_a[train_sub], tr_b[train_sub]).argmax(-1) == tr_y[train_sub]).float().mean().item()
                    test_acc = (model(te_a[test_sub], te_b[test_sub]).argmax(-1) == te_y[test_sub]).float().mean().item()
                curv_val = model.commutator_defect().item()
            history["step"].append(step)
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)
            history["curvature"].append(curv_val)
            max_test_acc = max(max_test_acc, test_acc)
            if grok_step is None and test_acc >= GROK_THRESHOLD:
                grok_step = step

    final_test_acc = history["test_acc"][-1]
    return dict(condition=condition, seed=seed, device=device_str,
                grok_step=grok_step if grok_step is not None else max_steps,
                ever_grokked=grok_step is not None,
                final_test_acc=final_test_acc,
                max_test_acc=max_test_acc,
                durable_grok=final_test_acc >= GROK_THRESHOLD,
                history=history)

jobs = []
for seed in range(N_SEEDS_CONFIRMED):
    jobs.append(("baseline", seed, MAX_STEPS_CONFIRMED))
    jobs.append(("suppress", seed, MAX_STEPS_CONFIRMED))
for seed in range(N_SEEDS_ENCOURAGE):
    jobs.append(("encourage", seed, MAX_STEPS_ENCOURAGE))

print(f"\nTotal jobs: {len(jobs)} across {len(DEVICES)} device(s)")

results_list = []
t0 = time.time()
n_workers = max(1, len(DEVICES))

with ThreadPoolExecutor(max_workers=n_workers) as executor:
    futures = {}
    for i, (cond, seed, max_steps) in enumerate(jobs):
        dev = DEVICES[i % len(DEVICES)]
        fut = executor.submit(train_one, cond, seed, dev, max_steps)
        futures[fut] = (cond, seed, dev)

    done_count = 0
    for fut in as_completed(futures):
        cond, seed, dev = futures[fut]
        r = fut.result()
        results_list.append(r)
        done_count += 1
        status = f"grokked@{r['grok_step']}" if r["ever_grokked"] else "did not grok"
        durability = "durable" if r["durable_grok"] else "LOST IT" if r["ever_grokked"] else "n/a"
        print(f"[{done_count}/{len(jobs)}] {cond:10s} seed {seed:2d} ({dev}) | {status:16s} | "
              f"{durability:8s} | final acc {r['final_test_acc']:.3f} | max acc {r['max_test_acc']:.3f} "
              f"| {time.time()-t0:.0f}s elapsed")

print(f"\nTotal time: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")

results = {"baseline": [], "suppress": [], "encourage": []}
for r in results_list:
    results[r["condition"]].append(r)

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

def cohens_d(a, b):
    a, b = np.array(a), np.array(b)
    pooled_std = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0.0

summary_rows = []
for cond in ["baseline", "suppress", "encourage"]:
    runs = results[cond]
    grok_steps = [r["grok_step"] for r in runs]
    ever_grokked_frac = np.mean([r["ever_grokked"] for r in runs])
    durable_frac = np.mean([r["durable_grok"] for r in runs])
    final_accs = [r["final_test_acc"] for r in runs]
    max_accs = [r["max_test_acc"] for r in runs]
    summary_rows.append(dict(
        condition=cond, n_seeds=len(runs),
        grok_step_mean=np.mean(grok_steps), grok_step_std=np.std(grok_steps),
        ever_grokked_frac=ever_grokked_frac, durable_grok_frac=durable_frac,
        final_test_acc_mean=np.mean(final_accs), max_test_acc_mean=np.mean(max_accs)))
summary_df = pd.DataFrame(summary_rows)
pd.set_option("display.width", 160)
print(summary_df.to_string(index=False))

print("\ngrok_step: does the intervention shift WHEN it first crosses threshold?")
base_steps = [r["grok_step"] for r in results["baseline"]]
for cond in ["suppress", "encourage"]:
    other_steps = [r["grok_step"] for r in results[cond]]
    t_stat, p_val = stats.ttest_ind(other_steps, base_steps)
    d = cohens_d(other_steps, base_steps)
    direction = "LATER" if np.mean(other_steps) > np.mean(base_steps) else "EARLIER"
    print(f"{cond:10s} vs baseline: t={t_stat:+.3f}, p={p_val:.4f}, Cohen's d={d:+.3f} -> {direction}")

print("\ndurable_grok_frac: does the intervention change whether grokking STICKS?")
for cond in ["suppress", "encourage"]:
    table = [[sum(r["durable_grok"] for r in results["baseline"]),
              len(results["baseline"]) - sum(r["durable_grok"] for r in results["baseline"])],
             [sum(r["durable_grok"] for r in results[cond]),
              len(results[cond]) - sum(r["durable_grok"] for r in results[cond])]]
    odds_ratio, p_val = stats.fisher_exact(table)
    print(f"{cond:10s} vs baseline (Fisher's exact test on durable_grok): "
          f"baseline {table[0][0]}/{sum(table[0])} durable, {cond} {table[1][0]}/{sum(table[1])} durable, "
          f"odds ratio={odds_ratio:.3f}, p={p_val:.4f}")

print("\nencourage specifically: blocked or just delayed?")
enc_runs = results["encourage"]
never_grokked = [r for r in enc_runs if not r["ever_grokked"]]
print(f"{len(never_grokked)}/{len(enc_runs)} encourage seeds never crossed {GROK_THRESHOLD} "
      f"within {MAX_STEPS_ENCOURAGE} steps.")
if never_grokked:
    print(f"Their max test accuracy ever reached: {[round(r['max_test_acc'],3) for r in never_grokked]}")

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

axes[0, 0].boxplot([[r["grok_step"] for r in results[c]] for c in ["baseline", "suppress", "encourage"]],
                    labels=["baseline", "suppress", "encourage"])
axes[0, 0].set_title("Step at first crossing 90% test acc")
axes[0, 0].set_ylabel("step")

bar_conds = ["baseline", "suppress", "encourage"]
ever = [np.mean([r["ever_grokked"] for r in results[c]]) for c in bar_conds]
durable = [np.mean([r["durable_grok"] for r in results[c]]) for c in bar_conds]
x = np.arange(len(bar_conds))
axes[0, 1].bar(x - 0.2, ever, width=0.4, label="ever grokked")
axes[0, 1].bar(x + 0.2, durable, width=0.4, label="durable at final step")
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(bar_conds)
axes[0, 1].set_title("Ever-grokked vs durable-grok fraction")
axes[0, 1].legend()

for cond, color in zip(bar_conds, ["tab:blue", "tab:red", "tab:green"]):
    h = results[cond][0]["history"]
    axes[1, 0].plot(h["step"], h["test_acc"], label=f"{cond} (seed {results[cond][0]['seed']})", color=color)
axes[1, 0].axhline(GROK_THRESHOLD, color="gray", linestyle="--")
axes[1, 0].set_title("Test accuracy over training")
axes[1, 0].set_xlabel("step")
axes[1, 0].legend(fontsize=8)

axes[1, 1].boxplot([[r["final_test_acc"] for r in results[c]] for c in bar_conds], labels=bar_conds)
axes[1, 1].set_title("Final test accuracy distribution")

plt.tight_layout()
plt.savefig("/kaggle/working/grokking_curvature_results_v2.png", dpi=140)
plt.show()

summary_df.to_csv("/kaggle/working/grokking_summary_v2.csv", index=False)
pd.DataFrame([{k: v for k, v in r.items() if k != "history"} for r in results_list]).to_csv(
    "/kaggle/working/grokking_raw_all_runs_v2.csv", index=False)
