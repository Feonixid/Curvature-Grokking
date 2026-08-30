# Curvature-Grokking

Does the "commutator defect" — a measure of non-commutativity between two
transformer blocks' weight-update directions — actually *cause* grokking, or
does it just correlate with it?

Prior work identified the commutator defect as a signal that reliably rises
before generalization onset in grokking (delayed, abrupt generalization on
small algorithmic tasks). That work was observational: the defect was
measured, never intervened on. This repo runs the causal test.

## Result

Training a 2-block transformer on modular addition (mod 97) under three
conditions — unmodified baseline, an auxiliary loss that **suppresses** the
commutator, and one that **rewards** it — across 44 seeded runs:

| Condition | Median steps to grok | Durable grok rate |
|---|---|---|
| Baseline | 1200 | 70% (n=10) |
| Suppress commutator | 1200 | 90% (n=10), 90% at n=30 |
| Encourage commutator | 1800 | 17% |

- **Encouraging** the commutator defect significantly delays grokking
  (Kaplan-Meier log-rank p = 0.0002) and sharply cuts how often the model
  reaches and *keeps* a generalizing solution (17% vs. 70% durable,
  Fisher's exact p = 0.0048).
- **Suppressing** it trends toward more durable grokking (odds ratio 2.74
  at n=30) but doesn't reach significance (p = 0.30).

So the relationship is causal, but asymmetric: pushing the defect up reliably
hurts generalization; pushing it down helps, but not conclusively yet.

A follow-up ablation testing whether baseline instability is a weight-decay
artifact hit a floor effect instead (wd=0.1 blocks grokking entirely within
the step budget) — left as an open question rather than forced into a
conclusion.

Full write-up, related work, and methodology: [`paper_grokking_curvature.pdf`](paper_grokking_curvature.pdf)
(source: [`paper_grokking_curvature.tex`](paper_grokking_curvature.tex)).

## Repo contents

| File | What it is |
|---|---|
| `grokking_curvature_test.py` | Main experiment: trains all 44 runs (baseline/suppress/encourage), computes t-tests, Cohen's d, Fisher's exact test, and plots results. |
| `followup_experiments.py` | Extends suppress/baseline to n=30 seeds and runs the weight-decay ablation. |
| `analysis.py` | Kaplan-Meier survival analysis and log-rank tests on time-to-grok. |
| `grokking_curvature.csv`, `Follow-UPs.csv`, `KAPLAN-MEIER.csv` | Logged output from the above runs. |
| `km_curves.png` | Kaplan-Meier survival curves by condition. |
| `paper_grokking_curvature.tex` / `.pdf` | Full paper. |

## Running it

Requires PyTorch, numpy, pandas, scipy, matplotlib, and `lifelines` (for
`analysis.py`). GPU is optional  the scripts auto detect CUDA and fall back
to CPU but expect the full 44 run sweep to take a lot on CPU.

```bash
pip install torch numpy pandas scipy matplotlib lifelines

python grokking_curvature_test.py     # main causal test, ~44 runs
python followup_experiments.py        # n=30 extension + weight-decay ablation
python analysis.py                    # Kaplan-Meier + log-rank tests
```

Note: the scripts currently hardcode output paths under `/kaggle/working/` —
update those paths if running locally.

## Citing

If you use this code or build on the result, please cite the accompanying
paper (see `paper_grokking_curvature.tex` for full references and BibTeX
context).
