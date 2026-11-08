"""
experiments from CW questions 3.1-3.3
run: python experiments.py

  outputs/   EVERYTHING: a full trace for every run, summary.txt, the plot
  results/  TRIMMED copy: summary, the runs cited in the report,
            the plot and INDEX.txt 
"""

import os, contextlib, shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tabu_search import tabu, preprocessing, check_precedence, tardiness, p, d, prec, x0

OUT = "outputs"     
RES = "results"    
for folder in (OUT, RES):
    shutil.rmtree(folder, ignore_errors=True)   # start clean
    os.makedirs(folder)

# runs in results/ 
EXAMPLES_32 = [(0, 20), (100, 1), (10, 100)]   # (gamma, L) traces used in the 3.2 discussion
PLOT_OMEGAS = (10, 20, 50)                     # omegas shown in the 3.3 plot
EDD_OMEGA = 25                                 # omega reported for the EDD start
_, _, job_info = preprocessing(len(p), prec, p, d)
fmt = lambda x: ",".join(map(str, x))

def run(fname=None, **kw):
    """run tabu()
    opt: if fname is given, write full per-iteration printout to outputs/fname
    """
    if fname is None:
        return tabu(p, d, prec, 1000, kw["gamma"], kw.get("lmbda", 0), kw["L"], x0=kw["x0"],
                    method=kw.get("method", "rejection"))
    path = os.path.join(OUT, fname)
    os.makedirs(os.path.dirname(path), exist_ok=True)   # allows sub-folders, e.g. outputs/3_2_grid/
    with open(path, "w") as fh, contextlib.redirect_stdout(fh):
        # header so every trace file says which run it is
        print(f"# method={kw.get('method', 'rejection')}  K={kw.get('K', 1000)}  gamma={kw['gamma']}  "
              f"L={kw['L']}  omega={kw.get('lmbda', 0) if kw.get('method') == 'penalty' else '-'}  "
              f"x0=[{fmt(kw['x0'])}]")
        return tabu(p, d, prec, kw.get("K", 1000), kw["gamma"], kw.get("lmbda", 0), kw["L"],
                    x0=kw["x0"], method=kw.get("method", "rejection"), verbose=True)

summary = open(os.path.join(OUT, "summary.txt"), "w")
def out(s=""):
    print(s); summary.write(s + "\n")

# 3.1
out("=== 3.1 Rejection method, L=20, gamma=10, K=10,100,1000, forced x0")
out(f"x0 = [{fmt(x0)}], g(x0) = {tardiness(x0, job_info)}")
res31 = {}
for K in (10, 100, 1000): 
    xb, gb, hist = run(f"3_1_rejection_K{K}.txt", K=K, gamma=10, L=20, x0=x0)
    res31[K] = hist
    out(f"K={K:5d}: best tardiness {gb}, iterations run {len(hist)-1}, best schedule [{fmt(xb)}]")
    out("   new-best iterations: " + ", ".join(f"k={h['k']}:{h['g_best']}" for i, h in enumerate(hist)
                                              if i > 0 and h['g_best'] < hist[i-1]['g_best']))

# 3.2
out("\n=== 3.2 Rejection method, K=1000, grid over gamma and L (forced x0)")
gammas = [0, 1, 2, 5, 10, 20, 30, 50, 100]; Ls = [1, 5, 10, 20, 30, 50, 100]
# full trace of every (gamma, L) combination -> outputs/3_2_grid/gamma{g}_L{L}.txt
grid = {(g, L): run(f"3_2_grid/gamma{g}_L{L}.txt", gamma=g, L=L, x0=x0) for L in Ls for g in gammas}
out("L \\ gamma " + "".join(f"{g:>7}" for g in gammas))
for L in Ls:
    out(f"{L:>9} " + "".join(f"{grid[(g, L)][1]:>7}" for g in gammas))
out("iterations actually run (<1000 means stopped: no admissible neighbour):")
for L in Ls:
    out(f"{L:>9} " + "".join(f"{len(grid[(g, L)][2])-1:>7}" for g in gammas))
bg, bL = min(grid, key=lambda k: (grid[k][1], k[1], k[0]))
out(f"Best: gamma={bg}, L={bL}, tardiness={grid[(bg, bL)][1]}, schedule=[{fmt(grid[(bg, bL)][0])}]")
run(f"3_2_best_gamma{bg}_L{bL}.txt", gamma=bg, L=bL, x0=x0)   # copy of the best run, easy to find

# 3.3
out("\n=== 3.3 Penalty method, L=20, gamma=10, K=1000, forced x0")
omegas = [0.5, 1, 2, 5, 10, 12, 15, 18, 20, 25, 50, 100]; pen = {}
out(f"{'omega':>7} {'best feasible T':>16} {'% iters infeasible':>19} {'iters run':>10}")
for om in omegas:
    xb, gb, hist = run(f"3_3_penalty_omega{om}.txt", gamma=10, L=20, x0=x0, method="penalty", lmbda=om)
    pen[om] = hist
    frac = sum(not h['valid'] for h in hist[1:]) / max(1, len(hist) - 1)
    out(f"{om:>7} {gb:>16} {100*frac:>18.1f}% {len(hist)-1:>10}")

plt.figure(figsize=(7, 4.2))
curve = lambda hist: ([h['k'] for h in hist], [h['g_best'] for h in hist])
plt.step(*curve(res31[1000]), where="post", label="rejection", lw=5, alpha=.35, color="k")
for om in PLOT_OMEGAS:
    plt.step(*curve(pen[om]), where="post", label=f"penalty, ω={om}", ls="--" if om == PLOT_OMEGAS[-1] else "-")
plt.xlabel("iteration k"); plt.ylabel("best total tardiness (feasible schedules)")
plt.title("Tabu search, L=20, γ=10, K=1000, same x0"); plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT, "3_3_convergence.png"), dpi=150)

edd = sorted(range(1, len(d) + 1), key=lambda j: (d[j-1], j))   # EDD, ties -> lowest index
out(f"\nEDD x0 = [{fmt(edd)}], g = {tardiness(edd, job_info)}, "
    f"feasible = {check_precedence(edd, prec, 0, job_info)[0]}")
for om in omegas:
    xb, gb, hist = run(f"3_3_penalty_EDD_omega{om}.txt", gamma=10, L=20, x0=edd, method="penalty", lmbda=om)
    first = next((h['k'] for h in hist if h['valid']), None)
    frac = sum(not h['valid'] for h in hist[1:]) / max(1, len(hist) - 1)
    out(f"  omega={om:>6}: first feasible visited at k={first}, returned T={gb}"
        f"{' (repaired, no feasible schedule visited)' if first is None else ''}, "
        f"% infeasible={100*frac:.1f}%, iters={len(hist)-1}")
summary.close()

# ---------------------------------------------------------------------------------------
# trimmed copy for the submission -> results/
# ---------------------------------------------------------------------------------------
best32 = f"3_2_best_gamma{bg}_L{bL}.txt"
copies = [  # (file in outputs/, name in results/, description for INDEX.txt)
    ("summary.txt", "summary.txt", "all tables reported in the PDF (3.1, 3.2, 3.3)"),
    ("3_1_rejection_K10.txt", None, "Q3.1  rejection, gamma=10, L=20, K=10"),
    ("3_1_rejection_K100.txt", None, "Q3.1  K=100"),
    ("3_1_rejection_K1000.txt", None, "Q3.1  K=1000"),
    (best32, None, f"Q3.2  best combination found (total tardiness {grid[(bg, bL)][1]})"),
]
for g, L in EXAMPLES_32:
    copies.append((f"3_2_grid/gamma{g}_L{L}.txt", f"3_2_gamma{g}_L{L}.txt",
                   f"Q3.2  gamma={g}, L={L}: best {grid[(g, L)][1]}, "
                   f"{len(grid[(g, L)][2]) - 1} iterations run"))
for om in PLOT_OMEGAS:
    copies.append((f"3_3_penalty_omega{om}.txt", None, f"Q3.3  penalty, omega={om} (plotted)"))
copies.append((f"3_3_penalty_EDD_omega{EDD_OMEGA}.txt", None,
               f"Q3.3  penalty from the EDD initial schedule, omega={EDD_OMEGA}"))
copies.append(("3_3_convergence.png", None, "Q3.3  best feasible total tardiness vs iteration"))

for src, dst, _ in copies:
    shutil.copy(os.path.join(OUT, src), os.path.join(RES, dst or src))

print(f"\nfull printouts -> {OUT}/   trimmed submission set -> {RES}/")