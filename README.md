# Tabu_Search for 1|prec|ΣTj - Non-preemptive precedence tardiness tabu search


## Requirements
- Python 3.8+. 
- `pip install -r requirements.txt` (`matplotlib` is needed only for the plot in `main.ipynb`)

## Files
| File | Purpose |
|---|---|
| `tabu_search.py` | The algorithm: `preprocessing` (adjacency lists, job data), `initialise_schedule` (valid start via topological sort), `tardiness`, `check_precedence` (validity and penalised cost), `repair_schedule`, `tabu(...)` with `method='rejection'` or `'penalty'`, and the Appendix A data. |
| `main.ipynb` | Test cases importing tabu search algorithm from `tabu_search.py`. Questions 3.1–3.3 + writes `outputs/` and `results/`. Sanity checks for the code against the lecture example and Tutorial 3. |
| `outputs/` | Output of the experiments. <br> Format: indented "cand" lines = neighbours examined in cyclic order (swap, schedule, cost, delta = cost(current) - cost(neighbour), tabu?, decision); <br> "k=" lines = accepted solution x_k with its cost, validity, best feasible total tardiness so far (g_best) and tabu list T. <br> For penalty runs "cost" is the penalised cost g'(S); g_best is always plain total tardiness. |


## Usage
```bash
python tabu_search.py               # initialise tabu search algorithm  
jupyter execute main.ipynb          # lecture + Tutorial 3 examples. 3.1-3.3 answers
```

Tabu search usage on any data:
```python
from tabu_search import tabu
x_best, g_best, history = tabu(p, d, prec, K=1000, gamma=10, lmbda=5, L=20,
                               x0=None,              # None -> valid initial schedule
                               method="penalty", verbose=True)
# x_best: best schedule satisfying all precedences; g_best: its total tardiness
```

NOTE:
* Jobs are numbered `1..n`. `p[0]` and `d[0]` refer to job 1.
* `prec` is a list of edges `(u, v)` meaning that u must finish before v starts.
* An optional `w` gives weighted tardiness, which the tests use.
* One iteration k is one accepted move, so the search produces x_1 … x_K. Early termination is possible if no neighbour is admissible.
* Neighbours from adjacent interchanges, tried cyclically from position after the last accepted swap - first accept.
* A move is accepted if g(y) < g_best (aspiration), or if Δ = g(x_k) − g(y) > −γ and the job pair is not tabu.
* The tabu list holds the last L interchanged job pairs.
* Penalty method: g'(S) = g(S) + ω Σ_k max(0, h_k(S)), with h_k(S) = pos(u) − pos(v) for each edge (u, v)
