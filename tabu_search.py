from typing import List, Tuple, Dict, Set, Optional
from collections import deque
import math

r"""
Minimise $\sum_{}^{j}T_j$ where $T_j = max(0, C_j-d_j)$

Solve a $1|prec|\sum_{}^{j}T_j$ problem with rejection method and penalty function method.

**Input:** DAG (in adjacency graph)

**Output:** Schedule (ordered list)
"""

def preprocessing(num_jobs: int, prec: List[Tuple[int, int]], p: List[int], d: List[int],
                  w: Optional[List[float]] = None
                  ) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]], Dict[int, Dict[str, int]]]:
    # GRETA: added w (weights for tutorial example checks - optional so it doesnt change anything) 

    successors = {i: set() for i in range(1, num_jobs + 1)}
    predecessors = {i: set() for i in range(1, num_jobs + 1)}

    # Adjacency list
    for u, v in prec:
        successors[u].add(v)
        predecessors[v].add(u)

    # Processing time and due date
    # GRETA: added weight (1 if w is None, else w[i-1]) for tutorial example checks
    job_info = {i: {'processing time': p[i-1], 'due date': d[i-1],
                    'weight': 1 if w is None else w[i-1]} for i in range(1, num_jobs + 1)}

    return successors, predecessors, job_info



def initialise_schedule(num_jobs: int, pred: Dict[int, Set[int]], suc: Dict[int, Set[int]]
                        ) -> List[int]:
    '''
        BFS for initialising a valid schedule

        Inputs:
            num_jobs: number of jobs need to be scheduled
            pred: predecessor dictionary
            suc: successor dictionary

        Return:
            a valid schedule that respect the precedences.
    '''
    # Kahn's topological sort
    schedule = []

    in_degree = {i: len(pred[i]) for i in range(1, num_jobs + 1)}
    queue = deque([job for job, deg in in_degree.items() if deg == 0])

    while queue:
        curr = queue.popleft()
        schedule.append(curr)

        for s in suc[curr]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
    return schedule


# GRETA: extra fallback function -> repair infeasable schedule to feasible one (for penalty method, if no feasible schedule is visited)
def repair_schedule(x: List[int], pred: Dict[int, Set[int]], suc: Dict[int, Set[int]]) -> List[int]:
    rank = {job: i for i, job in enumerate(x)}
    in_degree = {j: len(pred[j]) for j in pred}
    ready = [j for j in pred if in_degree[j] == 0]
    out = []
    while ready:
        ready.sort(key=lambda j: rank[j])
        curr = ready.pop(0)
        out.append(curr)
        for s in suc[curr]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                ready.append(s)
    return out



def tardiness(x: List[int], job_info: Dict[int, Dict[str, int]]) -> int:
    completion_time = 0
    tardilist = []

    for job in x:
        completion_time += job_info[job]['processing time']
        # GRETA: added weight (1 if w is None, else w[i-1]) for tutorial example checks
        tardilist.append(job_info[job]['weight'] * max(0, completion_time - job_info[job]['due date']))

    return sum(tardilist)


def check_precedence(x: List[int], prec: List[Tuple[int, int]], lmbda: float,
                     job_info: Dict[int, Dict[str, int]], method: str = 'rejection'
                     ) -> Tuple[bool, float]:
    '''
        Check if precedence of a schedule violates the rules.

        Inputs:
            x: current job schedule
            prec: a list of all precedences (u, v) where u is the predecessor of v
            lmbda: penalty weight parameter (omega)
            job_info: containing processing times and due dates per job.
            method: choose either 'rejection' for rejection method or 'penalty' for penalty function method.

        Returns:
            (valid_precedence, total_penalty_cost)

        Constraint functions (one per precedence k = (u, v)) - by how many positions job u is placed after job v:
            h_k(S) = pos_S(u) - pos_S(v)  <= 0
        (positions are distinct, so h_k <= 0 exactly when u comes before v)
    '''

    # positional measure
    position = {job: i for i, job in enumerate(x)}
    penalty = 0
    is_valid = True

    for u, v in prec:
        # GRETA moved the guard (silent error) to KeyError - missing job should raise an error instead of silently counting as "no violation"
        h_k = position[u] - position[v]
        if h_k > 0:
            is_valid = False
            penalty += lmbda * max(0, h_k)  # h_k measures the magnitude of the violation

    if method == 'rejection':
        return (is_valid, 0.0)

    elif method == 'penalty':
        cost = tardiness(x, job_info) + penalty
        return (is_valid, cost)

    else:
        raise ValueError("Invalid method specified. Options: 'rejection', 'penalty'.")


def tabu(p: List[int], d: List[int], prec: List[Tuple[int, int]], K: int, gamma: float, lmbda: float, L: int,
         x0=None, method='rejection', w=None, verbose=False):
    '''
        Compute the best schedule based on tabu search.

        Inputs:
            p: processing time
            d: due dates
            prec: a list of all precedences (u, v) where u is the predecessor of v (jobs numbered 1..n)
            K: maximum iterations (one iteration = one accepted move, producing x_1 ... x_K)
            gamma: threshold
            lmbda: penalty weight (omega), only used by the penalty method
            L: tabu list length
            x0: initial schedule (if given)
            method: choose either 'rejection' for rejection method or 'penalty' for penalty function method.
            w: optional weights (None = total tardiness)
            verbose: print every candidate and every accepted solution

        Returns:
            (x_best, g_best, history): best schedule satisfying all precedences, its total
            tardiness, and one record per iteration (for the 3.3 table and plot)
    '''
    # GRETA: num_jobs = max(max(u, v) for u, v in prec) gives the largest job index that appears in a prec not num jobs
    num_jobs = len(p)
    # GRETA: preprocessing returns (successors, predecessors, ...) - name swap
    suc, pred, job_info = preprocessing(num_jobs, prec, p, d, w)
    if not x0:
        x0 = initialise_schedule(num_jobs, pred, suc)
    x0 = list(x0)  

    # GRETA: def cost function actually minimised by the search: rejection -> g(S) = total tardiness; penalty -> g'(S) = g(S) + omega * sum max(0, h_k).
    def cost_of(x):
        valid, pen_cost = check_precedence(x, prec, lmbda, job_info, method)
        return valid, (pen_cost if method == 'penalty' else tardiness(x, job_info))

    # Note: the rejection method needs a feasible starting point 
    if method == 'rejection' and not cost_of(x0)[0]:
        raise ValueError("Rejection method needs an initial schedule satisfying all precedences")

    tabu_list = []
    t_best = math.inf # g_best from pseudocode
    x_best = x0 # best schedule
    x_curr = x0 # GRETA: current schedule (x_k in the pseudocode)
    start = 0 # GRETA: start index for cyclic enumeration of neighbours 
    # GRETA: track best schedule satisfying all precedences
    x_best_feasible, g_best_feasible = None, math.inf
    history = []

    for k in range(K+1):
        # GRETA: track first-accept (first admissible neighbour) - no list needed
        if k == 0: # iteration 0
            # GRETA: g_best should be measure with penalised cost g' for the penalty method
            valid0, c_curr = cost_of(x0)
            t_best = c_curr                       # g_best of the pseudocode (aspiration test)
            if valid0:
                x_best_feasible, g_best_feasible = list(x0), tardiness(x0, job_info)
            history.append(dict(k=0, x=list(x0), cost=c_curr, valid=valid0, g_best=g_best_feasible))
            if verbose:
                print(f"k=0  x={x0}  cost={c_curr}  valid={valid0}  g_best={g_best_feasible}")
        else: # all other iterations
            # GRETA: visit neighbours cyclically from `start` (not always pos 0) - stop at the first admissible neighbour (first-accept)
            accepted = None
            for step in range(num_jobs - 1):
                i = (start + step) % (num_jobs - 1)

                # GRETA: tabu list holds job pair interchanged instead of pos
                # GRETA: check if rec is tabu - dont just skip cz of aspiration criterion 
                x_tmp = x_curr.copy()
                x_tmp[i], x_tmp[i+1] = x_tmp[i+1], x_tmp[i] # swap from current solution
                pair = (min(x_tmp[i], x_tmp[i+1]), max(x_tmp[i], x_tmp[i+1]))
                is_tabu = pair in tabu_list

                # GRETA: new func cost_of() returns (valid, cost) -> if rejection and invalid need to skip
                valid_schedule, c = cost_of(x_tmp)
                if method == 'rejection' and not valid_schedule:
                    if verbose:
                        print(f"    cand pos {i+1}-{i+2} swap {pair}: violates precedence -> skipped")
                    continue

                # GRETA: lecture pseudocode line8 (Delta > -gamma and (i,j) not tabu)  or  g(y) < g_best with Delta = g(x_k) - g(y)
                # compares with current sol not g_best. + aspiration criterion (g(y) < g_best) was missing
                delta = c_curr - c
                accept = (delta > -gamma and not is_tabu) or c < t_best
                if verbose:
                    print(f"    cand pos {i+1}-{i+2} swap {pair}: {x_tmp} cost={c} delta={delta} "
                          f"tabu={is_tabu} -> {'ACCEPT' if accept else 'reject'}")
                if accept:
                    accepted = (x_tmp, c, valid_schedule, pair, i)
                    break

            if accepted is None:
                if verbose:
                    print(f"k={k}: no admissible neighbour -> terminate") # from specs
                break

            x_curr, c_curr, valid_curr, pair, i = accepted # GRETA: always move to accepted neighbour (even if worse)
            t_best = min(t_best, c_curr) # GRETA: update best if improved
            if valid_curr and tardiness(x_curr, job_info) < g_best_feasible:
                x_best_feasible, g_best_feasible = list(x_curr), tardiness(x_curr, job_info)

            if pair in tabu_list:
                tabu_list.remove(pair) # GRETA: if already in list (aspiration move) remove first and place it at the back of the list again
            tabu_list.append(pair)
            while len(tabu_list) > L: # GRETA: trim after appending so list doesnt grow to L+1 entries
                tabu_list.pop(0)
            start = (i + 1) % (num_jobs - 1)   # GRETA: cyclic enumeration 

            history.append(dict(k=k, x=list(x_curr), cost=c_curr, valid=valid_curr, g_best=g_best_feasible))
            if verbose:
                print(f"k={k}  x={x_curr}  cost={c_curr}  valid={valid_curr}  "
                      f"g_best={g_best_feasible}  T={tabu_list}")


    if x_best_feasible is None: # GRETA: if no feasible schedule visited, repair best penalised
        best_pen = min(history, key=lambda h: h['cost'])['x']
        x_best_feasible = repair_schedule(best_pen, pred, suc)
        g_best_feasible = tardiness(x_best_feasible, job_info)
        if verbose:
            print(f"No feasible schedule visited; repaired best penalised schedule -> "
                  f"{x_best_feasible}, tardiness {g_best_feasible}")
    if verbose:
        print(f"RESULT: best feasible schedule {x_best_feasible}, total tardiness {g_best_feasible}")
    return x_best_feasible, g_best_feasible, history # GRETA return extra cost and history for 3.3 table and plot


"""
Test cases
"""
# GRETA: we're numbering jobs 1..31, so the precedence list should be the MATLAB one in appendix A
prec0 = [(0, 1), (0, 2), (0, 4), (0, 5), (0, 6), (1, 7), (1, 8), (2, 3), (2, 9),
         (3, 10), (4, 11), (5, 12), (6, 13), (7, 14), (8, 14), (9, 15), (10, 19),
         (11, 16), (12, 17), (13, 20), (14, 18), (15, 19), (16, 26), (17, 20),
         (18, 24), (19, 21), (20, 21), (20, 22), (21, 23), (21, 25), (22, 25),
         (23, 27), (24, 26), (25, 26), (26, 28), (27, 29), (28, 29), (29, 30)
] # write in report the assumption for input
prec = [(u + 1, v + 1) for u, v in prec0]   # jobs numbered 1..31

p = [2, 6, 14, 3, 3, 5, 2, 3, 2, 3, 2, 3, 6, 2, 10, 2, 2, 5, 18, 6, 14, 6, 2, 5, 5, 6, 5, 5, 10, 3, 10]
d = [329, 307, 340, 122, 73, 287, 61, 164, 256, 181, 88, 279, 290, 18, 217, 71, 307, 295, 209, 77, 217, 233, 71, 355, 141, 34, 93, 253, 82, 172, 269]

# initial solution forced by question 3.1
x0 = [1, 2, 3, 6, 13, 5, 12, 17, 7, 4, 10, 14, 9, 8, 18, 21, 11, 16, 15, 20, 19, 22, 24, 23,
      25, 26, 27, 28, 29, 30, 31]

if __name__ == "__main__":
    # TEST: question 3.1 with K = 10
    tabu(p, d, prec, K=10, gamma=10, lmbda=0, L=20, x0=x0, method='rejection', verbose=True)