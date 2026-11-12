from typing import List, Tuple, Dict, Set, Optional
from collections import deque
import math

def preprocessing(num_jobs: int, prec: List[Tuple[int, int]], p: List[int], d: List[int],
                  w: Optional[List[float]] = None
                  ) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]], Dict[int, Dict[str, int]]]:
    ''' 
        Preprocess input data to build precedence graphs and metadata dictionaries.

        Inputs:
            num_jobs: total number of jobs to be scheduled
            prec: list of directed precedence constraints (u, v) where u must precede v
            p: processing times for each job
            d: due dates for each job
            w: optional weights for each job (defaults to 1.0 for unweighted tardiness)

        Returns:
            successors: dictionary mapping each job ID to its set of immediate successors
            predecessors: dictionary mapping each job ID to its set of immediate predecessors
            job_info: nested dictionary containing 'processing time', 'due date', and 'weight' per job
    '''
    successors = {i: set() for i in range(1, num_jobs + 1)}
    predecessors = {i: set() for i in range(1, num_jobs + 1)}

    # Adjacency list
    for u, v in prec:
        successors[u].add(v)
        predecessors[v].add(u)

    # Processing time and due date
    job_info = {i: {'processing time': p[i-1], 'due date': d[i-1],
                    'weight': 1 if w is None else w[i-1]} for i in range(1, num_jobs + 1)} # added weight for testing on weighted jobs, weight is 1 in this setting.

    return successors, predecessors, job_info



def initialise_schedule(num_jobs: int, pred: Dict[int, Set[int]], suc: Dict[int, Set[int]]
                        ) -> List[int]:
    ''' 
        BFS using Kahn's topological sort for initialising a valid schedule

        Inputs:
            num_jobs: number of jobs need to be scheduled
            pred: predecessor dictionary
            suc: successor dictionary

        Return:
            a valid schedule that respect the precedences.
    '''
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



def repair_schedule(x: List[int], pred: Dict[int, Set[int]], suc: Dict[int, Set[int]]) -> List[int]:
    '''
        Repair infeasable schedule to feasible one (for penalty method, if no feasible schedule is visited) using Kahn's topological sort.

        Inputs:
            x: current job schedule
            pred: predecessor dictionary
            suc: successor dictionary

        Return:
            A valid schedule where every predecessor is placed before its successors, preserving much of the original ordering in x.
    '''
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
        tardilist.append(job_info[job]['weight'] * max(0, completion_time - job_info[job]['due date']))
    
    return sum(tardilist)



def check_precedence(x: List[int], prec: List[Tuple[int, int]], lmbda: int, job_info: Dict[int, Dict[str, int]], method: str = 'rejection'
                     ) -> Tuple[bool, float]:
    '''
        Check if precedence of a schedule violates the rules.

        Inputs:
            x: current job schedule
            prec: a list of all precedences (u, v) where u is the predecessor of v
            lmbda: penalty weight parameter
            job_info: containing processing times and due dates per job.
            method: choose either 'rejection' for rejection method or 'penalty' for penalty function method.

        Returns:
            (valid_precedence, total_penalty_cost)
        
        Constraint functions (one per precedence k = (u, v)) - by how many positions job u is placed after job v:
            h_k(S) = pos_S(u) - pos_S(v)  <= 0
        (positions are distinct, so h_k <= 0 exactly when u comes before v)
    '''
    position = {job: i for i, job in enumerate(x)}
    penalty = 0
    is_valid = True

    for u, v in prec:
        if u in position and v in position:
            if position[u] > position[v]:
                is_valid = False
                penalty += lmbda * max(0, (position[u] - position[v])) # h_k measures the sum of magnitude of violations in the current schedule

    if method == 'rejection':
        return (is_valid, 0.0)

    elif method == 'penalty':
        cost = tardiness(x, job_info) + penalty
        return (is_valid, cost)

    else:
        raise ValueError("Invalid method specified. Options: 'rejection', 'penalty'.")



def tabu(p: List[int], 
        d: List[int], 
        prec: List[Tuple[int, int]], 
        K: int, 
        gamma: float, 
        lmbda: float, 
        L: int, 
        x0: List[int] = None, 
        method = 'rejection', 
        w: float = None, 
        verbose: bool = False
    ) -> Tuple[List[int], float, List[Dict]]:
    '''
        Compute the best schedule based on tabu search.

        Inputs:
            p: processing time
            d: due dates
            prec: a list of all precedences (u, v) where u is the predecessor of v
            K: maximum iterations
            gamma: threshold
            L: tabu list length
            handling: 0 for rejection method, 1 for penalty function method. Any other int is not valid.
            x0: initial schedule (if given)
            method: choose either 'rejection' for rejection method or 'penalty' for penalty function method.
            w: optional weights (None = total tardiness)
            verbose: print every candidate and every accepted solution
            
        Returns:
            (x_best, g_best, history): best schedule satisfying all precedences, its total tardiness, and one record per iteration (for the 3.3 table and plot)
    '''

    def cost_of(x): # check precedence validity and calculate corresponding cost
        valid, pen_cost = check_precedence(x, prec, lmbda, job_info, method)
        return valid, (pen_cost if method == 'penalty' else tardiness(x, job_info))

    num_jobs = len(p)
    suc, pred, job_info = preprocessing(num_jobs, prec, p, d, w)
    if not x0:
        x0 = initialise_schedule(num_jobs, pred, suc)

    # Note: the rejection method needs a feasible starting point 
    if method == 'rejection' and not cost_of(x0)[0]:
        raise ValueError("Rejection method needs an initial schedule satisfying all precedences")
        
    tabu_list = []
    t_best = math.inf # g_best from pseudocode
    x_curr = x0 # current schedule (x_k in the pseudocode)
    start = 0 # start index for cyclic enumeration of neighbours 
    x_best_feasible, g_best_feasible = None, math.inf # track best schedule satisfying all precedences
    history = []

    for k in range(K+1):
        if k == 0: # iteration 0
            valid0, c_curr = cost_of(x0)
            t_best = c_curr # g_best of the pseudocode (aspiration test)

            if valid0:
                x_best_feasible, g_best_feasible = x0, tardiness(x0, job_info)
            history.append(dict(k=0, x=list(x0), cost=c_curr, valid=valid0, g_best=g_best_feasible))
            if verbose:
                print(f"k=0  x={x0}  cost={c_curr}  valid={valid0}  g_best={g_best_feasible}")

        else: # visit neighbours cyclically from `start` (not always pos 0) - stop at the first admissible neighbour (first-accept)
            accepted = None
            for step in range(num_jobs - 1):
                i = (start + step) % (num_jobs - 1)

                # Swapping
                x_tmp = x_curr.copy()
                x_tmp[i], x_tmp[i+1] = x_tmp[i+1], x_tmp[i]
                pair = (min(x_tmp[i], x_tmp[i+1]), max(x_tmp[i], x_tmp[i+1]))
                is_tabu = pair in tabu_list # check tabu list

                # Check if schedule satisfy precedence
                valid_schedule, c = cost_of(x_tmp)
                if method == 'rejection' and not valid_schedule:
                    if verbose:
                        print(f"    cand pos {i+1}-{i+2} swap {pair}: violates precedence -> skipped")
                    continue

                # (Delta > -gamma and (i,j) not tabu)  or  g(y) < g_best with Delta = g(x_k) - g(y)
                delta = c_curr - c
                accept = (delta > -gamma and not is_tabu) or c < t_best
                if verbose:
                    print(f"    cand pos {i+1}-{i+2} swap {pair}: {x_tmp} cost={c} delta={delta} "
                            f"tabu={is_tabu} -> {'ACCEPT' if accept else 'reject'}")
                if accept:
                    accepted = (x_tmp, c, valid_schedule, pair, i)
                    break # good pair

            if accepted is None:
                if verbose:
                    print(f"k={k}: no admissible neighbour -> terminate")
                break

            x_curr, c_curr, valid_curr, pair, i = accepted
            t_best = min(t_best, c_curr)
            if valid_curr and tardiness(x_curr, job_info) < g_best_feasible:
                x_best_feasible, g_best_feasible = list(x_curr), tardiness(x_curr, job_info)

            # If pair already in list, remove first and place it at the back of the list again
            if pair in tabu_list:
                tabu_list.remove(pair) 
            tabu_list.append(pair)

            while len(tabu_list) > L: # Satisfy tabu list length requirement
                tabu_list.pop(0)
            start = (i + 1) % (num_jobs - 1) # cyclic enumeration 

            # Track historical swaps
            history.append(dict(k=k, x=list(x_curr), cost=c_curr, valid=valid_curr, g_best=g_best_feasible))
            if verbose:
                print(f"k={k}  x={x_curr}  cost={c_curr}  valid={valid_curr}  "
                        f"g_best={g_best_feasible}  T={tabu_list}")


    if x_best_feasible is None: # If no feasible schedule visited, repair best penalised
        best_pen = min(history, key=lambda h: h['cost'])['x']
        x_best_feasible = repair_schedule(best_pen, pred, suc)
        g_best_feasible = tardiness(x_best_feasible, job_info)
        if verbose:
            print(f"No feasible schedule visited; repaired best penalised schedule -> "
                    f"{x_best_feasible}, tardiness {g_best_feasible}")
            
    if verbose:
        print(f"RESULT: best feasible schedule {x_best_feasible}, total tardiness {g_best_feasible}")
    
    return x_best_feasible, g_best_feasible, history
    
    