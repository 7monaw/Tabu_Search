"""
sanity TESTS using lecture and tutorial 3 examples
run: python test_tabu.py
"""

from tabu_search import tabu, preprocessing, initialise_schedule, check_precedence, tardiness, p, d, prec, x0

def run(name, p_, d_, w, x0_, gamma, L, K, expected_path, expected_best):
    print(f"\n##### {name}")
    xb, gb, hist = tabu(p_, d_, [], K, gamma, 0, L, x0=x0_, w=w, verbose=True)
    assert [h['x'] for h in hist] == expected_path, f"{name}: wrong path"
    assert gb == expected_best, f"{name}: best {gb} != {expected_best}"
    print(f"PASS {name}")

run("Lecture example", [10, 10, 13, 4], [4, 2, 1, 12], [13, 12, 1, 12], [1, 3, 4, 2], 100, 3, 50,
    [[1, 3, 4, 2], [1, 4, 3, 2], [1, 4, 2, 3], [4, 1, 2, 3], [4, 2, 1, 3], [2, 4, 1, 3]], 402)
run("Tutorial 3 Problem 9", [10, 10, 13, 4], [4, 2, 1, 12], [14, 12, 1, 12], [2, 1, 4, 3], 100, 2, 3,
    [[2, 1, 4, 3], [1, 2, 4, 3], [1, 4, 2, 3], [4, 1, 2, 3]], 408)
run("Tutorial 3 Problem 10a", [16, 11, 4, 8], [1, 2, 7, 9], [3, 4, 5, 7], [4, 2, 1, 3], 20, 2, 4,
    [[4, 2, 1, 3], [4, 2, 3, 1], [4, 3, 2, 1], [3, 4, 2, 1]], 219)
run("Tutorial 3 Problem 11", [3, 13, 10, 12], [13, 2, 8, 6], [7, 1, 5, 9], [4, 1, 2, 3], 20, 2, 4,
    [[4, 1, 2, 3], [1, 4, 2, 3], [1, 4, 3, 2], [4, 1, 3, 2]], 189)

suc, pred, job_info = preprocessing(31, prec, p, d)
assert check_precedence(x0, prec, 0, job_info)[0], "given x0 must satisfy all precedences"
assert check_precedence(initialise_schedule(31, pred, suc), prec, 0, job_info)[0]
print("\nx0 total tardiness:", tardiness(x0, job_info))
print("ALL TESTS PASSED")
