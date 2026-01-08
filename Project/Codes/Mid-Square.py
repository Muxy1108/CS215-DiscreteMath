# middle_square_full.py
# Full analysis for middle-square method using all n-digit seeds (default n=6).
#
# Requirements:
#   pip install matplotlib
#
# Notes:
# - For n=6, state space size is 1,000,000, which is feasible for full enumeration.
# - This script computes exact period / preperiod for every state via functional-graph analysis.

from collections import deque, Counter
from array import array
import matplotlib.pyplot as plt


# ============================================================
# 1) Middle-square transition
# ============================================================
def f_middle_square(x, n_digits):
    """
    Middle-square transition using integer arithmetic (supports leading zeros semantics).
    Assumes n_digits is even.

    Let n = n_digits, half = n/2.
    next = floor(x^2 / 10^half) mod 10^n
    """
    half = n_digits // 2
    base = 10 ** n_digits
    cut = 10 ** half
    return (x * x // cut) % base


# ============================================================
# 2) Exact period/preperiod for all seeds via functional graph
# ============================================================
def analyze_all_seeds(n_digits=6):
    """
    Build functional graph on N=10^n states (each has out-degree 1),
    compute for every node:
      - period[node]   : length of the cycle it eventually enters
      - dist[node]     : preperiod (distance to the cycle; 0 if on cycle)

    Also returns:
      - E_period, E_preperiod (exact averages over all states)
      - cycle_len_counter: {cycle_length -> number_of_cycles}
    """
    assert n_digits % 2 == 0, "This implementation assumes n_digits is even (e.g., 4, 6, 8)."

    N = 10 ** n_digits

    # nxt[x] = f(x)
    nxt = array("I", [0]) * N
    indeg = array("I", [0]) * N

    # reverse adjacency lists using head/enxt/frm arrays (memory-efficient)
    # head[v] = index of first predecessor edge (stored by predecessor node id)
    head = array("i", [-1]) * N
    frm = array("I", [0]) * N
    enxt = array("i", [0]) * N

    # Build the graph: for each x, one edge x -> y
    for x in range(N):
        y = f_middle_square(x, n_digits)
        nxt[x] = y
        indeg[y] += 1

        # store predecessor x in v=y's list
        frm[x] = x
        enxt[x] = head[y]
        head[y] = x

    # ------------------------------------------------------------
    # Step A: Kahn-like pruning removes all nodes not on cycles
    # Nodes left (removed==0) are exactly cycle nodes.
    # ------------------------------------------------------------
    removed = bytearray(N)  # 1 => removed (not in cycle)
    q = deque(i for i in range(N) if indeg[i] == 0)

    while q:
        u = q.popleft()
        removed[u] = 1
        v = nxt[u]
        indeg[v] -= 1
        if indeg[v] == 0:
            q.append(v)

    # ------------------------------------------------------------
    # Step B: identify each cycle, set period and dist for cycle nodes
    # ------------------------------------------------------------
    in_cycle = bytearray(N)
    visited_cycle = bytearray(N)

    dist = array("i", [-1]) * N     # dist to cycle (preperiod); cycle nodes have dist=0
    period = array("I", [0]) * N    # cycle length for each node

    cycle_len_counter = Counter()

    for i in range(N):
        if removed[i] == 0 and visited_cycle[i] == 0:
            # walk until we revisit within the cycle-subgraph
            cur = i
            while visited_cycle[cur] == 0:
                visited_cycle[cur] = 1
                cur = nxt[cur]

            # cur is now inside a cycle; extract that cycle
            cycle_nodes = []
            start = cur
            while True:
                cycle_nodes.append(cur)
                cur = nxt[cur]
                if cur == start:
                    break

            L = len(cycle_nodes)
            cycle_len_counter[L] += 1

            for v in cycle_nodes:
                in_cycle[v] = 1
                dist[v] = 0
                period[v] = L

    # ------------------------------------------------------------
    # Step C: BFS from cycle nodes backwards along reverse edges
    # to assign dist/period for tree nodes (in-arborescences).
    # ------------------------------------------------------------
    bfs = deque([i for i in range(N) if in_cycle[i] == 1])

    while bfs:
        v = bfs.popleft()
        e = head[v]
        while e != -1:
            u = frm[e]  # predecessor node
            if in_cycle[u] == 0 and dist[u] == -1:
                dist[u] = dist[v] + 1
                period[u] = period[v]
                bfs.append(u)
            e = enxt[e]

    # ------------------------------------------------------------
    # Step D: exact expectations
    # ------------------------------------------------------------
    sum_period = 0
    sum_preperiod = 0
    for i in range(N):
        sum_period += period[i]
        sum_preperiod += dist[i]

    E_period = sum_period / N
    E_preperiod = sum_preperiod / N

    return {
        "N": N,
        "E_period": E_period,
        "E_preperiod": E_preperiod,
        "period": period,
        "dist": dist,
        "cycle_len_counter": cycle_len_counter,
    }


# ============================================================
# 3) Basin size & LaTeX tables
# ============================================================
def basin_size_by_period(period_arr):
    """
    period_arr[i] = period length for state i.
    Returns dict: L -> number of states whose period is L.
    """
    basin = {}
    for L in period_arr:
        basin[L] = basin.get(L, 0) + 1
    return basin


def print_latex_cycle_count_table(cycle_len_counter, caption, label):
    """
    LaTeX table: Cycle length -> # cycles
    """
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(rf"\caption{{{caption}}}")
    print(rf"\label{{{label}}}")
    print(r"\begin{tabular}{rr}")
    print(r"\hline")
    print(r"Cycle length & \# cycles \\")
    print(r"\hline")
    for L in sorted(cycle_len_counter.keys()):
        print(f"{L} & {cycle_len_counter[L]} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")


def print_latex_cycle_basin_table(cycle_len_counter, basin_counts, N, caption, label):
    """
    LaTeX table:
      Cycle length | # cycles | Basin size | Basin (%)
    """
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(rf"\caption{{{caption}}}")
    print(rf"\label{{{label}}}")
    print(r"\begin{tabular}{rrrr}")
    print(r"\hline")
    print(r"Cycle length & \# cycles & Basin size & Basin (\%) \\")
    print(r"\hline")

    all_L = sorted(set(cycle_len_counter.keys()) | set(basin_counts.keys()))
    for L in all_L:
        num_cycles = cycle_len_counter.get(L, 0)
        basin = basin_counts.get(L, 0)
        pct = 100.0 * basin / N
        print(f"{L} & {num_cycles} & {basin} & {pct:.4f} \\\\")

    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")


# ============================================================
# 4) Plots: period / preperiod distributions
# ============================================================
def plot_period_counts_discrete(period_arr, n_digits, logy=True, max_xticks=20):
    """
    Period distribution as a discrete bar chart (best for sparse period values).
    """
    pc = Counter(period_arr)
    xs = sorted(pc.keys())
    ys = [pc[x] for x in xs]

    plt.figure()
    idx = list(range(len(xs)))
    plt.bar(idx, ys)

    step = max(1, len(xs) // max_xticks)
    tick_pos = idx[::step]
    tick_lab = [str(xs[i]) for i in range(0, len(xs), step)]
    plt.xticks(tick_pos, tick_lab)

    plt.xlabel("Period length")
    plt.ylabel("Count" + (" (log scale)" if logy else ""))
    plt.title(f"Period Length Counts (n_digits={n_digits})")
    if logy:
        plt.yscale("log")
    plt.tight_layout()
    plt.show()


def plot_preperiod_hist(dist_arr, n_digits, logy=False, bins="auto"):
    """
    Preperiod distribution (histogram).
    """
    plt.figure()
    plt.hist(dist_arr, bins=bins)
    plt.xlabel("Preperiod length")
    plt.ylabel("Count" + (" (log scale)" if logy else ""))
    plt.title(f"Preperiod Length Distribution (n_digits={n_digits})")
    if logy:
        plt.yscale("log")
    plt.tight_layout()
    plt.show()


# ============================================================
# 5) Digit frequency in one-step output y=f(x)
# ============================================================
def digit_frequency_one_step(n_digits):
    """
    Count digit frequencies over all x in [0,10^n-1] for one-step output y=f(x).
    Counts exactly n_digits digits (including leading zeros).
    Returns (counts[10], total_digits).
    """
    N = 10 ** n_digits
    counts = [0] * 10
    for x in range(N):
        y = f_middle_square(x, n_digits)
        for _ in range(n_digits):
            counts[y % 10] += 1
            y //= 10
    total_digits = N * n_digits
    return counts, total_digits


def plot_digit_frequency(counts, total_digits, n_digits):
    digits = list(range(10))
    expected = total_digits / 10.0

    plt.figure()
    plt.bar(digits, counts)
    plt.axhline(expected, linewidth=1)
    plt.xticks(digits)
    plt.xlabel("Digit (0-9)")
    plt.ylabel("Count")
    plt.title(f"Digit Frequency in one-step output y=f(x) (n_digits={n_digits})")
    plt.tight_layout()
    plt.show()

    plt.figure()
    freqs = [c / total_digits for c in counts]
    plt.bar(digits, freqs)
    plt.axhline(0.1, linewidth=1)
    plt.xticks(digits)
    plt.xlabel("Digit (0-9)")
    plt.ylabel("Frequency")
    plt.title(f"Digit Frequency (Proportions) in one-step output (n_digits={n_digits})")
    plt.tight_layout()
    plt.show()


# ============================================================
# 6) Scatter plot (x, f(x)) using deterministic sampling (no randomness)
# ============================================================
def plot_scatter_step(n_digits: int, step: int = 10):
    N = 10**n_digits
    xs, ys = [], []
    for x in range(0, N, step):
        xs.append(x)
        ys.append(f_middle_square(x, n_digits))
    plt.figure()
    plt.scatter(xs, ys, s=1, alpha=0.3)
    plt.xlabel("x"); plt.ylabel("f(x)")
    plt.title(f"Scatter of (x, f(x)) (step={step})")
    plt.tight_layout(); plt.show()



# ============================================================
# 7) Main
# ============================================================
def main():
    n_digits = 6

    # ---- Exact functional-graph analysis for all seeds ----
    result = analyze_all_seeds(n_digits=n_digits)

    print(f"State space N = {result['N']}")
    print(f"Exact E[period]    = {result['E_period']:.6f}")
    print(f"Exact E[preperiod] = {result['E_preperiod']:.6f}")

    # Cycle length -> number of cycles
    print("\nCycle length -> number of cycles:")
    for L in sorted(result["cycle_len_counter"].keys()):
        print(f"  {L}: {result['cycle_len_counter'][L]}")

    # Basin size table
    basin_counts = basin_size_by_period(result["period"])

    # Ring-node total (useful for report)
    ring_nodes = 0
    for L, c in result["cycle_len_counter"].items():
        ring_nodes += L * c
    print(f"\nTotal number of cycle (ring) states = {ring_nodes} (out of N={result['N']})")

    # ---- LaTeX tables ----
    print("\nLaTeX table: cycle length -> # cycles")
    print_latex_cycle_count_table(
        result["cycle_len_counter"],
        caption="Cycle length distribution (middle-square, $n=6$)",
        label="tab:cycle-length-cycles"
    )

    print("\nLaTeX table: cycle length -> # cycles -> basin size")
    print_latex_cycle_basin_table(
        cycle_len_counter=result["cycle_len_counter"],
        basin_counts=basin_counts,
        N=result["N"],
        caption="Cycle length, number of cycles, and basin size (middle-square, $n=6$)",
        label="tab:cycle-basin-n6"
    )

    # ---- Plots ----
    plot_period_counts_discrete(result["period"], n_digits=n_digits, logy=True)
    plot_preperiod_hist(result["dist"], n_digits=n_digits, logy=False, bins="auto")

    # one-step digit frequency y=f(x)
    counts, total_digits = digit_frequency_one_step(n_digits)
    plot_digit_frequency(counts, total_digits, n_digits=n_digits)

    # scatter (x, f(x)) deterministic sampling
    plot_scatter_step(n_digits=n_digits)


if __name__ == "__main__":
    main()
