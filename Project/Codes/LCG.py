import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# ============================================================
# Configuration (match middle-square sample space: all 6-digit seeds)
# ============================================================
M = 10**6  # sample space size, seeds 0..999999 (6 digits)

# A full-period mixed LCG for m=10^6 (Hull–Dobell satisfied):
# gcd(c,m)=1, primes dividing m are 2 and 5, need (a-1) divisible by 2 and 5,
# and since 4|m, need (a-1) divisible by 4. So (a-1) divisible by lcm(10,4)=20.
A = 21
C = 1

# Plot sampling for scatter (million points is too heavy)
SCATTER_N = 200_000  # number of points to draw
SCATTER_STRIDE = max(1, M // SCATTER_N)  # deterministic sampling

# ============================================================
# Precompute f(x) for all x to speed up functional graph analysis
# ============================================================
def build_next_array_lcg(m: int, a: int, c: int) -> np.ndarray:
    x = np.arange(m, dtype=np.int64)
    nxt = (a * x + c) % m
    return nxt.astype(np.int32)

# ============================================================
# Functional graph analysis:
# For each node x, compute:
#   period[x]   = cycle length (lambda)
#   preperiod[x]= distance to cycle (mu)
# Also count number of cycles by their cycle length.
# ============================================================
def analyze_functional_graph(next_arr: np.ndarray):
    m = next_arr.shape[0]

    done = np.zeros(m, dtype=np.uint8)        # 0/1 whether node is fully processed
    period = np.zeros(m, dtype=np.int32)      # cycle length for each node
    preperiod = np.zeros(m, dtype=np.int32)   # distance to cycle for each node

    seen_id = np.full(m, -1, dtype=np.int32)  # which run_id last saw this node
    seen_pos = np.zeros(m, dtype=np.int32)    # position in current path (valid if seen_id==run_id)

    cycle_len_counts = Counter()
    run_id = 0

    for start in range(m):
        if done[start]:
            continue

        run_id += 1
        path = []
        x = start

        while True:
            if done[x]:
                # attach current path to an already processed node
                lam = int(period[x])
                mu = int(preperiod[x])
                # unwind path backwards
                for node in reversed(path):
                    mu += 1
                    period[node] = lam
                    preperiod[node] = mu
                    done[node] = 1
                break

            if seen_id[x] == run_id:
                # found a cycle within current path
                cycle_start = int(seen_pos[x])
                cycle_nodes = path[cycle_start:]
                lam = len(cycle_nodes)
                cycle_len_counts[lam] += 1

                # cycle nodes: preperiod=0
                for node in cycle_nodes:
                    period[node] = lam
                    preperiod[node] = 0
                    done[node] = 1

                # nodes before cycle: distance increasing backwards
                mu = 0
                for node in reversed(path[:cycle_start]):
                    mu += 1
                    period[node] = lam
                    preperiod[node] = mu
                    done[node] = 1
                break

            # first time see x in this run
            seen_id[x] = run_id
            seen_pos[x] = len(path)
            path.append(x)
            x = int(next_arr[x])

    return period, preperiod, cycle_len_counts

# ============================================================
# Digit frequency for 6-digit zero-padded outputs
#   We compute digits of y = f(x) for all x in 0..m-1
#   using vectorized arithmetic (no string formatting).
# ============================================================
def digit_frequency_six_digits(values: np.ndarray):
    # values: int array in [0, 10^6)
    v = values.astype(np.int64)
    # Extract 6 digits (units to hundred-thousands)
    counts_total = np.zeros(10, dtype=np.int64)
    counts_by_pos = np.zeros((6, 10), dtype=np.int64)  # pos 0=units, ..., pos 5=10^5

    pow10 = np.array([1, 10, 100, 1000, 10000, 100000], dtype=np.int64)
    for pos in range(6):
        digits = (v // pow10[pos]) % 10
        bc = np.bincount(digits, minlength=10)
        counts_by_pos[pos, :] = bc
        counts_total += bc

    return counts_total, counts_by_pos

# ============================================================
# Plot helpers
# ============================================================
def plot_histogram_from_counts(counts: np.ndarray, title: str, xlabel: str, ylabel: str, out_path: str):
    # counts[i] = number of nodes with value i
    xs = np.nonzero(counts)[0]
    ys = counts[xs]

    plt.figure()
    plt.plot(xs, ys)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_digit_frequency(counts_total: np.ndarray, out_path: str):
    digits = np.arange(10)
    plt.figure()
    plt.bar(digits, counts_total)
    plt.title("Digit frequency (6-digit, zero-padded)")
    plt.xlabel("Digit")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_scatter_x_fx(next_arr: np.ndarray, out_path: str, stride: int):
    xs = np.arange(0, next_arr.shape[0], stride, dtype=np.int32)
    ys = next_arr[xs]
    plt.figure()
    plt.scatter(xs, ys, s=1)
    plt.title("x - f(x) scatter (sampled)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# ============================================================
# Print LaTeX table for cycle length distribution
# ============================================================
def print_cycle_table_latex(cycle_len_counts: Counter, caption: str, label: str):
    items = sorted(cycle_len_counts.items())
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(rf"\caption{{{caption}}}")
    print(rf"\label{{{label}}}")
    print(r"\begin{tabular}{rr}")
    print(r"\hline")
    print(r"Cycle length & \# cycles \\")
    print(r"\hline")
    for length, cnt in items:
        print(f"{length} & {cnt} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")

# ============================================================
# Main
# ============================================================
def main():
    print(f"LCG parameters: m={M}, a={A}, c={C}")
    print("Building next array...")
    next_arr = build_next_array_lcg(M, A, C)

    print("Analyzing functional graph (period & preperiod for all seeds)...")
    period, preperiod, cycle_len_counts = analyze_functional_graph(next_arr)

    # Expectations
    E_period = float(period.mean())
    E_preperiod = float(preperiod.mean())
    print(f"E[period]    = {E_period:.4f}")
    print(f"E[preperiod] = {E_preperiod:.4f}")
    print(f"#cycles      = {sum(cycle_len_counts.values())}")

    # Histograms (counts per value)
    print("Computing histograms...")
    period_counts = np.bincount(period, minlength=int(period.max()) + 1)
    preperiod_counts = np.bincount(preperiod, minlength=int(preperiod.max()) + 1)

    # Plots
    print("Saving plots...")
    plot_histogram_from_counts(
        period_counts,
        title="LCG Period distribution",
        xlabel="period",
        ylabel="#seeds",
        out_path="LCG-Period.png",
    )
    plot_histogram_from_counts(
        preperiod_counts,
        title="LCG Preperiod distribution",
        xlabel="preperiod",
        ylabel="#seeds",
        out_path="LCG-Preperiod.png",
    )

    # Digit frequency over outputs f(x) (same sample: all seeds x, output y=f(x))
    print("Computing digit frequency for outputs f(x)...")
    counts_total, counts_by_pos = digit_frequency_six_digits(next_arr)
    plot_digit_frequency(counts_total, out_path="LCG-Digit-frequency.png")

    # Scatter (sampled)
    print("Saving scatter plot...")
    plot_scatter_x_fx(next_arr, out_path="LCG-Scatter.png", stride=SCATTER_STRIDE)

    # LaTeX table for cycle length distribution
    print("\nLaTeX table for cycle length distribution:")
    print_cycle_table_latex(
        cycle_len_counts,
        caption=f"Cycle length distribution (LCG, m={M}, a={A}, c={C})",
        label="tab:lcg-cycle-length-cycles",
    )

    # Optional: also print digit frequency totals
    print("\nDigit frequency totals (0..9) over 6 digits across all outputs f(x):")
    for d in range(10):
        print(f"{d}: {int(counts_total[d])}")

    print("\nDone. Generated: LCG-Period.png, LCG-Preperiod.png, LCG-Digit-frequency.png, LCG-Scatter.png")

if __name__ == "__main__":
    main()
