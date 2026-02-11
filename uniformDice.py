from fractions import Fraction

# ---------- Fraction Gaussian elimination ----------
def solve_linear_system_fraction(A, b):
    """
    Solve A x = b exactly over Fractions using Gauss-Jordan elimination.
    A: list[list[Fraction]] (n×n)
    b: list[Fraction]       (n)
    returns: list[Fraction] x
    """
    n = len(A)
    M = [A[i][:] + [b[i]] for i in range(n)]

    row = 0
    for col in range(n):
        # find pivot
        pivot = None
        for r in range(row, n):
            if M[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            continue
        # swap into place
        if pivot != row:
            M[row], M[pivot] = M[pivot], M[row]

        # normalize pivot row
        piv = M[row][col]
        for c in range(col, n + 1):
            M[row][c] /= piv

        # eliminate all other rows
        for r in range(n):
            if r == row:
                continue
            factor = M[r][col]
            if factor == 0:
                continue
            for c in range(col, n + 1):
                M[r][c] -= factor * M[row][c]

        row += 1
        if row == n:
            break

    # extract solution
    x = [Fraction(0) for _ in range(n)]
    for i in range(n):
        # find leading 1 if any
        lead = None
        for j in range(n):
            if M[i][j] == 1:
                lead = j
                break
            elif M[i][j] != 0:
                # inconsistent / singular
                raise ValueError("System is singular or inconsistent.")
        if lead is not None:
            x[lead] = M[i][n]
    return x

# ---------- KMP-style automaton for pattern matching ----------
def build_kmp_transition(pattern, alphabet):
    """
    pattern: list of symbols (e.g. [1,2,3,3,1])
    alphabet: iterable of possible symbols (e.g. range(1, N+1))
    Returns next_state[s][x] where s=matched prefix length in {0..m-1},
    x is a symbol, and next_state gives new matched length in {0..m}.
    Absorption when new matched length == m.
    """
    m = len(pattern)
    pat = pattern[:]

    def next_len(s, x):
        seq = pat[:s] + [x]
        tmax = min(m, len(seq))
        for t in range(tmax, -1, -1):
            if pat[:t] == seq[-t:]:
                return t
        return 0

    next_state = []
    for s in range(m):
        row = {}
        for x in alphabet:
            row[x] = next_len(s, x)
        next_state.append(row)
    return next_state

# ---------- Main: probability (T ≡ r mod p) and expected waiting time ----------
def pattern_stats_uniform(N, r, p, pattern):
    """
    Rolls are iid uniform on {1,2,...,N}.
    pattern: list of symbols in {1..N}

    Returns:
      (prob_mod, expected_time) as Fractions, where
        prob_mod = P(T ≡ r (mod p)) for the first occurrence time T of pattern,
        expected_time = E[T].
    """
    if p <= 0:
        raise ValueError("p must be positive.")
    r %= p
    if any((x < 1 or x > N) for x in pattern):
        raise ValueError("Pattern symbols must be in {1..N}.")

    alphabet = range(1, N + 1)
    m = len(pattern)
    prob = Fraction(1, N)

    # Build automaton transitions
    nxt = build_kmp_transition(pattern, alphabet)

    # ----- Part A: Probability T ≡ r (mod p) -----
    # Unknowns: u[s, t] = P(T ≡ t (mod p) | current matched length = s, time=0)
    # for s=0..m-1 and t=0..p-1. Total m*p variables.
    # Recurrence:
    # u[s,t] = sum_x P(x) * ( 1{T=1, so 1 mod p == t} if nxt[s][x]==m
    #                         else u[s', (t-1 mod p)] )
    # because if not absorbed, T = 1 + T', so T' mod p must be (t-1).
    idx = {}
    nvars = m * p
    for s in range(m):
        for t in range(p):
            idx[(s, t)] = s * p + t

    A = [[Fraction(0) for _ in range(nvars)] for _ in range(nvars)]
    b = [Fraction(0) for _ in range(nvars)]

    for s in range(m):
        for t in range(p):
            row = idx[(s, t)]
            A[row][row] = Fraction(1)
            const = Fraction(0)

            for x in alphabet:
                ns = nxt[s][x]
                if ns == m:
                    # absorbed in 1 step: residue is 1 mod p
                    if (1 % p) == t:
                        const += prob
                else:
                    col = idx[(ns, (t - 1) % p)]
                    A[row][col] -= prob

            b[row] = const

    sol = solve_linear_system_fraction(A, b)
    prob_mod = sol[idx[(0, r)]]

    # ----- Part B: Expected waiting time E[T] -----
    # Unknowns: e[s] = E[T | matched length = s]
    # Recurrence:
    # e[s] = 1 + sum_x P(x) * (0 if absorbed else e[ns])
    # => e[s] - sum_{ns<m} P(x) e[ns] = 1
    Ae = [[Fraction(0) for _ in range(m)] for _ in range(m)]
    be = [Fraction(1) for _ in range(m)]

    for s in range(m):
        Ae[s][s] = Fraction(1)
        for x in alphabet:
            ns = nxt[s][x]
            if ns < m:
                Ae[s][ns] -= prob
            # if ns==m, add nothing (absorbed: expected remaining = 0)

    e = solve_linear_system_fraction(Ae, be)
    expected_time = e[0]

    return prob_mod, expected_time


# ----------------- Example -----------------
if __name__ == "__main__":
    N = 6
    pattern = [1, 2, 3, 3, 1]
    r = 2
    p = 5

    prob_mod, expT = pattern_stats_uniform(N, r, p, pattern)
    print("P(T ≡ r mod p) =", prob_mod, "≈", float(prob_mod))
    print("E[T] =", expT, "≈", float(expT))
