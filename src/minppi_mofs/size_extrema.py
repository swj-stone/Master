"""Certify integer maxima of coalition-size score curves on their stored binary64 grid."""
import numpy as np
from numba import njit


@njit(cache=True)
def difference_tree(w):
    """Dyadic enclosures for each stored basis column's forward differences."""
    n, d = w.shape
    base = 1
    while base < n - 1:
        base *= 2
    low = np.zeros((2*base, d))
    high = np.zeros_like(low)
    absmax = np.zeros(d)
    for h in range(d):
        for s in range(n):
            absmax[h] = max(absmax[h], abs(w[s, h]))
        if n < 2:
            continue
        for k in range(base):
            s = min(k, n-2)
            v = w[s+1, h] - w[s, h]
            # Identical input doubles subtract exactly. Enclose every other subtraction outwards.
            if w[s+1, h] != w[s, h]:
                low[base+k, h] = np.nextafter(v, -np.inf)
                high[base+k, h] = np.nextafter(v, np.inf)
        for i in range(base-1, 0, -1):
            low[i, h] = min(low[i*2, h], low[i*2+1, h])
            high[i, h] = max(high[i*2, h], high[i*2+1, h])
    return low, high, absmax


@njit(cache=True, inline="always")
def _dot(c, w, s):
    result = 0.
    for h in range(len(c)):
        result += c[h]*w[s, h]
    return result


@njit(cache=True)
def maximum(c, w, low, high, absmax, last, veto, stack):
    """Return the scalar exhaustive maximum, or a witnessed score above veto."""
    n = len(w)
    arg = last
    worst = _dot(c, w, last)
    checks = 1
    for s in (n-1, 0):
        if worst > veto:
            return worst, arg, checks, 0
        if s != last and (n > 1 or s != 0):
            v = _dot(c, w, s)
            checks += 1
            if v > worst:
                worst, arg = v, s
    if worst > veto or n <= 2:
        return worst, arg, checks, 0
    magnitude = 0.
    for h in range(len(c)):
        magnitude += abs(c[h])*absmax[h]
    # Bounds both endpoint dot errors and rounding in summing the derivative interval.
    # |D_h| <= 2 max|w_h|. Gradual underflow gets a separate absolute allowance.
    guard = (16*(len(c)+2))*(np.finfo(np.float64).eps*magnitude + 5e-324)
    count = 0
    length = 1
    base = len(low)//2
    stack[0, 0], stack[0, 1], stack[0, 2] = 1, 0, base
    while length:
        length -= 1
        i, a, width = stack[length, 0], stack[length, 1], stack[length, 2]
        b = min(n-1, a+width)
        # Ancestors have already evaluated both endpoints of every queued interval.
        if b-a <= 1:
            continue
        if b-a <= 6:
            for s in range(a+1, b):
                v = _dot(c, w, s)
                checks += 1
                if v > worst:
                    worst, arg = v, s
                if worst > veto:
                    return worst, arg, checks, count
            continue
        lower = upper = 0.
        flat = True
        for h in range(len(c)):
            if c[h] != 0.:
                left = low[i, h]
                right = high[i, h]
                flat = flat and left == 0. and right == 0.
                if c[h] > 0.:
                    lower += c[h]*left
                    upper += c[h]*right
                else:
                    lower += c[h]*right
                    upper += c[h]*left
        count += 1
        if flat or lower > guard or upper < -guard:
            continue
        half = width//2
        middle = a+half
        if middle < b:
            v = _dot(c, w, middle)
            checks += 1
            if v > worst:
                worst, arg = v, middle
            if worst > veto:
                return worst, arg, checks, count
            stack[length, 0], stack[length, 1], stack[length, 2] = i*2+1, middle, half
            length += 1
        stack[length, 0], stack[length, 1], stack[length, 2] = i*2, a, half
        length += 1
    return worst, arg, checks, count


@njit(cache=True)
def select(current, delta, keep, w, tol, last, low, high, absmax):
    """Match the exhaustive smallest feature within the specified absolute tolerance."""
    values = np.full(len(keep), np.inf)
    args = np.zeros(len(keep), np.int64)
    best = np.inf
    checks = intervals = 0
    c = np.empty_like(current)
    stack = np.empty((2+int(np.log2(len(low))), 3), np.int64)
    for j in range(len(keep)):
        if not keep[j]:
            continue
        for h in range(len(c)):
            c[h] = current[h]+delta[j, h]
        score, arg, nq, ni = maximum(c, w, low, high, absmax, last, best+tol, stack)
        checks += nq
        intervals += ni
        if score <= best+tol:
            values[j], args[j] = score, arg
            best = min(score, best)
    for j in range(len(keep)):
        if keep[j] and values[j] <= best+tol:
            return j, args[j], checks, intervals
    raise ArithmeticError("No finite worst-size vote")


@njit(cache=True)
def _disable(feature, alive, live, delta, linear, gp, gi, gv, ip, ii, jp, ji, unions):
    """Retire every live group hit by the deletion and each affected pair once."""
    for pos in range(jp[feature], jp[feature+1]):
        g = ji[pos]
        if not alive[g]:
            continue
        a,b = ip[g],ip[g+1]
        for k in range(a,b):
            col = ii[k]
            live[col] -= 1
            delta[col,b-a] -= linear[g]
        for edge in range(gp[g],gp[g+1]):
            h = gi[edge]
            if not alive[h]:
                continue
            degree = unions[edge]
            magnitude = gv[edge] if h==g else 2.*gv[edge]
            for k in range(a,b):
                delta[ii[k],degree] += magnitude
            if h!=g:
                k = a
                for m in range(ip[h],ip[h+1]):
                    f = ii[m]
                    while k<b and ii[k]<f:
                        k += 1
                    if k==b or ii[k]!=f:
                        delta[f,degree] += magnitude
        alive[g] = False


@njit(cache=True)
def rank_arrays(initial, linear, gp, gi, gv, ip, ii, jp, ji, unions, w, tol):
    """Trace one exact greedy path; compress identical dormant proposals."""
    low,high,absmax = difference_tree(w)
    p = len(jp)-1
    live = np.diff(jp)
    keep = np.ones(p,np.bool_)
    dormant = np.zeros(p,np.bool_)
    alive = np.ones(len(ip)-1,np.bool_)
    delta = initial.copy()
    current = np.zeros(initial.shape[1])
    order = np.empty(p,np.int64)
    t = queries = intervals = 0
    last = p-1
    while t<p:
        candidates = keep.copy()
        first = -1
        for j in range(p):
            if not keep[j] or live[j]:
                continue
            # State updates can leave tiny nonzero residues after the last live group dies.
            # A batched deletion must leave the stored binary64 current vector unchanged.
            unchanged = True
            for h in range(len(current)):
                if current[h]+delta[j,h]!=current[h]:
                    unchanged = False
                    break
            dormant[j] = unchanged
            if unchanged:
                if first<0:
                    first = j
                else:
                    candidates[j] = False
        j,last,nq,ni = select(current,delta,candidates,w,tol,last,low,high,absmax)
        queries += nq
        intervals += ni
        if dormant[j]:
            for col in range(j,p):
                if keep[col]:
                    if not dormant[col]:
                        break
                    order[t] = col
                    t += 1
                    keep[col] = False
        else:
            current += delta[j]
            _disable(j,alive,live,delta,linear,gp,gi,gv,ip,ii,jp,ji,unions)
            keep[j] = False
            order[t] = j
            t += 1
    return order,queries,intervals


def rank_minimum(game):
    """Return the complete minimum-vote deletion order and number of scalar size scores."""
    structure = game.structure
    inc,by,gram = structure.incidence,structure.by_feature,structure.gram
    p = inc.shape[1]
    w = np.ascontiguousarray(structure.weights[1:]*p/np.arange(1,p+1)[:,None])
    tol = 128*np.finfo(float).eps*(np.abs(game.linear).sum()+np.abs(gram.data).sum()+game.null)
    order,n,_ = rank_arrays(game.initial,game.linear,gram.indptr,gram.indices,gram.data,
                            inc.indptr,inc.indices,by.indptr,by.indices,structure.unions,w,tol)
    return order,n
