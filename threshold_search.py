def find_maximum(f, min_x, max_x, epsilon=1e-5):
    def binary_search(l, r, fl, fr, epsilon):
        mid = l + (r - l) / 2
        fm = f(mid)
        binary_search.eval_count += 1
        if (fm == fl and fm == fr) or r - l < epsilon:
            return mid, fm
        if fl > fm >= fr:
            return binary_search(l, mid, fl, fm, epsilon)
        if fl <= fm < fr:
            return binary_search(mid, r, fm, fr, epsilon)
        p1, f1 = binary_search(l, mid, fl, fm, epsilon)
        p2, f2 = binary_search(mid, r, fm, fr, epsilon)
        if f1 > f2:
            return p1, f1
        else:
            return p2, f2

    binary_search.eval_count = 0

    best_th, best_value = binary_search(min_x, max_x, f(min_x), f(max_x), epsilon)
    # print("Found maximum %f at x = %f in %d evaluations" % (best_value, best_th, binary_search.eval_count))
    return best_th, best_value