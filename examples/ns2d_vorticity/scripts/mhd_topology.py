#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sub-grid periodic X/O tracker for 2-D magnetic flux a(x,y).

B = (∂y a, −∂x a), so magnetic nulls are ∇a = 0 and B · ∇a = 0: magnetic
field lines lie on a = const. This module locates nulls by Newton plus a
Fourier evaluation of ∇a and the Hessian, tracks them in time, and traces
the critical level a = a_X along B (magnetic separatrices).

Morse ±∇a walks are kept as a scalar-field diagnostic only. They are not
magnetic connectivity.

It is analysis-only. Agreement of Ez = −∂t a with η j at an X-point is a
resistive Ohm diagnostic, not by itself a reconnection claim.
"""

from __future__ import print_function

import math

import numpy as np

TWOPI = 2.0 * math.pi
KIND_X = "X"
KIND_OMAX = "O_max"
KIND_OMIN = "O_min"
KIND_DEGEN = "degenerate"


def wrap(x, length=TWOPI):
    return np.mod(x, length)


def periodic_delta(a, b, length=TWOPI):
    d = wrap(a - b, length)
    return np.where(d > 0.5 * length, d - length, d)


def periodic_dist(x0, y0, x1, y1, length=TWOPI):
    dx = float(periodic_delta(x0, x1, length))
    dy = float(periodic_delta(y0, y1, length))
    return math.hypot(dx, dy)


def fourier_point(a, x, y, length=TWOPI, hat=None, kx=None, ky=None):
    """Evaluate a, ∇a and Hessian at (x,y) from the DFT of a.

    numpy fft2 is unnormalized; the inverse factor is 1/N². Wave numbers
    use the same 2π/L convention as `spectral_derivs`. Pass cached `hat`,
    `kx`, `ky` when evaluating many points of one field.
    """
    n = a.shape[0]
    dx = length / float(n)
    if kx is None:
        kx = 2.0 * math.pi * np.fft.fftfreq(n, d=dx)
    if ky is None:
        ky = kx
    if hat is None:
        hat = np.fft.fft2(a)
    # Samples live at (i+1/2) dx; DFT index 0 is that first sample.
    x0 = x - 0.5 * dx
    y0 = y - 0.5 * dx
    px = np.exp(1j * kx * x0)[:, None]
    py = np.exp(1j * ky * y0)[None, :]
    phase = px * py
    scale = 1.0 / float(n * n)
    val = np.sum(hat * phase) * scale
    ax = np.sum(1j * kx[:, None] * hat * phase) * scale
    ay = np.sum(1j * ky[None, :] * hat * phase) * scale
    axx = np.sum(-(kx[:, None] ** 2) * hat * phase) * scale
    ayy = np.sum(-(ky[None, :] ** 2) * hat * phase) * scale
    axy = np.sum(-(kx[:, None] * ky[None, :]) * hat * phase) * scale
    return (float(val.real), float(ax.real), float(ay.real),
            float(axx.real), float(ayy.real), float(axy.real))


def bilinear(field, x, y, dx):
    """Interpolate a cell-centered field; nodes at (i+1/2) dx."""
    n = field.shape[0]
    s = x / dx - 0.5
    t = y / dx - 0.5
    i0 = int(math.floor(s)) % n
    j0 = int(math.floor(t)) % n
    fx = s - math.floor(s)
    fy = t - math.floor(t)
    i1 = (i0 + 1) % n
    j1 = (j0 + 1) % n
    v00 = field[i0, j0]
    v10 = field[i1, j0]
    v01 = field[i0, j1]
    v11 = field[i1, j1]
    return (1.0 - fx) * (1.0 - fy) * v00 + fx * (1.0 - fy) * v10 + (
        1.0 - fx
    ) * fy * v01 + fx * fy * v11


def spectral_derivs(a, length=TWOPI):
    """Periodic spectral ∂x, ∂y, Hessian of a cell-centered field."""
    n = a.shape[0]
    kx = 2.0 * math.pi * np.fft.fftfreq(n, d=length / n)
    ky = kx.copy()
    hat = np.fft.fft2(a)
    ax = np.fft.ifft2(1j * kx[:, None] * hat).real
    ay = np.fft.ifft2(1j * ky[None, :] * hat).real
    axx = np.fft.ifft2(-(kx[:, None] ** 2) * hat).real
    ayy = np.fft.ifft2(-(ky[None, :] ** 2) * hat).real
    axy = np.fft.ifft2(-kx[:, None] * ky[None, :] * hat).real
    return ax, ay, axx, ayy, axy


def _classify(axx, ayy, axy, degen_ratio=0.05):
    det = axx * ayy - axy * axy
    tr = axx + ayy
    disc = tr * tr - 4.0 * det
    if disc < 0.0:
        disc = 0.0
    lam1 = 0.5 * (tr + math.sqrt(disc))
    lam2 = 0.5 * (tr - math.sqrt(disc))
    mag = max(abs(lam1), abs(lam2), 1.0e-30)
    ratio = min(abs(lam1), abs(lam2)) / mag
    cond = mag / max(min(abs(lam1), abs(lam2)), 1.0e-30)
    degenerate = ratio < degen_ratio
    if det < 0.0:
        kind = KIND_DEGEN if degenerate else KIND_X
    elif axx < 0.0 or tr < 0.0:
        kind = KIND_DEGEN if degenerate else KIND_OMAX
    else:
        kind = KIND_DEGEN if degenerate else KIND_OMIN
    return kind, det, ratio, cond, lam1, lam2


def _eigvecs(axx, ayy, axy, lam1, lam2):
    def vec(lam):
        # (H - lam I) v = 0
        if abs(axy) > 1.0e-14:
            vx, vy = axy, lam - axx
        elif abs(axx - lam) <= abs(ayy - lam):
            vx, vy = 1.0, 0.0
        else:
            vx, vy = 0.0, 1.0
        nrm = math.hypot(vx, vy)
        if nrm < 1.0e-30:
            return 1.0, 0.0
        return vx / nrm, vy / nrm

    return vec(lam1), vec(lam2)


def locate_critical_points(a, j=None, length=TWOPI, max_newton=12, degen_ratio=0.05):
    """Return sub-grid critical points of a on the periodic square."""
    n = a.shape[0]
    dx = length / n
    ax, ay, axx, ayy, axy = spectral_derivs(a, length)
    kx = 2.0 * math.pi * np.fft.fftfreq(n, d=dx)
    hat = np.fft.fft2(a)
    g2 = ax * ax + ay * ay
    local_min = np.ones((n, n), dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            local_min &= g2 <= np.roll(np.roll(g2, di, 0), dj, 1)
    # Seed Newton from every local |∇a|² minimum; duplicates collapse later.
    seeds = np.argwhere(local_min)
    pts = []
    for i0, j0 in seeds:
        x = (i0 + 0.5) * dx
        y = (j0 + 0.5) * dx
        ok = False
        for it in range(max_newton):
            if it < max_newton - 3:
                gx = bilinear(ax, x, y, dx)
                gy = bilinear(ay, x, y, dx)
                hxx = bilinear(axx, x, y, dx)
                hyy = bilinear(ayy, x, y, dx)
                hxy = bilinear(axy, x, y, dx)
            else:
                _, gx, gy, hxx, hyy, hxy = fourier_point(
                    a, x, y, length, hat=hat, kx=kx, ky=kx)
            det = hxx * hyy - hxy * hxy
            if abs(det) < 1.0e-18:
                break
            dxn = (-gx * hyy + gy * hxy) / det
            dyn = (-gy * hxx + gx * hxy) / det
            x = wrap(x + dxn, length)
            y = wrap(y + dyn, length)
            if dxn * dxn + dyn * dyn < 1.0e-16 * dx * dx:
                ok = True
                break
        aval, gx, gy, hxx, hyy, hxy = fourier_point(
            a, x, y, length, hat=hat, kx=kx, ky=kx)
        if (not ok) and (gx * gx + gy * gy > 1.0e-6):
            continue
        kind, det, ratio, cond, lam1, lam2 = _classify(hxx, hyy, hxy, degen_ratio)
        rec = {
            "kind": kind,
            "x": float(x),
            "y": float(y),
            "a": aval,
            "grad2": float(gx * gx + gy * gy),
            "hess_det": float(det),
            "eig_ratio": float(ratio),
            "hess_cond": float(cond),
            "lam1": float(lam1),
            "lam2": float(lam2),
            # Spectral j = -∇²a at the refined point, not bilinear of a j-grid.
            "j": float(-(hxx + hyy)),
        }
        rec["v1"], rec["v2"] = _eigvecs(hxx, hyy, hxy, lam1, lam2)
        pts.append(rec)
    # Collapse Newton duplicates (periodic).
    pts.sort(key=lambda p: p["grad2"])
    uniq = []
    for p in pts:
        if any(periodic_dist(p["x"], p["y"], q["x"], q["y"], length) < 0.35 * dx
               for q in uniq):
            continue
        uniq.append(p)
    uniq.sort(key=lambda p: (p["kind"], p["a"], p["x"], p["y"]))
    return uniq


def _walk_to_extremum(ax, ay, start, vec, lam, extrema, dx, length=TWOPI,
                      nstep=160):
    """Integrate gradient flow from an offset along Hessian eigenvector `vec`.

    Does not start at the X-point (where ∇a=0). Offset is 1.5 grid cells,
    capped at 0.2 so it shrinks under refinement. No nearest-O fallback.
    λ>0: follow +∇a toward a max; λ<0: follow −∇a toward a min.
    """
    offset = min(1.5 * dx, 0.2)
    x = wrap(start[0] + offset * vec[0], length)
    y = wrap(start[1] + offset * vec[1], length)
    follow = 1.0 if lam >= 0.0 else -1.0
    for _ in range(nstep):
        gx = bilinear(ax, x, y, dx)
        gy = bilinear(ay, x, y, dx)
        gx *= follow
        gy *= follow
        gn = math.hypot(gx, gy)
        if gn < 1.0e-14:
            break
        x = wrap(x + 0.5 * dx * gx / gn, length)
        y = wrap(y + 0.5 * dx * gy / gn, length)
        for k, o in enumerate(extrema):
            if periodic_dist(x, y, o["x"], o["y"], length) < 1.8 * dx:
                return k
    return None


def pair_morse(points, a, length=TWOPI):
    """Scalar Morse pairing (gradient flow of a), not magnetic connectivity."""
    xs = [p for p in points if p["kind"] == KIND_X]
    os_ = [p for p in points if p["kind"] in (KIND_OMAX, KIND_OMIN)]
    n = a.shape[0]
    dx = length / float(n)
    ax, ay, _, _, _ = spectral_derivs(a, length)
    pairs = []
    for xi, xpt in enumerate(xs):
        branches = []
        rays = (
            ("+v1", xpt["v1"], +1.0, xpt["lam1"]),
            ("-v1", xpt["v1"], -1.0, xpt["lam1"]),
            ("+v2", xpt["v2"], +1.0, xpt["lam2"]),
            ("-v2", xpt["v2"], -1.0, xpt["lam2"]),
        )
        for name, vec, sgn, lam in rays:
            direction = (sgn * vec[0], sgn * vec[1])
            k = _walk_to_extremum(ax, ay, (xpt["x"], xpt["y"]), direction, lam,
                                  os_, dx, length)
            rec = {"branch": name, "o_index": k}
            if k is not None:
                rec["o_kind"] = os_[k]["kind"]
                rec["o_x"] = os_[k]["x"]
                rec["o_y"] = os_[k]["y"]
                rec["o_a"] = os_[k]["a"]
            branches.append(rec)
        o_indices = sorted({b["o_index"] for b in branches if b["o_index"] is not None})
        pairs.append({
            "x_index": xi,
            "o_indices": o_indices,
            "n_links": len(o_indices),
            "branches": branches,
        })
    return pairs, xs, os_


def connectivity_graph(points, a, length=TWOPI):
    """Adjacency: each well-conditioned X and the O basins of its 4 rays."""
    pairs, xs, os_ = pair_morse(points, a, length)
    graph = []
    for pr, xpt in zip(pairs, xs):
        basins = []
        for b in pr["branches"]:
            if b["o_index"] is None:
                basins.append({"branch": b["branch"], "o": None})
            else:
                basins.append({
                    "branch": b["branch"],
                    "o": {
                        "kind": b["o_kind"],
                        "x": b["o_x"],
                        "y": b["o_y"],
                        "a": b["o_a"],
                    },
                })
        graph.append({
            "x": xpt["x"],
            "y": xpt["y"],
            "a": xpt["a"],
            "hess_cond": xpt["hess_cond"],
            "eig_ratio": xpt["eig_ratio"],
            "basins": basins,
        })
    return graph


def graph_key(graph, bin_size=0.3, length=TWOPI):
    """Resolution-tolerant fingerprint of the X–O adjacency graph."""
    def bpos(x, y):
        ix = int(round(wrap(x, length) / bin_size)) % int(round(length / bin_size))
        iy = int(round(wrap(y, length) / bin_size)) % int(round(length / bin_size))
        return (ix, iy)

    rows = []
    for node in graph:
        oset = []
        for b in node["basins"]:
            if b["o"] is None:
                oset.append((b["branch"], None))
            else:
                oset.append((b["o"]["kind"],) + bpos(b["o"]["x"], b["o"]["y"]))
        rows.append((bpos(node["x"], node["y"]), tuple(sorted(oset))))
    return tuple(sorted(rows))


def graphs_match(g1, g2, dist_tol=0.25, length=TWOPI):
    """Match X-points by position and compare connected O basins."""
    if len(g1) != len(g2):
        return False, "nX %d vs %d" % (len(g1), len(g2))
    used = set()
    for n1 in g1:
        best = None
        best_d = 1.0e9
        for i2, n2 in enumerate(g2):
            if i2 in used:
                continue
            d = periodic_dist(n1["x"], n1["y"], n2["x"], n2["y"], length)
            if d < best_d:
                best_d = d
                best = i2
        if best is None or best_d > dist_tol:
            return False, "no matching X for (%.3f,%.3f)" % (n1["x"], n1["y"])
        used.add(best)
        n2 = g2[best]
        o1 = sorted(
            (b["o"]["kind"], round(b["o"]["x"], 1), round(b["o"]["y"], 1))
            for b in n1["basins"] if b["o"] is not None)
        o2 = sorted(
            (b["o"]["kind"], round(b["o"]["x"], 1), round(b["o"]["y"], 1))
            for b in n2["basins"] if b["o"] is not None)
        if o1 != o2:
            return False, "basins differ at X (%.3f,%.3f): %s vs %s" % (
                n1["x"], n1["y"], o1, o2)
    return True, "ok"


def levelset_rays(xpt):
    """Four unit directions of the critical level H(v,v)=0 at a saddle.

    In Hessian eigen-coordinates a-a_X ≈ ½(λ1 ξ1² + λ2 ξ2²). The magnetic
    separatrices satisfy λ1 ξ1² + λ2 ξ2² = 0, so they are not the
    eigenvectors (those are the Morse ±∇a directions).
    """
    l1, l2 = xpt["lam1"], xpt["lam2"]
    if l1 * l2 >= 0.0:
        return []
    v1, v2 = xpt["v1"], xpt["v2"]
    r = math.sqrt(-l1 / l2)
    rays = []
    for s in (+1.0, -1.0):
        vx = v1[0] + s * r * v2[0]
        vy = v1[1] + s * r * v2[1]
        nrm = math.hypot(vx, vy)
        if nrm < 1.0e-18:
            continue
        rays.append((vx / nrm, vy / nrm))
        rays.append((-vx / nrm, -vy / nrm))
    return rays


def _winding(path, ox, oy, length=TWOPI):
    """Winding of a possibly wrapped polyline around (ox,oy)."""
    if len(path) < 3:
        return 0.0
    ux = [float(path[0][0])]
    uy = [float(path[0][1])]
    for i in range(1, len(path)):
        ux.append(ux[-1] + float(periodic_delta(path[i][0], path[i - 1][0], length)))
        uy.append(uy[-1] + float(periodic_delta(path[i][1], path[i - 1][1], length)))
    ux.append(ux[-1] + float(periodic_delta(path[0][0], path[-1][0], length)))
    uy.append(uy[-1] + float(periodic_delta(path[0][1], path[-1][1], length)))
    # Shift O by the lattice vector nearest the path centroid.
    cx = 0.5 * (min(ux) + max(ux))
    cy = 0.5 * (min(uy) + max(uy))
    oxu = ox + round((cx - ox) / length) * length
    oyu = oy + round((cy - oy) / length) * length
    ang = 0.0
    for i in range(len(ux) - 1):
        x0, y0 = ux[i] - oxu, uy[i] - oyu
        x1, y1 = ux[i + 1] - oxu, uy[i + 1] - oyu
        ang += math.atan2(x0 * y1 - y0 * x1, x0 * x1 + y0 * y1)
    return ang / (2.0 * math.pi)


def trace_magnetic_branch(ax, ay, a_grid, xpt, ray, xs, dx, length=TWOPI):
    """Trace B=(a_y,-a_x) on a=a_X from an offset along a level-set ray."""
    offset = min(1.5 * dx, 0.2)
    a_crit = xpt["a"]
    ux = xpt["x"] + offset * ray[0]
    uy = xpt["y"] + offset * ray[1]
    x, y = wrap(ux, length), wrap(uy, length)
    path = [(x, y)]
    nstep = int(10.0 * length / dx)
    orient = None
    for i in range(nstep):
        gx = bilinear(ax, x, y, dx)
        gy = bilinear(ay, x, y, dx)
        aval = bilinear(a_grid, x, y, dx)
        gn2 = gx * gx + gy * gy
        if gn2 > 1.0e-20:
            corr = (aval - a_crit) / gn2
            ux -= corr * gx
            uy -= corr * gy
            x, y = wrap(ux, length), wrap(uy, length)
            gx = bilinear(ax, x, y, dx)
            gy = bilinear(ay, x, y, dx)
        bx, by = gy, -gx
        bn = math.hypot(bx, by)
        if bn < 1.0e-14:
            break
        if orient is None:
            orient = 1.0 if (bx * ray[0] + by * ray[1]) >= 0.0 else -1.0
        bx *= orient
        by *= orient
        step = 0.4 * dx
        ux += step * bx / bn
        uy += step * by / bn
        x, y = wrap(ux, length), wrap(uy, length)
        path.append((x, y))
        if i < 5:
            continue
        for j, xp in enumerate(xs):
            if periodic_dist(x, y, xp["x"], xp["y"], length) < 2.2 * dx:
                return j, path
    return None, path


def _loop_from_two_edges(e1, e2):
    """Concatenate two directed polylines that share the same endpoints."""
    s1, d1, p1 = e1["src"], e1["dst"], e1["path"]
    s2, d2, p2 = e2["src"], e2["dst"], e2["path"]
    if d1 == s2 and s1 == d2:
        return p1 + p2
    if d1 == d2 and s1 == s2:
        return p1 + list(reversed(p2))
    if d1 == s2:
        return p1 + p2
    if s1 == s2:
        return list(reversed(p1)) + p2
    if d1 == d2:
        return p1 + list(reversed(p2))
    return p1 + list(reversed(p2))


def _assign_enclosed(graph, xs_idx, loop, os_, length):
    if loop is None or len(loop) < 8:
        return
    for oi, opt in enumerate(os_):
        w = _winding(loop, opt["x"], opt["y"], length)
        if abs(w) < 0.4:
            continue
        for xi in xs_idx:
            already = any(e["o_index"] == oi for e in graph[xi]["enclosed_O"])
            if already:
                continue
            graph[xi]["enclosed_O"].append({
                "o_index": oi,
                "kind": opt["kind"],
                "x": opt["x"],
                "y": opt["y"],
                "a": opt["a"],
                "winding": float(w),
                "delta_a": opt["a"] - graph[xi]["a"],
            })


def magnetic_connectivity(points, a, length=TWOPI):
    """Embedded magnetic-separatrix *multigraph* and O enclosure.

    Every traced a=a_X branch is a distinct edge, including parallel
    X–X branches and self-loops. Faces include 2-edge cycles. An O is
    paired to an X only if such a face winds around that O.
    """
    xs = [p for p in points if p["kind"] == KIND_X]
    os_ = [p for p in points if p["kind"] in (KIND_OMAX, KIND_OMIN)]
    n = a.shape[0]
    dx = length / float(n)
    ax, ay, _, _, _ = spectral_derivs(a, length)
    graph = []
    edges = []
    for xi, xpt in enumerate(xs):
        rays = levelset_rays(xpt)
        branches = []
        for ri, ray in enumerate(rays):
            hit, path = trace_magnetic_branch(
                ax, ay, a, xpt, ray, xs, dx, length)
            rec = {
                "ray": ri,
                "hit_x": hit,
                "n_path": len(path),
                "edge_id": None,
            }
            if hit is not None:
                rec["hit_x_pos"] = (xs[hit]["x"], xs[hit]["y"])
                rec["hit_x_a"] = xs[hit]["a"]
                eid = len(edges)
                rec["edge_id"] = eid
                edges.append({
                    "id": eid,
                    "src": xi,
                    "dst": hit,
                    "path": path,
                })
            branches.append(rec)
        xx = [b["hit_x"] for b in branches if b["hit_x"] is not None]
        graph.append({
            "x_index": xi,
            "x": xpt["x"],
            "y": xpt["y"],
            "a": xpt["a"],
            "hess_cond": xpt["hess_cond"],
            "eig_ratio": xpt["eig_ratio"],
            "branches": branches,
            "connects_x": sorted(set(xx)),
            "n_edges": len(xx),
            "enclosed_O": [],
        })
    # Self-loops (one branch returns to the same X).
    for e in edges:
        if e["src"] == e["dst"]:
            _assign_enclosed(graph, [e["src"]], e["path"], os_, length)
    # Parallel-edge 2-cycles: two distinct branches with the same endpoints.
    by_pair = {}
    for e in edges:
        key = tuple(sorted((e["src"], e["dst"])))
        by_pair.setdefault(key, []).append(e)
    for key, elist in by_pair.items():
        for i in range(len(elist)):
            for j in range(i + 1, len(elist)):
                loop = _loop_from_two_edges(elist[i], elist[j])
                _assign_enclosed(graph, list(key), loop, os_, length)
    # Longer vertex-simple cycles (3–4), using one representative path per
    # directed pair only as a supplement to the multigraph faces.
    adj = [[] for _ in xs]
    one_path = {}
    for e in edges:
        if e["dst"] not in adj[e["src"]]:
            adj[e["src"]].append(e["dst"])
        if e["src"] not in adj[e["dst"]]:
            adj[e["dst"]].append(e["src"])
        one_path.setdefault((e["src"], e["dst"]), e["path"])
    cycles = []

    def dfs(start, node, trail, seen):
        if len(trail) > 4:
            return
        for nb in adj[node]:
            if nb == start and len(trail) >= 3:
                cycles.append(list(trail))
                continue
            if nb in seen:
                continue
            seen.add(nb)
            trail.append(nb)
            dfs(start, nb, trail, seen)
            trail.pop()
            seen.remove(nb)

    for i in range(len(xs)):
        dfs(i, i, [i], set([i]))
    seen_c = set()
    for cyc in cycles:
        key = tuple(sorted(cyc))
        if key in seen_c:
            continue
        seen_c.add(key)
        loop = []
        ok = True
        for u, v in zip(cyc, cyc[1:] + cyc[:1]):
            path = one_path.get((u, v))
            if path is None:
                path = one_path.get((v, u))
                if path is not None:
                    path = list(reversed(path))
            if path is None:
                ok = False
                break
            loop.extend(path)
        if ok:
            _assign_enclosed(graph, cyc, loop, os_, length)
    return graph, xs, os_


def magnetic_graphs_match(g1, g2, dist_tol=0.35, length=TWOPI):
    """Compare X–X magnetic connections after matching X by position."""
    if len(g1) != len(g2):
        return False, "nX %d vs %d" % (len(g1), len(g2))
    used = set()
    for n1 in g1:
        best, best_d = None, 1.0e9
        for i2, n2 in enumerate(g2):
            if i2 in used:
                continue
            d = periodic_dist(n1["x"], n1["y"], n2["x"], n2["y"], length)
            if d < best_d:
                best_d, best = d, i2
        if best is None or best_d > dist_tol:
            return False, "no matching X for (%.3f,%.3f)" % (n1["x"], n1["y"])
        used.add(best)
        n2 = g2[best]
        def hit_list(node):
            out = []
            for b in node["branches"]:
                if b.get("hit_x_pos") is None:
                    continue
                out.append(b["hit_x_pos"])
            return out
        h1, h2 = hit_list(n1), hit_list(n2)
        if len(h1) != len(h2):
            return False, "n hits %d vs %d at (%.3f,%.3f)" % (
                len(h1), len(h2), n1["x"], n1["y"])
        usedh = set()
        for p1 in h1:
            best, bd = None, 1.0e9
            for i2, p2 in enumerate(h2):
                if i2 in usedh:
                    continue
                d = periodic_dist(p1[0], p1[1], p2[0], p2[1], length)
                if d < bd:
                    bd, best = d, i2
            if best is None or bd > dist_tol:
                return False, "X–X hits differ at (%.3f,%.3f)" % (n1["x"], n1["y"])
            usedh.add(best)
    return True, "ok"


# Conservative Alfvén-scale bound on [0,2π]² with VA ~ O(1).
DEFAULT_CP_SPEED = 2.0


def track_points(frames, times=None, max_speed=DEFAULT_CP_SPEED, length=TWOPI):
    """Greedy periodic tracking with a physical displacement gate.

    If `times` is given, the match radius is `max_speed * Δt` (same physical
    radius at every N). If `times` is omitted, `max_speed` is a per-frame
    displacement. Default `max_speed=2` is a VA~O(1) bound, not 4*dx.
    """
    tracks = []
    events = []
    next_id = 0

    def new_track(p, t_index):
        nonlocal next_id
        tid = next_id
        next_id += 1
        tracks.append({
            "id": tid,
            "kind": p["kind"],
            "history": [(t_index, dict(p))],
            "alive": True,
        })
        events.append({"type": "birth", "id": tid, "t_index": t_index, "p": p})
        return tid

    if not frames:
        return tracks, events
    for p in frames[0]:
        new_track(p, 0)

    for t in range(1, len(frames)):
        prev = [(tr, tr["history"][-1][1]) for tr in tracks if tr["alive"]]
        curr = list(frames[t])
        used_prev = set()
        used_curr = set()
        if times is not None:
            dt = abs(float(times[t]) - float(times[t - 1]))
            max_disp = max_speed * dt
        else:
            max_disp = max_speed
        cands = []
        for ip, (tr, pp) in enumerate(prev):
            for ic, pc in enumerate(curr):
                d = periodic_dist(pp["x"], pp["y"], pc["x"], pc["y"], length)
                if max_disp is not None and d > max_disp:
                    continue
                kind_ok = (pp["kind"] == pc["kind"]
                           or KIND_DEGEN in (pp["kind"], pc["kind"]))
                if not kind_ok:
                    continue
                cands.append((d, ip, ic))
        cands.sort()
        for d, ip, ic in cands:
            if ip in used_prev or ic in used_curr:
                continue
            used_prev.add(ip)
            used_curr.add(ic)
            tr, _ = prev[ip]
            pc = curr[ic]
            tr["history"].append((t, dict(pc)))
            if pc["kind"] != KIND_DEGEN:
                tr["kind"] = pc["kind"]
        for ip, (tr, pp) in enumerate(prev):
            if ip in used_prev:
                continue
            tr["alive"] = False
            events.append({
                "type": "death",
                "id": tr["id"],
                "t_index": t,
                "p": pp,
            })
        for ic, pc in enumerate(curr):
            if ic in used_curr:
                continue
            new_track(pc, t)

        deaths = [e for e in events if e["type"] == "death" and e["t_index"] == t]
        births = [e for e in events if e["type"] == "birth" and e["t_index"] == t]
        degen_now = [p for p in curr if p["kind"] == KIND_DEGEN]
        thresh = length / 32.0
        for dth in deaths:
            for bth in births:
                if periodic_dist(dth["p"]["x"], dth["p"]["y"],
                                 bth["p"]["x"], bth["p"]["y"], length) < thresh:
                    events.append({
                        "type": "possible_replace",
                        "t_index": t,
                        "death_id": dth["id"],
                        "birth_id": bth["id"],
                    })
        if degen_now and (deaths or births):
            events.append({
                "type": "degenerate_hessian",
                "t_index": t,
                "n_degen": len(degen_now),
                "n_death": len(deaths),
                "n_birth": len(births),
            })
    return tracks, events


def sample_grid(func, n, length=TWOPI):
    dx = length / n
    i = (np.arange(n) + 0.5) * dx
    x = i[:, None]
    y = i[None, :]
    return func(x, y)


def da_dt_from_track(history, times):
    """Central difference of a along a track; one-sided at endpoints."""
    a = np.array([h[1]["a"] for h in history], dtype=float)
    t_idx = [h[0] for h in history]
    t = np.array([times[i] for i in t_idx], dtype=float)
    dadt = np.zeros_like(a)
    if len(a) == 1:
        return t, a, dadt
    dadt[0] = (a[1] - a[0]) / max(t[1] - t[0], 1.0e-30)
    dadt[-1] = (a[-1] - a[-2]) / max(t[-1] - t[-2], 1.0e-30)
    for k in range(1, len(a) - 1):
        dadt[k] = (a[k + 1] - a[k - 1]) / max(t[k + 1] - t[k - 1], 1.0e-30)
    return t, a, dadt
