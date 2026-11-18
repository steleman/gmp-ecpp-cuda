/*
 * fftpoly.cu -- polynomial arithmetic mod n over FftMod; see fftpoly.cuh.
 *
 * Bounds, on absolute values (residues are signed): FftMod's products
 * take inputs below 4n (with poly_len terms per coefficient) and return
 * values below n; kout()'s differences below 1.5n. Negations are 0 - x.
 * rem()'s A - F*Q, with |A| < 2n, is below 3n -- a legal product input,
 * which is all a remainder is used as -- so no bound grows down a
 * tree.
 *
 * Copyright (c) 2026 Stefan Teleman and contributors.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.  See <http://www.gnu.org/licenses/>.
 */

#include "fftpoly.cuh"

#include <set>

namespace gmpecmfft_detail {

slot_t PolyOps::take() {
  if (!free_.empty()) {
    slot_t s = free_.back();
    free_.pop_back();
    return s;
  }
  return M.slot();
}

void PolyOps::give(slot_t s) {
  if (s)
    free_.push_back(s);
}

Poly PolyOps::take(unsigned len) {
  Poly p(len);
  for (auto &s : p)
    s = take();
  return p;
}

void PolyOps::give(const Poly &p) {
  for (slot_t s : p)
    give(s);
}

std::vector<Poly> PolyOps::mul(const std::vector<MulReq> &r) {
  std::vector<Poly> out(r.size());
  std::vector<PolyJob> jobs(r.size());
  for (size_t i = 0; i < r.size(); i++) {
    out[i] = take(r[i].nd);
    jobs[i] = PolyJob{(unsigned)r[i].A->size(), r[i].A->data(),
                      (unsigned)r[i].B->size(), r[i].B->data(), r[i].lo,
                      r[i].nd, out[i].data()};
  }
  M.polymul(jobs);
  return out;
}

Poly PolyOps::mul(const Poly &A, const Poly &B, unsigned lo, unsigned nd) {
  return mul(std::vector<MulReq>{MulReq{&A, &B, lo, nd}})[0];
}

std::vector<Poly> PolyOps::inverse(const std::vector<const Poly *> &F,
                                   const std::vector<unsigned> &s) {
  const size_t n = F.size();
  std::vector<Poly> revF(n), I(n);
  std::vector<unsigned> t(n, 1);
  for (size_t i = 0; i < n; i++) {
    revF[i].assign(F[i]->rbegin(), F[i]->rend()); // revF[0] = leading 1
    I[i] = take(1);
    M.copy(I[i][0], ONE);
  }
  // Every inverse doubles its precision in step with the others, so each
  // step's products are one batch across all of them still short of s.
  for (;;) {
    std::vector<size_t> act;
    for (size_t i = 0; i < n; i++)
      if (t[i] < s[i])
        act.push_back(i);
    if (act.empty())
      break;
    const size_t na = act.size();
    std::vector<unsigned> t2(na), h(na);
    std::vector<Poly> rf(na);
    std::vector<MulReq> rq(na);
    for (size_t a = 0; a < na; a++) {
      const size_t i = act[a];
      t2[a] = 2 * t[i] < s[i] ? 2 * t[i] : s[i];
      h[a] = t2[a] - t[i];
      rf[a].assign(revF[i].begin(),
                   revF[i].begin() +
                       (t2[a] < revF[i].size() ? t2[a] : revF[i].size()));
    }
    // E = (rev(F) * I)[t, t2); 1 - rev(F) I = -X^t E (mod X^t2).
    for (size_t a = 0; a < na; a++)
      rq[a] = MulReq{&rf[a], &I[act[a]], t[act[a]], h[a]};
    std::vector<Poly> E = mul(rq);
    {
      std::vector<slot_t> e;
      for (const Poly &p : E)
        e.insert(e.end(), p.begin(), p.end());
      std::vector<slot_t> z(e.size(), ZERO);
      std::vector<int> ng(e.size(), 1);
      M.add((unsigned)e.size(), e.data(), z.data(), e.data(),
            ng.data()); // E = -E
    }
    for (size_t a = 0; a < na; a++)
      rq[a] = MulReq{&I[act[a]], &E[a], 0, h[a]};
    std::vector<Poly> U = mul(rq);
    for (size_t a = 0; a < na; a++) {
      const size_t i = act[a];
      give(E[a]);
      I[i].insert(I[i].end(), U[a].begin(), U[a].end());
      t[i] = t2[a];
    }
  }
  return I;
}

Poly PolyOps::inverse(const Poly &F, unsigned s) {
  return inverse(std::vector<const Poly *>{&F}, std::vector<unsigned>{s})[0];
}

std::vector<Poly> PolyOps::rem(const std::vector<RemReq> &r) {
  const size_t n = r.size();
  std::vector<Poly> R(n);
  std::vector<size_t> div; // the requests that need a division
  for (size_t i = 0; i < n; i++) {
    const unsigned s = (unsigned)r[i].F->size() - 1,
                   la = (unsigned)r[i].A->size();
    R[i] = take(s);
    if (la <= s)
      for (unsigned j = 0; j < s; j++)
        M.copy(R[i][j], j < la ? (*r[i].A)[j] : ZERO);
    else
      div.push_back(i);
  }
  const size_t nd = div.size();
  if (!nd)
    return R;
  // rev(Q) = rev(A) / rev(F) mod X^m (m the quotient length), from A's
  // top m coefficients.
  std::vector<Poly> ra(nd), ifm(nd);
  std::vector<MulReq> rq(nd);
  for (size_t d = 0; d < nd; d++) {
    const RemReq &q = r[div[d]];
    const unsigned m = (unsigned)(q.A->size() - (q.F->size() - 1));
    ra[d].assign(q.A->rbegin(), q.A->rbegin() + m);
    ifm[d].assign(q.IF->begin(), q.IF->begin() + m);
    rq[d] = MulReq{&ra[d], &ifm[d], 0, m};
  }
  std::vector<Poly> qr = mul(rq);
  // R = A - F*Q, of which only the low s coefficients survive; F's
  // leading term only reaches X^s and above.
  std::vector<Poly> Q(nd), Fl(nd);
  for (size_t d = 0; d < nd; d++) {
    const RemReq &q = r[div[d]];
    const unsigned s = (unsigned)q.F->size() - 1;
    Q[d].assign(qr[d].rbegin(), qr[d].rend());
    Fl[d].assign(q.F->begin(), q.F->begin() + s);
    rq[d] = MulReq{&Fl[d], &Q[d], 0, s};
  }
  std::vector<Poly> P = mul(rq);
  std::vector<slot_t> dst, a, b;
  for (size_t d = 0; d < nd; d++) {
    const RemReq &q = r[div[d]];
    const unsigned s = (unsigned)q.F->size() - 1;
    dst.insert(dst.end(), R[div[d]].begin(), R[div[d]].end());
    a.insert(a.end(), q.A->begin(), q.A->begin() + s);
    b.insert(b.end(), P[d].begin(), P[d].end());
  }
  std::vector<int> ng(dst.size(), 1);
  // |A| < 2n, |P| < n: |A - P| < 3n, a legal product input.
  M.add((unsigned)dst.size(), dst.data(), a.data(), b.data(), ng.data());
  for (size_t d = 0; d < nd; d++) {
    give(qr[d]);
    give(P[d]);
  }
  return R;
}

Poly PolyOps::rem(const Poly &A, const Poly &F, const Poly &IF) {
  return rem(std::vector<RemReq>{RemReq{&A, &F, &IF}})[0];
}

Poly PolyOps::mulrem(const Poly &A, const Poly &B, const Poly &F,
                     const Poly &IF) {
  const unsigned s = (unsigned)F.size() - 1,
                 la = (unsigned)(A.size() + B.size() - 1);
  if (la <= s) // no reduction: the product, padded with zeros
    return mul(A, B, 0, s);
  const unsigned m = la - s; // quotient length
  // A*B, kept exact in region 1 (the quotient's product below is a
  // polymul(), which uses region 0); its top m coefficients reduced.
  const unsigned NK = M.kron_len((unsigned)A.size(), (unsigned)B.size());
  M.kprod(1, (unsigned)A.size(), A.data(), (unsigned)B.size(), B.data(),
          NK);
  Poly top = take(m);
  M.kout(1, -1, s, m, top.data());
  // rev(Q) = rev(A*B) / rev(F) mod X^m.
  const Poly rt(top.rbegin(), top.rend()), ifm(IF.begin(), IF.begin() + m);
  Poly qr = mul(rt, ifm, 0, m);
  const Poly Q(qr.rbegin(), qr.rend()), Fl(F.begin(), F.begin() + s);
  // R = (A*B - F*Q) mod X^s, F's leading term only reaching X^s and up:
  // F*Q at A*B's length in region 0, the low s coefficients of the
  // difference reduced once each.
  M.kprod(0, s, Fl.data(), m, Q.data(), NK);
  Poly R = take(s);
  M.kout(1, 0, 0, s, R.data());
  give(top);
  give(qr);
  return R;
}

PolyOps::Tree PolyOps::product_tree(const std::vector<Poly> &leaves,
                                    bool keep) {
  Tree t;
  t.push_back(leaves);
  // The caller owns the leaves, even one carried up a level or more.
  std::set<slot_t> leaf;
  for (const Poly &p : leaves)
    if (!p.empty())
      leaf.insert(p[0]);
  while (t.back().size() > 1) {
    std::vector<Poly> nx;
    {
      const std::vector<Poly> &lv = t.back();
      std::vector<MulReq> rq;
      for (size_t i = 0; i + 1 < lv.size(); i += 2)
        rq.push_back(MulReq{&lv[i], &lv[i + 1], 0,
                            (unsigned)(lv[i].size() + lv[i + 1].size() - 1)});
      nx = mul(rq);
      if (lv.size() & 1)
        nx.push_back(lv.back()); // carried up: the same slots
    }
    if (!keep && t.size() > 1) {
      // Drop this level, except a node carried up and the leaves.
      std::vector<Poly> &lv = t.back();
      for (size_t i = 0; i + 1 < lv.size(); i += 2)
        for (size_t j : {i, i + 1})
          if (!leaf.count(lv[j][0]))
            give(lv[j]);
      lv.clear();
    }
    t.push_back(nx);
  }
  return t;
}

void PolyOps::give(const Tree &t, bool leaves_too) {
  // A node carried up a level shares its slots with the level above, so
  // give each slot once -- and none of the leaves' unless asked, even
  // where a leaf was carried up.
  std::set<slot_t> seen;
  if (!leaves_too)
    for (const Poly &p : t[0])
      seen.insert(p.begin(), p.end());
  for (size_t l = leaves_too ? 0 : 1; l < t.size(); l++)
    for (const Poly &p : t[l])
      for (slot_t s : p)
        if (seen.insert(s).second)
          give(s);
}

std::vector<slot_t> PolyOps::evaluate(const Poly &A, const Tree &tree,
                                      const Poly &IF) {
  // Root: (A mod F)/F = A/F = sum_{u >= 0} y_u X^-(u+1), and with
  // F = X^d rev(F)(1/X), A = X^(d-1) rev_d(A)(1/X):
  // y = rev_d(A) * IF mod X^d.
  const Poly &root = tree.back()[0];
  const unsigned d = (unsigned)root.size() - 1, la = (unsigned)A.size();
  Poly rA(d), ifd(IF.begin(), IF.begin() + d);
  for (unsigned u = 0; u < d; u++)
    rA[u] = d - 1 - u < la ? A[d - 1 - u] : ZERO;
  std::vector<Poly> cur(1, mul(rA, ifd, 0, d));
  for (size_t l = tree.size() - 1; l > 0; l--) {
    const std::vector<Poly> &kids = tree[l - 1];
    std::vector<Poly> nx(kids.size());
    // Child c of parent i with sibling o: y_c[t] = sum_s o_s y_i[t + s]
    // for t < deg c, i.e. coefficients deg o .. deg o + deg c - 1 of
    // rev(o) * y_i.
    std::vector<size_t> par, kid, sib;
    for (size_t i = 0; i < cur.size(); i++) {
      const size_t a = 2 * i, b = 2 * i + 1;
      if (b >= kids.size()) { // a carried node: same polynomial
        nx[a] = cur[i];
        continue;
      }
      par.insert(par.end(), {i, i});
      kid.insert(kid.end(), {a, b});
      sib.insert(sib.end(), {b, a});
    }
    std::vector<Poly> rev(kid.size());
    for (size_t x = 0; x < kid.size(); x++)
      rev[x].assign(kids[sib[x]].rbegin(), kids[sib[x]].rend());
    std::vector<MulReq> rq(kid.size());
    for (size_t x = 0; x < kid.size(); x++)
      rq[x] = MulReq{&rev[x], &cur[par[x]],
                     (unsigned)kids[sib[x]].size() - 1,
                     (unsigned)kids[kid[x]].size() - 1};
    std::vector<Poly> Y = mul(rq);
    for (size_t x = 0; x < kid.size(); x++) {
      nx[kid[x]] = Y[x];
      if (x & 1)
        give(cur[par[x]]); // both children done
    }
    cur.swap(nx);
  }
  std::vector<slot_t> vals(cur.size());
  for (size_t j = 0; j < cur.size(); j++)
    vals[j] = cur[j][0]; // a linear leaf's single coefficient
  return vals;
}

} // namespace gmpecmfft_detail
