"""Genera las graficas y tablas de Resultados a partir de resultados.csv.

Uso:  python graficas.py [--csv resultados.csv] [--out ../figuras]

Toma la mediana de las repeticiones de cada configuracion. Ajusta por minimos
cuadrados las constantes del modelo teorico del informe:
    T_comp = k * (n'/sqrt p) * (2*ceil(log2(n'/sqrt p)) + 1)
    T_comm = alpha * (ceil(log2 p) + 3*ceil(log2 sqrt p))
           + beta  * (n - 2n/p + n/sqrt p + 2*(n/sqrt p)*ceil(log2 sqrt p))
y escribe los valores en resumen.txt junto con las tablas de medianas.
"""
import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))

# Paleta categorica de referencia (modo claro, orden fijo): azul, naranja, aqua, amarillo
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRIS = "#8a8984"
INK = "#0b0b0b"
INK2 = "#52514e"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#b9b8b2", "axes.labelcolor": INK, "axes.titlesize": 10.5,
    "axes.titleweight": "bold", "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.7,
    "lines.linewidth": 2.0, "lines.markersize": 6.5, "legend.frameon": False,
    "legend.fontsize": 8.5,
})


def lg(x):
    return math.ceil(math.log2(x)) if x > 1 else 0


def modelo_comp(n, p):
    q = int(round(math.sqrt(p)))
    npad = ((n + p - 1) // p) * p
    b = npad / q
    return b * (2 * lg(b) + 1)


def modelo_comm(n, p):
    q = int(round(math.sqrt(p)))
    a = lg(p) + 3 * lg(q)
    b = n - 2 * n / p + n / q + 2 * (n / q) * lg(q)   # ec. Tcomm2 (gamma absorbido en beta)
    if p == 1:
        a, b = 0.0, 0.0
    return a, b


def guardar(fig, out, nombre):
    fig.savefig(os.path.join(out, nombre + ".pdf"))
    fig.savefig(os.path.join(out, nombre + ".png"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(AQUI, "resultados.csv"))
    ap.add_argument("--out", default=os.path.join(AQUI, "..", "figuras"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = pd.read_csv(a.csv)
    malos = df[df.estado != "OK"]
    num = ["t_scatter", "t_gossip", "t_bcast", "t_sort", "t_rank", "t_reduce",
           "t_comp", "t_comm", "t_par", "t_post", "comp_max", "comp_total"]
    med = (df.groupby(["tag", "version", "n", "p"], as_index=False)[num].median())
    reps = df.groupby(["tag", "version", "n", "p"]).size().min()
    lines = [f"Filas: {len(df)}  configuraciones: {len(med)}  repeticiones minimas: {reps}",
             f"Ejecuciones con verificacion distinta de OK: {len(malos)}", ""]

    def sel(tag, version, **kw):
        d = med[(med.tag == tag) & (med.version == version)]
        for k, v in kw.items():
            d = d[d[k] == v]
        return d.sort_values(["n", "p"])

    rs = sel("fuerte", "beta3")
    qs = sel("fuerte", "quicksort")
    Ns = sorted(rs.n.unique())
    colores_n = dict(zip(Ns, [C1, C2, C3, C4]))
    etiqueta_n = {n: f"n = {n/1e6:g}·10⁶" for n in Ns}

    # ---------- Ajuste del modelo teorico ----------
    xc = np.array([modelo_comp(n, p) for n, p in zip(rs.n, rs.p)])
    k = float(np.dot(xc, rs.t_comp) / np.dot(xc, xc))
    # alpha, beta: solo con p in {4, 9} (p = 16 comparte nucleos por SMT)
    A = np.array([modelo_comm(n, p) for n, p in zip(rs.n, rs.p)])
    msk = rs.p.isin([4, 9]).values
    (alpha, beta), *_ = np.linalg.lstsq(A[msk], rs.t_comm.values[msk], rcond=None)
    alpha, beta = max(alpha, 0.0), max(beta, 0.0)
    pred_comp = k * xc
    r2c = 1 - np.sum((rs.t_comp - pred_comp) ** 2) / np.sum((rs.t_comp - rs.t_comp.mean()) ** 2)
    # k por cada n, calibrado con la corrida p = 1 (costo real por comparacion)
    kn = {n: float(rs[(rs.n == n) & (rs.p == 1)].t_comp.iloc[0] / modelo_comp(n, 1)) for n in Ns}
    lines += ["Modelo teorico ajustado (Beta 3, escalabilidad fuerte):",
              f"  k global (s por comparacion del modelo) = {k:.3e}   R2(T_comp) = {r2c:.4f}",
              "  k por n (calibrado con p=1): " + "  ".join(f"n={n:g}: {v*1e9:.2f} ns" for n, v in kn.items()),
              f"  alpha = {alpha:.3e} s   beta = {beta:.3e} s/par (ajuste con p=4,9)", ""]
    lines.append("Prediccion del modelo (k por n) vs medido, T_comp y T_par:")
    for n in Ns:
        for p in [1, 4, 9, 16]:
            r = rs[(rs.n == n) & (rs.p == p)].iloc[0]
            pc_ = kn[n] * modelo_comp(n, p); pm_ = float(np.dot(modelo_comm(n, p), [alpha, beta]))
            lines.append(f"  n={n:>9d} p={p:2d}  T_comp med={r.t_comp:7.3f} mod={pc_:7.3f}   "
                         f"T_comm med={r.t_comm:6.3f} mod={pm_:6.3f}   T_par med={r.t_par:7.3f} mod={pc_+pm_:7.3f}")
    lines.append("")

    # ---------- 1. Tiempos de ejecucion, computo y comunicacion ----------
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, col, tit in zip(axs, ["t_par", "t_comp", "t_comm"],
                            ["Ejecución $T_{par}$", "Cómputo $T_{comp}$", "Comunicación $T_{comm}$"]):
        for n in Ns:
            d = rs[rs.n == n]
            ax.plot(d.p, d[col], "o-", color=colores_n[n], label=etiqueta_n[n])
        ax.set_yscale("log"); ax.set_xticks([1, 4, 9, 16]); ax.set_xlabel("procesos p")
        ax.set_title(tit); ax.axvspan(8.5, 16.5, color="#f2f1ec", zorder=0, lw=0)
    axs[0].set_ylabel("tiempo (s)")
    axs[0].legend(loc="lower left")
    axs[2].text(12.5, axs[2].get_ylim()[0] * 1.3, "SMT\n(p > 8 núcleos)", ha="center",
                color=INK2, fontsize=8)
    guardar(fig, a.out, "res_tiempos")

    # ---------- 2. Desglose por fase (n = 32e6) ----------
    nbig = Ns[-1]
    d = rs[rs.n == nbig]
    fases = [("t_scatter", "Scatter", C2), ("t_gossip", "Gossip", C4), ("t_bcast", "Broadcast", C3),
             ("t_reduce", "Reduce", "#e87ba4"), ("t_sort", "Sort", C1), ("t_rank", "Ranking", "#6250d6")]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(d)); base = np.zeros(len(d))
    for col, lab, c in fases:
        ax.bar(x, d[col].values, 0.55, bottom=base, color=c, label=lab, edgecolor="white", linewidth=1.2)
        base += d[col].values
    ax.set_xticks(x, [f"p = {p}" for p in d.p]); ax.set_ylabel("tiempo (s, máx. entre procesos)")
    ax.set_title(f"Desglose por fase de la Beta 3, n = {nbig/1e6:g}·10⁶", loc="left")
    ax.set_ylim(0, base.max() * 1.32)
    ax.legend(ncol=3, loc="upper center"); ax.grid(axis="x", visible=False)
    guardar(fig, a.out, "res_fases")
    lines.append(f"Desglose por fase, n = {nbig}:")
    lines.append(d[["p", "t_scatter", "t_gossip", "t_bcast", "t_sort", "t_rank", "t_reduce",
                    "t_comp", "t_comm", "t_par", "t_post"]].to_string(index=False))
    lines.append("")

    # ---------- 3. Medido vs teoria ----------
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
    for n in Ns:
        dd = rs[rs.n == n]
        pc = np.array([kn[n] * modelo_comp(n, p) for p in dd.p])
        pm = np.array([np.dot(modelo_comm(n, p), [alpha, beta]) for p in dd.p])
        axs[0].plot(dd.p, dd.t_comp, "o", color=colores_n[n], label=etiqueta_n[n] + " medido")
        axs[0].plot(dd.p, pc, "--", color=colores_n[n], lw=1.4)
        axs[1].plot(dd.p, dd.t_par, "o", color=colores_n[n], label=etiqueta_n[n])
        axs[1].plot(dd.p, pc + pm, "--", color=colores_n[n], lw=1.4)
    for ax in axs:
        ax.set_yscale("log"); ax.set_xticks([1, 4, 9, 16]); ax.set_xlabel("procesos p")
        ax.set_ylabel("tiempo (s)")
    axs[0].set_title("$T_{comp}$: medido (puntos) y modelo $k_n\\,W/p$ (línea)", loc="left")
    axs[1].set_title("$T_{par}$: medido y modelo $T_{comp}+T_{comm}$", loc="left")
    axs[0].legend(loc="upper right", fontsize=7.5, ncol=2)
    guardar(fig, a.out, "res_teoria")

    # ---------- 4. Speedup y eficiencia ----------
    def speedup(d):
        t1 = d[d.p == 1].t_par.values[0]
        return d.p.values, t1 / d.t_par.values

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    pp = np.linspace(1, 16, 100)
    axs[0].plot(pp, pp, ":", color=GRIS, lw=1.3)
    axs[0].text(10.5, 11.6, "ideal S = p", color=INK2, fontsize=8, rotation=30)
    axs[0].plot(pp, np.sqrt(pp), ":", color=GRIS, lw=1.3)
    axs[0].text(15.2, math.sqrt(15.2) - 0.75, "√p", color=INK2, fontsize=8.5)
    for n, mk in [(Ns[-2], "s"), (Ns[-1], "o")]:
        p_, s_ = speedup(rs[rs.n == n]); axs[0].plot(p_, s_, mk + "-", color=C1,
                                                    label=f"Ranking Sort (Beta 3), {etiqueta_n[n]}",
                                                    alpha=1 if n == Ns[-1] else .6)
        p_, s_ = speedup(qs[qs.n == n]); axs[0].plot(p_, s_, mk + "-", color=C2,
                                                    label=f"Quicksort paralelo, {etiqueta_n[n]}",
                                                    alpha=1 if n == Ns[-1] else .6)
        p_r, s_r = speedup(rs[rs.n == n]); p_q, s_q = speedup(qs[qs.n == n])
        axs[1].plot(p_r, s_r / p_r, mk + "-", color=C1, alpha=1 if n == Ns[-1] else .6)
        axs[1].plot(p_q, s_q / p_q, mk + "-", color=C2, alpha=1 if n == Ns[-1] else .6)
    axs[1].plot(pp, 1 / np.sqrt(pp), ":", color=GRIS, lw=1.3)
    axs[1].text(12, 1 / math.sqrt(12) + 0.04, "1/√p", color=INK2, fontsize=8.5)
    for ax in axs:
        ax.set_xticks([1, 2, 4, 8, 9, 16]); ax.set_xlabel("procesos p")
        ax.axvspan(8.5, 16.5, color="#f2f1ec", zorder=0, lw=0)
    axs[0].set_ylabel("speedup S(p) = T(1) / T(p)"); axs[0].set_title("Speedup", loc="left")
    axs[1].set_ylabel("eficiencia E(p) = S(p) / p"); axs[1].set_title("Eficiencia", loc="left")
    axs[1].set_ylim(0, 1.15)
    axs[0].legend(loc="upper left", fontsize=7.8)
    guardar(fig, a.out, "res_speedup")

    lines.append("Speedup y eficiencia (T_par):")
    for nombre, d0 in [("beta3", rs), ("quicksort", qs)]:
        for n in Ns:
            p_, s_ = speedup(d0[d0.n == n])
            lines.append(f"  {nombre:9s} n={n:>9d}  " + "  ".join(
                f"p={pi}: S={si:.2f} E={si/pi:.2f}" for pi, si in zip(p_, s_)))
    lines.append("")

    # ---------- 5. FLOPs (comparaciones) vs p ----------
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
    for n in Ns:
        dd = rs[rs.n == n]
        axs[0].plot(dd.p, dd.comp_total / 1e9, "o-", color=colores_n[n], label=etiqueta_n[n])
        axs[0].plot(dd.p, dd.comp_max / 1e9, "s--", color=colores_n[n], lw=1.3, ms=5)
        nop = dd.comp_total + dd.n * (np.sqrt(dd.p) - 1)
        ts = float(dd[dd.p == 1].comp_total.iloc[0])
        axs[1].plot(dd.p, nop / dd.t_par / 1e6, "o-", color=colores_n[n], label=etiqueta_n[n])
        axs[1].plot(dd.p, ts / dd.t_par / 1e6, "s--", color=colores_n[n], lw=1.3, ms=5)
    axs[0].set_title("Comparaciones: W total (—), máx./proceso (- -)", loc="left")
    axs[0].set_ylabel("comparaciones (×10⁹)")
    axs[0].set_yscale("log")
    axs[1].set_title("MOPS: total $F$ (—) y útil $F_{útil}$ (- -)", loc="left")
    axs[1].set_ylabel("millones de operaciones / s")
    for ax in axs:
        ax.set_xticks([1, 4, 9, 16]); ax.set_xlabel("procesos p")
        ax.axvspan(8.5, 16.5, color="#f2f1ec", zorder=0, lw=0)
    axs[1].legend(loc="upper left")
    guardar(fig, a.out, "res_flops")
    lines.append("Comparaciones (FLOP del codigo) y tasa:")
    lines.append(rs[["n", "p", "comp_max", "comp_total", "t_par"]].assign(
        gcomp_s=lambda x: x.comp_total / x.t_par / 1e9,
        modelo_max=lambda x: [modelo_comp(n, p) for n, p in zip(x.n, x.p)]).to_string(index=False))
    lines.append("")

    # ---------- Metricas por hipotesis H1, H4, H5, H6 ----------
    lines.append("H1: speedup de computo S_comp = T_comp(1)/T_comp(p) vs teoria sqrt(p)/lambda:")
    for n in Ns:
        dd = rs[rs.n == n]
        tc1 = float(dd[dd.p == 1].t_comp.iloc[0]); c1 = float(dd[dd.p == 1].comp_max.iloc[0])
        for _, r in dd[dd.p > 1].iterrows():
            q = math.sqrt(r.p); lam = (2 * lg(n / q) + 1) / (2 * lg(n) + 1)
            lines.append(f"  n={n:>9d} p={int(r.p):2d}  S_comp(tiempo)={tc1 / r.t_comp:5.2f}  "
                         f"S_comp(operaciones)={c1 / r.comp_max:5.2f}  teoria sqrt(p)/lambda={q / lam:5.2f}  "
                         f"S={float(dd[dd.p == 1].t_par.iloc[0]) / r.t_par:5.2f}")
    lines.append("H4: F = (W + n(sqrt p - 1))/T_par y F_util = Ts/T_par, en MOPS; F/F_util vs sqrt(p)*lambda:")
    for n in Ns:
        dd = rs[rs.n == n]
        ts = float(dd[dd.p == 1].comp_total.iloc[0]); f1 = None
        for _, r in dd.iterrows():
            q = math.sqrt(r.p); lam = (2 * lg(n / q) + 1) / (2 * lg(n) + 1)
            nop = r.comp_total + r.n * (q - 1)
            F = nop / r.t_par / 1e6; Fu = ts / r.t_par / 1e6
            f1 = F if f1 is None else f1
            lines.append(f"  n={n:>9d} p={int(r.p):2d}  F={F:7.1f}  F/F(1)={F / f1:5.2f}  F_util={Fu:7.1f}  "
                         f"F/F_util={F / Fu:5.2f}  sqrt(p)*lambda={q * lam:5.2f}")
    lines.append(f"H5: velocidad n/T_par (Mclaves/s); techo 1/beta = {1 / beta / 1e6:,.0f} Mclaves/s")
    for n in Ns:
        dd = rs[rs.n == n]
        lines.append(f"  n={n:>9d}  " + "  ".join(f"p={int(r.p)}: {r.n / r.t_par / 1e6:5.2f}" for _, r in dd.iterrows()))
    lines.append("H6: cociente T_comp(RS)/T_comp(QS) y su crecimiento respecto de p=1:")
    for n in Ns:
        base = None
        for p_ in [1, 4, 16]:
            a_ = rs[(rs.n == n) & (rs.p == p_)].t_comp.iloc[0]; b_ = qs[(qs.n == n) & (qs.p == p_)].t_comp.iloc[0]
            ratio = a_ / b_; base = ratio if base is None else base
            sc = rs[(rs.n == n) & (rs.p == p_)].t_scatter.iloc[0] if "t_scatter" in rs else float("nan")
            lines.append(f"  n={n:>9d} p={p_:2d}  RS/QS={ratio:6.2f}  crecimiento={ratio / base:5.2f}  "
                         f"T_par RS={rs[(rs.n == n) & (rs.p == p_)].t_par.iloc[0]:.3f} "
                         f"QS={qs[(qs.n == n) & (qs.p == p_)].t_par.iloc[0]:.3f}")
    lines.append("")

    # ---------- 6. Velocidad del algoritmo: elementos ordenados por segundo ----------
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for n, mk in [(Ns[-1], "o")]:
        d1 = rs[rs.n == n]; d2 = qs[qs.n == n]
        ax.plot(d1.p, d1.n / d1.t_par / 1e6, mk + "-", color=C1, label="Ranking Sort (Beta 3)")
        ax.plot(d2.p, d2.n / d2.t_par / 1e6, mk + "-", color=C2, label="Quicksort paralelo")
    ax.set_xticks([1, 2, 4, 8, 9, 16]); ax.set_xlabel("procesos p")
    ax.set_ylabel("millones de claves / s"); ax.axvspan(8.5, 16.5, color="#f2f1ec", zorder=0, lw=0)
    ax.set_title(f"Velocidad n / $T_{{par}}$, {etiqueta_n[Ns[-1]]}", loc="left"); ax.legend()
    guardar(fig, a.out, "res_velocidad")

    # ---------- 7. Escalabilidad con n variable: Ranking vs Quicksort ----------
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, p in zip(axs, [4, 16]):
        d1 = rs[rs.p == p]; d2 = qs[qs.p == p]
        ax.plot(d1.n, d1.t_par, "o-", color=C1, label="Ranking Sort (Beta 3)")
        ax.plot(d2.n, d2.t_par, "o-", color=C2, label="Quicksort paralelo")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("n (claves)")
        ax.set_ylabel("$T_{par}$ (s)"); ax.set_title(f"p = {p}", loc="left")
    axs[0].legend(loc="upper left")
    guardar(fig, a.out, "res_escala_n")

    # ---------- 8. Escalabilidad debil ----------
    db = sel("debil_bloque", "beta3"); dn = sel("debil_np", "beta3"); dq = sel("debil_np", "quicksort")
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
    axs[0].plot(db.p, db.t_comp, "o-", color=C1, label="$T_{comp}$")
    axs[0].plot(db.p, db.t_comm, "o-", color=C2, label="$T_{comm}$")
    axs[0].plot(db.p, db.t_par, "o-", color=C3, label="$T_{par}$")
    axs[0].set_title("Beta 3, bloque $n/\\sqrt{p}$ = 4·10⁶ fijo", loc="left")
    axs[1].plot(dn.p, dn.t_par, "o-", color=C1, label="Ranking Sort (Beta 3)")
    axs[1].plot(dq.p, dq.t_par, "o-", color=C2, label="Quicksort paralelo")
    axs[1].set_title("n/p = 2·10⁶ fijo (n = 2·10⁶·p)", loc="left")
    for ax in axs:
        ax.set_xticks([1, 2, 4, 8, 9, 16]); ax.set_xlabel("procesos p"); ax.set_ylabel("tiempo (s)")
        ax.axvspan(8.5, 16.5, color="#f2f1ec", zorder=0, lw=0); ax.legend(loc="upper left")
        ax.set_ylim(bottom=0)
    guardar(fig, a.out, "res_debil")
    lines.append("Escalabilidad debil (bloque fijo):")
    lines.append(db[["n", "p", "t_comp", "t_comm", "t_par", "comp_max"]].to_string(index=False))
    lines.append("Escalabilidad debil (n/p fijo):")
    lines.append(pd.concat([dn, dq])[["version", "n", "p", "t_comp", "t_comm", "t_par"]].to_string(index=False))
    lines.append("")

    # ---------- 9. Betas ----------
    b1 = sel("b1vsb3", "beta1"); b3 = sel("b1vsb3", "beta3")
    g_ag = sel("gossip", "beta2-ag"); g_mb = sel("gossip", "beta2-mb")
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    x = np.arange(len(b1)); w = 0.36
    axs[0].bar(x - w/2, b1.t_comm.values, w, color=C4, label="Beta 1 (colectivas v)", edgecolor="white")
    axs[0].bar(x + w/2, b3.t_comm.values, w, color=C1, label="Beta 3 (colectivas simples)", edgecolor="white")
    axs[0].set_xticks(x, [f"p = {p}" for p in b1.p]); axs[0].set_ylabel("$T_{comm}$ (s)")
    axs[0].set_title("Comunicación, n = 16·10⁶", loc="left"); axs[0].legend(loc="upper left")
    x = np.arange(len(g_ag))
    axs[1].bar(x - w/2, g_ag.t_gossip.values * 1e3, w, color=C1, label="V1: Allgatherv", edgecolor="white")
    axs[1].bar(x + w/2, g_mb.t_gossip.values * 1e3, w, color=C2, label="V2: Multi-Broadcast", edgecolor="white")
    axs[1].set_xticks(x, [f"p = {p}" for p in g_ag.p]); axs[1].set_ylabel("tiempo del Gossip (ms)")
    axs[1].set_title("Beta 2: fase Gossip, n = 16·10⁶", loc="left"); axs[1].legend(loc="upper left")
    for ax in axs: ax.grid(axis="x", visible=False)
    guardar(fig, a.out, "res_betas")
    lines.append("Beta 1 vs Beta 3 (n=16e6):")
    lines.append(pd.concat([b1, b3])[["version", "p", "t_scatter", "t_gossip", "t_bcast", "t_reduce",
                                      "t_comm", "t_comp", "t_par", "t_post"]].to_string(index=False))
    lines.append("Beta 2 Gossip:")
    lines.append(pd.concat([g_ag, g_mb])[["version", "p", "t_gossip", "t_comm", "t_par"]].to_string(index=False))
    lines.append("")

    # ---------- 10. Hibrido y secuencial (tablas) ----------
    hb = med[med.tag == "hibrido"].sort_values(["p", "version"])
    lines.append("Hibrido (n=16e6):")
    lines.append(hb[["version", "p", "t_sort", "t_rank", "t_comp", "t_comm", "t_par"]].to_string(index=False))
    sq = sel("secuencial", "secuencial")
    n2 = sq.n.astype(float) ** 2
    cs = float(np.dot(n2, sq.t_par) / np.dot(n2, n2))
    lines.append("Secuencial O(n^2):  T = c n^2,  c = %.3e s" % cs)
    lines.append(sq[["n", "t_par"]].assign(modelo=lambda x: cs * x.n.astype(float) ** 2).to_string(index=False))
    t16 = rs[(rs.n == Ns[-1])]
    lines.append("Speedup relativo vs Ranking Sort secuencial extrapolado (c n^2), n=%d:" % Ns[-1])
    for _, r in t16.iterrows():
        lines.append(f"  p={int(r.p)}: S_rel = {cs * float(r.n) ** 2 / r.t_par:,.0f}")

    with open(os.path.join(AQUI, "resumen.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
