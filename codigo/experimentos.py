"""Ejecuciones para la seccion de Resultados del informe.

Uso:  python experimentos.py [--reps 5] [--mpiexec mpiexec] [--salida resultados.csv]

Corre cada configuracion `reps` veces y guarda una fila por ejecucion (la linea
"CSV,..." que imprime cada programa) en resultados.csv. Requiere los binarios en
bin/ (compilar_windows.ps1 o make).

Equipo de referencia: AMD Ryzen 7 5800X, 8 nucleos fisicos / 16 hilos. Las
corridas con p > 8 usan hilos SMT (hyperthreading) y se marcan asi en el informe.
"""
import argparse
import os
import subprocess
import sys
import time

AQUI = os.path.dirname(os.path.abspath(__file__))
EXT = ".exe" if os.name == "nt" else ""
COLS = ("tag,version,n,p,procesos,t_scatter,t_gossip,t_bcast,t_sort,t_rank,t_reduce,"
        "t_comp,t_comm,t_par,t_post,comp_max,comp_total,estado")


def binario(nombre):
    return os.path.join(AQUI, "bin", nombre + EXT)


def plan():
    """Lista de (tag, procesos, programa, argumentos, hilos OpenMP)."""
    P_GRID = [1, 4, 9, 16]          # cuadrados perfectos para la grilla q x q
    P_QS = [1, 2, 4, 8, 16]         # potencias de 2 para el quicksort
    N_FUERTE = [1_000_000, 4_000_000, 16_000_000, 32_000_000]
    c = []
    # 1) Escalabilidad fuerte: n fijo, p creciente
    for n in N_FUERTE:
        for p in P_GRID:
            c.append(("fuerte", p, "rs_beta3", [n], 1))
        for p in P_QS:
            c.append(("fuerte", p, "qs_par", [n], 1))
    # 2) Escalabilidad debil
    for p in P_GRID:                 # bloque n/sqrt(p) constante (H3)
        c.append(("debil_bloque", p, "rs_beta3", [4_000_000 * int(p ** 0.5)], 1))
    for p in P_GRID:                 # datos por proceso n/p constantes
        c.append(("debil_np", p, "rs_beta3", [2_000_000 * p], 1))
    for p in P_QS:
        c.append(("debil_np", p, "qs_par", [2_000_000 * p], 1))
    # 3) Beta 1 vs Beta 3
    for p in P_GRID:
        c.append(("b1vsb3", p, "rs_beta1", [16_000_000], 1))
        c.append(("b1vsb3", p, "rs_beta3", [16_000_000], 1))
    # 4) Beta 2: Gossip Allgatherv vs Multi-Broadcast
    for p in [4, 9, 16]:
        c.append(("gossip", p, "rs_beta2", [16_000_000, "ag"], 1))
        c.append(("gossip", p, "rs_beta2", [16_000_000, "mb"], 1))
    # 5) Hibrido MPI+OpenMP (p procesos x t hilos <= 16 hilos logicos)
    for p, t in [(1, 1), (1, 2), (1, 4), (1, 8), (4, 1), (4, 2), (4, 4)]:
        c.append(("hibrido", p, "rs_hibrido", [16_000_000], t))
    # 6) Secuencial O(n^2): referencia del speedup relativo
    for n in [10_000, 20_000, 40_000, 80_000]:
        c.append(("secuencial", 1, "rs_sec", [n], 1))
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--mpiexec", default="mpiexec")
    ap.add_argument("--salida", default=os.path.join(AQUI, "resultados.csv"))
    a = ap.parse_args()

    casos = plan()
    with open(a.salida, "w", encoding="ascii", newline="\n") as f:
        f.write(COLS + "\n")
        t0 = time.time()
        for idx, (tag, p, prog, args, hilos) in enumerate(casos, 1):
            env = dict(os.environ, OMP_NUM_THREADS=str(hilos))
            if prog == "rs_sec":
                cmd = [binario(prog)] + [str(x) for x in args]
            else:
                cmd = [a.mpiexec, "-n", str(p), binario(prog)] + [str(x) for x in args]
            for r in range(a.reps):
                out = subprocess.run(cmd, capture_output=True, text=True, env=env).stdout
                filas = [l.strip() for l in out.splitlines() if l.startswith("CSV,")]
                if not filas:
                    print("  sin salida CSV:", " ".join(cmd), file=sys.stderr)
                for l in filas:
                    f.write(tag + "," + l[4:] + "\n")
                f.flush()
            print(f"[{idx:3d}/{len(casos)}] {time.time() - t0:7.1f}s  {tag:13s} p={p:<2d} "
                  f"t={hilos} {prog} {' '.join(map(str, args))}", flush=True)
    print("Resultados en", a.salida)


if __name__ == "__main__":
    main()
