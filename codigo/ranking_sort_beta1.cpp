/*************************************************
* Proyecto 4 - Ranking Sort (ordenamiento eficiente por ranking)
* CS4052 Computacion Paralela y Distribuida - UTEC 2026-II
* Autores:     O. Chu, C. Rodriguez, C. Pajuelo
* objetivo:    BETA 1. Pipeline completo y correcto en grilla q x q:
*              Input -> Gossip -> Broadcast -> Sort -> Local ranking -> Reduce
*              con las variantes "v" de las colectivas (Scatterv, Allgatherv,
*              Gatherv) para admitir cualquier n (no necesariamente multiplo de p).
* compilacion: mpic++ -O3 -std=c++17 -o rs_beta1 ranking_sort_beta1.cpp
* ejecucion:   mpirun -np 4 ./rs_beta1 2000000 42
*              (argumentos: n (numero de claves), semilla)
* contenido:   codigo fuente en MPI-C++
*
* Convencion de comentarios:
*   [NUEVO]  comando que no aparece en las clases ni en el codigo del curso.
*   (U6.2)   colectiva listada en la clase U6.2 "Patrones de comunicacion en MPI".
 *************************************************/
#include <mpi.h>
#include <algorithm>   // std::sort, std::lower_bound
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>      // generador de claves reproducible (mt19937)
#include <vector>

struct Par { int key; int idx; };             // (clave, indice original)

// Orden lexicografico: primero clave, luego indice (regla de desempate)
static bool prec(const Par& a, const Par& b) {
    return a.key < b.key || (a.key == b.key && a.idx < b.idx);
}

enum { T_SCATTER, T_GOSSIP, T_BCAST, T_SORT, T_RANK, T_REDUCE, NT };

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int wrank, wsize;
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
    MPI_Comm_size(MPI_COMM_WORLD, &wsize);

    const int n = (argc > 1) ? std::atoi(argv[1]) : 1000000;
    const unsigned seed = (argc > 2) ? (unsigned)std::atoi(argv[2]) : 42u;
    if (n < 1) {
        if (wrank == 0) std::fprintf(stderr, "n debe ser >= 1\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Grilla q x q: trabajan q*q procesos, el resto queda ocioso
    int q = (int)std::sqrt((double)wsize);
    while ((q + 1) * (q + 1) <= wsize) ++q;
    while (q * q > wsize) --q;
    const int p = q * q;

    // [NUEVO] MPI_Comm_split / MPI_UNDEFINED: crea sub-comunicadores (grilla activa,
    // filas y columnas) para que las colectivas actuen solo sobre un subgrupo de
    // procesos. En clase todas las colectivas usaban MPI_COMM_WORLD.
    MPI_Comm grid;
    MPI_Comm_split(MPI_COMM_WORLD, wrank < p ? 0 : MPI_UNDEFINED, wrank, &grid);
    if (wrank >= p) { MPI_Finalize(); return 0; }

    const int k = wrank, i = k / q, j = k % q;  // fila i, columna j
    MPI_Comm row, col;
    MPI_Comm_split(grid, i, j, &row);           // rango en row == j
    MPI_Comm_split(grid, j, i, &col);           // rango en col == i

    // Tamanos: m_k = n/p + [k < n mod p];  b_j = suma de la columna j
    std::vector<int> m(p), b(q, 0);
    for (int t = 0; t < p; ++t) { m[t] = n / p + (t < n % p); b[t % q] += m[t]; }
    const int mk = m[k];

    // Conteos y desplazamientos (en enteros: un Par = 2 MPI_INT)
    std::vector<int> cnt(p), dsp(p);
    for (int t = 0, off = 0; t < p; ++t) { cnt[t] = 2 * m[t]; dsp[t] = 2 * off; off += m[t]; }
    std::vector<int> ccnt(q), cdsp(q);
    for (int r = 0, off = 0; r < q; ++r) {
        ccnt[r] = 2 * m[r * q + j]; cdsp[r] = 2 * off; off += m[r * q + j];
    }

    // El proceso 0 genera A (fuera del tiempo medido)
    std::vector<Par> A, C(mk);
    if (k == 0) {
        A.resize(n);
        std::mt19937 gen(seed);
        std::uniform_int_distribution<int> dist(0, n / 2);   // claves repetidas a proposito
        for (int t = 0; t < n; ++t) A[t] = { dist(gen), t };
    }

    double tt[NT] = {0};
    MPI_Barrier(grid);                                       // (U6.2)
    const double tini = MPI_Wtime();
    double t = tini;

    // ---- Fase 0: Input. El proceso 0 reparte los fragmentos C_k ----
    MPI_Scatterv(A.data(), cnt.data(), dsp.data(), MPI_INT,
                 C.data(), 2 * mk, MPI_INT, 0, grid);        // (U6.2)
    tt[T_SCATTER] = MPI_Wtime() - t; t = MPI_Wtime();

    // ---- Fase 1: Gossip. All-gather en la columna j  ->  S = B_j ----
    std::vector<Par> S(b[j]);
    MPI_Allgatherv(C.data(), 2 * mk, MPI_INT,
                   S.data(), ccnt.data(), cdsp.data(), MPI_INT, col);   // (U6.2)
    tt[T_GOSSIP] = MPI_Wtime() - t; t = MPI_Wtime();

    // ---- Fase 2: Broadcast. La diagonal P_ii envia B_i a su fila  ->  R = B_i ----
    std::vector<Par> R(b[i]);
    if (i == j) R = S;
    MPI_Bcast(R.data(), 2 * b[i], MPI_INT, i, row);
    tt[T_BCAST] = MPI_Wtime() - t; t = MPI_Wtime();

    // ---- Fase 3: Sort local de B_j ----
    std::sort(S.begin(), S.end(), prec);
    tt[T_SORT] = MPI_Wtime() - t; t = MPI_Wtime();

    // ---- Fase 4: Local ranking. r[t] = #{ y en B_j : y prec R[t] } (busqueda binaria) ----
    std::vector<int> r(b[i]), rk(b[i]);
    for (int u = 0; u < b[i]; ++u)
        r[u] = (int)(std::lower_bound(S.begin(), S.end(), R[u], prec) - S.begin());
    tt[T_RANK] = MPI_Wtime() - t; t = MPI_Wtime();

    // ---- Fase 5: Reduce. Suma por fila hacia la diagonal ----
    MPI_Reduce(r.data(), rk.data(), b[i], MPI_INT, MPI_SUM, i, row);  // (U6.2)
    tt[T_REDUCE] = MPI_Wtime() - t;
    const double tpar = MPI_Wtime() - tini;

    // ---- Fase 6: Colocacion. Las diagonales envian (rank, clave, indice) al proceso 0 ----
    // Los procesos no diagonales envian 0 elementos (por eso se necesita Gatherv).
    const double tp0 = MPI_Wtime();
    const int nsend = (i == j) ? 3 * b[i] : 0;
    std::vector<int> out(nsend);
    if (i == j)
        for (int u = 0; u < b[i]; ++u) { out[3*u] = rk[u]; out[3*u+1] = R[u].key; out[3*u+2] = R[u].idx; }
    std::vector<int> gcnt(p), gdsp(p), all;
    MPI_Gather(&nsend, 1, MPI_INT, gcnt.data(), 1, MPI_INT, 0, grid);
    if (k == 0) { for (int u = 0, off = 0; u < p; ++u) { gdsp[u] = off; off += gcnt[u]; } all.resize(3 * n); }
    MPI_Gatherv(out.data(), nsend, MPI_INT, all.data(), gcnt.data(), gdsp.data(), MPI_INT, 0, grid);  // (U6.2)
    std::vector<Par> B;
    if (k == 0) {
        B.assign(n, Par{0, -1});
        for (int u = 0; u < n; ++u) {
            const int pos = all[3*u];
            if (pos >= 0 && pos < n) B[pos] = { all[3*u+1], all[3*u+2] };
        }
    }
    const double tpost = MPI_Wtime() - tp0;

    // Verificacion en el proceso 0 (no cronometrada): permutacion, orden y estabilidad
    bool ok = true;
    if (k == 0) {
        std::vector<char> visto(n, 0);
        for (int u = 0; u < n && ok; ++u) {
            const int pos = all[3*u];
            if (pos < 0 || pos >= n || visto[pos]) ok = false; else visto[pos] = 1;
        }
        for (int u = 1; u < n && ok; ++u) if (!prec(B[u-1], B[u])) ok = false;
        for (int u = 0; u < n && ok; ++u)
            if (B[u].idx < 0 || B[u].idx >= n || B[u].key != A[B[u].idx].key) ok = false;
    }

    double ttmax[NT], tparmax;
    MPI_Reduce(tt, ttmax, NT, MPI_DOUBLE, MPI_MAX, 0, grid);
    MPI_Reduce(&tpar, &tparmax, 1, MPI_DOUBLE, MPI_MAX, 0, grid);

    if (k == 0) {
        const double tcomp = ttmax[T_SORT] + ttmax[T_RANK];
        const double tcomm = ttmax[T_SCATTER] + ttmax[T_GOSSIP] + ttmax[T_BCAST] + ttmax[T_REDUCE];
        std::printf("Beta1  n=%d  procesos=%d  grilla=%dx%d (p=%d, ociosos=%d)\n",
                    n, wsize, q, q, p, wsize - p);
        std::printf("  Scatterv=%.6f  Allgatherv=%.6f  Bcast=%.6f  Sort=%.6f  Rank=%.6f  Reduce=%.6f\n",
                    ttmax[T_SCATTER], ttmax[T_GOSSIP], ttmax[T_BCAST],
                    ttmax[T_SORT], ttmax[T_RANK], ttmax[T_REDUCE]);
        std::printf("  T_comp=%.6f s  T_comm=%.6f s  T_par=%.6f s  T_post=%.6f s\n",
                    tcomp, tcomm, tparmax, tpost);
        std::printf("  Verificacion: %s\n", ok ? "OK" : "FALLO");
        std::printf("CSV,beta1,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,,,%s\n",
                    n, p, wsize, ttmax[T_SCATTER], ttmax[T_GOSSIP], ttmax[T_BCAST],
                    ttmax[T_SORT], ttmax[T_RANK], ttmax[T_REDUCE],
                    tcomp, tcomm, tparmax, tpost, ok ? "OK" : "FALLO");
    }

    // [NUEVO] MPI_Comm_free: libera los comunicadores creados con MPI_Comm_split.
    MPI_Comm_free(&row); MPI_Comm_free(&col); MPI_Comm_free(&grid);
    MPI_Finalize();
    return 0;
}
