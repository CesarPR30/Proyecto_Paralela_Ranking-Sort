/*************************************************
* Proyecto 4 - Ranking Sort (ordenamiento eficiente por ranking)
* CS4052 Computacion Paralela y Distribuida - UTEC 2026-II
* Autores:     O. Chu, C. Rodriguez, C. Pajuelo
* objetivo:    BETA 3 (VERSION FINAL). Pipeline de seis fases en grilla q x q:
*              Input -> Gossip -> Broadcast -> Sort -> Local ranking -> Reduce
*              con colectivas simples (Scatter, Allgather, Gather), relleno con
*              centinelas cuando p no divide a n, temporizadores por fase y
*              contador de comparaciones.
* compilacion: mpic++ -O3 -std=c++17 -o rs_beta3 ranking_sort_beta3.cpp
* ejecucion:   mpirun -np 4 ./rs_beta3 2000000 42
*              (argumentos: n (numero de claves), semilla)
* contenido:   codigo fuente en MPI-C++
*
* Convencion de comentarios:
*   [NUEVO]  comando que no aparece en las clases ni en el codigo del curso;
*            se explica por que se usa.
*   (U6.2)   colectiva listada en la clase U6.2 "Patrones de comunicacion en MPI"
*            pero que no aparece en los codigos de ejemplo del curso.
*   (U3.4)   tipo derivado / comunicacion vista en la clase U3.4 "Tipos MPI".
 *************************************************/
#include <mpi.h>
#include <algorithm>   // std::sort, std::lower_bound
#include <climits>     // INT_MAX (clave centinela)
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>      // generador de claves reproducible (mt19937)
#include <vector>

// Cada elemento viaja con su indice original: (A[t], t)
struct Par { int key; int idx; };

// Contador de comparaciones del proceso (para la grafica "operaciones vs p")
static long long ncomp = 0;

// Orden lexicografico (ec. prec del informe): primero la clave, luego el indice.
// Es la regla de desempate que hace al algoritmo estable.
static bool prec(const Par& a, const Par& b) {
    ++ncomp;
    return a.key < b.key || (a.key == b.key && a.idx < b.idx);
}

// Indices de los tiempos por fase
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

    // ---------------- Grilla q x q ----------------
    // q = floor(sqrt(p)); los p - q^2 procesos sobrantes quedan ociosos.
    int q = (int)std::sqrt((double)wsize);
    while ((q + 1) * (q + 1) <= wsize) ++q;      // corrige redondeo de sqrt
    while (q * q > wsize) --q;
    const int p = q * q;

    // [NUEVO] MPI_Comm_split / MPI_UNDEFINED: en clase todas las colectivas usan
    // MPI_COMM_WORLD. El algoritmo necesita colectivas sobre SUBGRUPOS de procesos
    // (solo la columna j en el Gossip, solo la fila i en Broadcast y Reduce, solo la
    // diagonal en el Gather). MPI_Comm_split(comm, color, clave, &nuevo) crea un
    // comunicador por cada valor de 'color'; 'clave' fija el rango dentro de el.
    // Con color = MPI_UNDEFINED el proceso no entra a ningun comunicador (ociosos).
    MPI_Comm grid;
    MPI_Comm_split(MPI_COMM_WORLD, wrank < p ? 0 : MPI_UNDEFINED, wrank, &grid);
    if (wrank >= p) { MPI_Finalize(); return 0; }

    const int k = wrank, i = k / q, j = k % q;   // P_{i,j}: fila i, columna j
    MPI_Comm row, col, diag;
    MPI_Comm_split(grid, i, j, &row);            // fila i    (rango en row == j)
    MPI_Comm_split(grid, j, i, &col);            // columna j (rango en col == i)
    MPI_Comm_split(grid, i == j ? 0 : MPI_UNDEFINED, i, &diag);  // diagonal (q procesos)

    // (U3.4) Tipo derivado contiguo: un Par = 2 enteros consecutivos.
    // Permite escribir los conteos de las colectivas en "pares" y no en enteros.
    MPI_Datatype MPI_PAR;
    MPI_Type_contiguous(2, MPI_INT, &MPI_PAR);
    MPI_Type_commit(&MPI_PAR);

    // n' = p*ceil(n/p): tamanos uniformes para Scatter / Allgather / Gather
    const int np = ((n + p - 1) / p) * p;
    const int m  = np / p;                       // n'/p        pares por proceso
    const int w  = np / q;                       // n'/sqrt(p)  pares por bloque

    // Solo el proceso 0 genera A (fuera del tiempo medido).
    // Claves en [0, n/2] para forzar repeticiones y ejercitar el desempate.
    std::vector<Par> A, C(m), S(w), R(w);
    if (k == 0) {
        A.resize(np);
        std::mt19937 gen(seed);
        std::uniform_int_distribution<int> dist(0, n / 2);
        for (int t = 0; t < n;  ++t) A[t] = { dist(gen), t };
        for (int t = n; t < np; ++t) A[t] = { INT_MAX, t };   // centinelas (+inf, t)
    }

    double tt[NT] = {0};
    MPI_Barrier(grid);                           // (U6.2) todos parten juntos
    const double tini = MPI_Wtime();
    double t = tini;

    // ---- Fase 0: Input. Scatter de n'/p pares a cada proceso ----
    MPI_Scatter(A.data(), m, MPI_PAR, C.data(), m, MPI_PAR, 0, grid);
    tt[T_SCATTER] = MPI_Wtime() - t; t = MPI_Wtime();

    // ---- Fase 1: Gossip. All-gather en la columna j  ->  S = B_j ----
    MPI_Allgather(C.data(), m, MPI_PAR, S.data(), m, MPI_PAR, col);   // (U6.2)
    tt[T_GOSSIP] = MPI_Wtime() - t; t = MPI_Wtime();

    // ---- Fase 2: Broadcast. La diagonal P_ii difunde B_i a su fila  ->  R = B_i ----
    if (i == j) R = S;                           // en la diagonal, S ya es B_i
    MPI_Bcast(R.data(), w, MPI_PAR, i, row);     // raiz: rango i de la fila = P_ii
    tt[T_BCAST] = MPI_Wtime() - t; t = MPI_Wtime();

    // ---- Fase 3: Sort local de B_j bajo prec ----
    // Se usa std::sort (introsort, O(x log x)) en lugar del qsort de la clase U5.2:
    // misma idea (ordenamiento por comparaciones con funcion comparadora), pero el
    // comparador se pasa por plantilla y el compilador lo puede expandir en linea.
    const long long c0 = ncomp;
    std::sort(S.begin(), S.end(), prec);
    tt[T_SORT] = MPI_Wtime() - t; t = MPI_Wtime();

    // ---- Fase 4: Local ranking. r[t] = #{ y en B_j : y prec R[t] } ----
    // std::lower_bound es la busqueda binaria (U5.3): devuelve la posicion del primer
    // elemento de S que NO es menor que R[t], es decir, cuantos son menores.
    std::vector<int> r(w), rk(w);
    for (int u = 0; u < w; ++u)
        r[u] = (int)(std::lower_bound(S.begin(), S.end(), R[u], prec) - S.begin());
    tt[T_RANK] = MPI_Wtime() - t; t = MPI_Wtime();
    const long long comps = ncomp - c0;          // comparaciones de las fases 3 y 4

    // ---- Fase 5: Reduce. Suma por fila hacia la diagonal ----
    MPI_Reduce(r.data(), rk.data(), w, MPI_INT, MPI_SUM, i, row);     // (U6.2)
    tt[T_REDUCE] = MPI_Wtime() - t;
    const double tpar = MPI_Wtime() - tini;      // T_par: Scatter ... Reduce

    // ---- Fase 6: Output (post-procesamiento, fuera de T_par) ----
    // Las q diagonales envian (rank, clave, indice) a P_00 por el comunicador diag.
    // El indice original solo se envia para la verificacion.
    double tpost = 0.0;
    std::vector<int> all;
    std::vector<Par> B;
    if (i == j) {
        const double tp0 = MPI_Wtime();
        std::vector<int> out(3 * w);
        for (int u = 0; u < w; ++u) {
            out[3*u] = rk[u]; out[3*u+1] = R[u].key; out[3*u+2] = R[u].idx;
        }
        if (k == 0) all.resize(3 * np);
        MPI_Gather(out.data(), 3 * w, MPI_INT, all.data(), 3 * w, MPI_INT, 0, diag);
        if (k == 0) {                            // B[rank] <- clave; descarta centinelas
            B.assign(n, Par{0, -1});
            for (int u = 0; u < np; ++u)
                if (all[3*u] < n) B[all[3*u]] = { all[3*u+1], all[3*u+2] };
        }
        tpost = MPI_Wtime() - tp0;
    }

    // ---- Verificacion en el proceso 0 (no se cronometra) ----
    // (i) rank es permutacion de 0..n-1, (ii) B no decreciente, (iii) estable
    // (iv) cada B[t] es exactamente el par original (A[idx], idx).
    bool ok = true;
    if (k == 0) {
        std::vector<char> visto(n, 0);
        int reales = 0;
        for (int u = 0; u < np && ok; ++u) {
            const int pos = all[3*u], idx = all[3*u+2];
            if (idx < n) {                       // elemento real
                if (pos < 0 || pos >= n || visto[pos]) ok = false;
                else { visto[pos] = 1; ++reales; }
            } else if (pos < n) ok = false;      // un centinela no puede caer en 0..n-1
        }
        if (reales != n) ok = false;
        for (int u = 1; u < n && ok; ++u) if (!prec(B[u-1], B[u])) ok = false;
        for (int u = 0; u < n && ok; ++u)
            if (B[u].idx < 0 || B[u].idx >= n || B[u].key != A[B[u].idx].key) ok = false;
    }

    // ---- Tiempos maximos entre procesos y conteo de operaciones ----
    double ttmax[NT], tparmax;
    long long cmax, csum;
    MPI_Reduce(tt, ttmax, NT, MPI_DOUBLE, MPI_MAX, 0, grid);
    MPI_Reduce(&tpar, &tparmax, 1, MPI_DOUBLE, MPI_MAX, 0, grid);
    // MPI_LONG_LONG: tipo basico de MPI (tabla de tipos basicos, U3.4); los conteos
    // de comparaciones superan el rango de int para n grande.
    MPI_Reduce(&comps, &cmax, 1, MPI_LONG_LONG, MPI_MAX, 0, grid);
    MPI_Reduce(&comps, &csum, 1, MPI_LONG_LONG, MPI_SUM, 0, grid);

    if (k == 0) {
        const double tcomp = ttmax[T_SORT] + ttmax[T_RANK];
        const double tcomm = ttmax[T_SCATTER] + ttmax[T_GOSSIP] + ttmax[T_BCAST] + ttmax[T_REDUCE];
        std::printf("Beta3  n=%d  n'=%d  procesos=%d  grilla=%dx%d (p=%d, ociosos=%d)\n",
                    n, np, wsize, q, q, p, wsize - p);
        std::printf("  Scatter   = %.6f s\n  Gossip    = %.6f s\n  Broadcast = %.6f s\n",
                    ttmax[T_SCATTER], ttmax[T_GOSSIP], ttmax[T_BCAST]);
        std::printf("  Sort      = %.6f s\n  Ranking   = %.6f s\n  Reduce    = %.6f s\n",
                    ttmax[T_SORT], ttmax[T_RANK], ttmax[T_REDUCE]);
        std::printf("  T_comp=%.6f s  T_comm=%.6f s  T_par=%.6f s  T_post=%.6f s\n",
                    tcomp, tcomm, tparmax, tpost);
        std::printf("  Comparaciones: max/proceso=%lld  total(W)=%lld\n", cmax, csum);
        std::printf("  Verificacion: %s\n", ok ? "OK" : "FALLO");
        if (n <= 20) {
            std::printf("  A = ");
            for (int u = 0; u < n; ++u) std::printf("%d ", A[u].key);
            std::printf("\n  B = ");
            for (int u = 0; u < n; ++u) std::printf("%d ", B[u].key);
            std::printf("\n");
        }
        // Linea CSV para juntar las ejecuciones del informe
        std::printf("CSV,beta3,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%lld,%lld,%s\n",
                    n, p, wsize, ttmax[T_SCATTER], ttmax[T_GOSSIP], ttmax[T_BCAST],
                    ttmax[T_SORT], ttmax[T_RANK], ttmax[T_REDUCE],
                    tcomp, tcomm, tparmax, tpost, cmax, csum, ok ? "OK" : "FALLO");
    }

    MPI_Type_free(&MPI_PAR);
    // [NUEVO] MPI_Comm_free: libera los comunicadores creados con MPI_Comm_split.
    // [NUEVO] MPI_COMM_NULL: valor que deja MPI_Comm_split en los procesos que
    //         pidieron color MPI_UNDEFINED (aqui, los no diagonales en diag).
    if (diag != MPI_COMM_NULL) MPI_Comm_free(&diag);
    MPI_Comm_free(&row); MPI_Comm_free(&col); MPI_Comm_free(&grid);
    MPI_Finalize();
    return 0;
}
