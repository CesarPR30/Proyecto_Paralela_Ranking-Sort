/*************************************************
* Proyecto 4 - Ranking Sort: algoritmo de comparacion
* CS4052 Computacion Paralela y Distribuida - UTEC 2026-II
* Autores:     O. Chu, C. Rodriguez, C. Pajuelo
* objetivo:    QUICKSORT EN PARALELO (caso n > p, clase U5.2) para comparar con
*              el Ranking Sort. Esquema de hipercubo:
*                1. Scatter: n/p elementos por proceso y sort local O((n/p)log(n/p)).
*                2. Mientras el grupo tenga mas de un proceso:
*                   - se elige un pivote comun (mediana de las medianas locales),
*                   - cada proceso separa sus elementos en  < pivote  y  >= pivote,
*                   - la mitad inferior del grupo se queda con los menores y la
*                     superior con los mayores (intercambio con la pareja r <-> r+P/2),
*                   - mezcla de las dos secuencias ordenadas y el grupo se parte en dos.
*                3. Gatherv: la concatenacion por rango ya es el arreglo ordenado.
*              Usa el mismo generador, semilla, tipo Par y comparador que el Ranking
*              Sort, de modo que ordena exactamente los mismos datos.
*              Requiere p potencia de 2: se usa la mayor potencia de 2 <= procesos.
* compilacion: mpic++ -O3 -std=c++17 -o qs_par quicksort_paralelo.cpp
* ejecucion:   mpirun -np 4 ./qs_par 2000000 42
*              (argumentos: n (numero de claves), semilla)
* contenido:   codigo fuente en MPI-C++
*
* Convencion de comentarios:
*   [NUEVO]  comando que no aparece en las clases ni en el codigo del curso.
*   (U6.2)   colectiva listada en la clase U6.2 "Patrones de comunicacion en MPI".
*   (U3.4)   visto en la clase U3.4 "Tipos MPI" (Type_contiguous, Sendrecv).
 *************************************************/
#include <mpi.h>
#include <algorithm>   // std::sort, std::lower_bound, std::merge
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

struct Par { int key; int idx; };

static bool prec(const Par& a, const Par& b) {
    return a.key < b.key || (a.key == b.key && a.idx < b.idx);
}

enum { T_SCATTER, T_SORT, T_EXCH, T_MERGE, NT };

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

    // p = mayor potencia de 2 <= procesos; el resto queda ocioso
    int p = 1;
    while (2 * p <= wsize) p *= 2;

    // [NUEVO] MPI_Comm_split / MPI_UNDEFINED: separa los p procesos activos y, en cada
    // nivel de la recursion, parte el grupo en dos mitades (ver mas abajo).
    MPI_Comm act;
    MPI_Comm_split(MPI_COMM_WORLD, wrank < p ? 0 : MPI_UNDEFINED, wrank, &act);
    if (wrank >= p) { MPI_Finalize(); return 0; }
    const int k = wrank;

    // (U3.4) Tipo derivado: un Par = 2 enteros contiguos
    MPI_Datatype MPI_PAR;
    MPI_Type_contiguous(2, MPI_INT, &MPI_PAR);
    MPI_Type_commit(&MPI_PAR);

    // Relleno con centinelas igual que en la Beta 3 (Scatter exige tamanos iguales)
    const int np = ((n + p - 1) / p) * p;
    const int m  = np / p;
    std::vector<Par> A, D(m);
    if (k == 0) {
        A.resize(np);
        std::mt19937 gen(seed);
        std::uniform_int_distribution<int> dist(0, n / 2);
        for (int t = 0; t < n;  ++t) A[t] = { dist(gen), t };
        for (int t = n; t < np; ++t) A[t] = { INT_MAX, t };
    }

    double tt[NT] = {0};
    MPI_Barrier(act);                            // (U6.2)
    const double tini = MPI_Wtime();
    double t = tini;

    // ---- 1. Input y sort local ----
    MPI_Scatter(A.data(), m, MPI_PAR, D.data(), m, MPI_PAR, 0, act);
    tt[T_SCATTER] = MPI_Wtime() - t; t = MPI_Wtime();
    std::sort(D.begin(), D.end(), prec);
    tt[T_SORT] = MPI_Wtime() - t;

    // ---- 2. Particion recursiva del grupo (log2 p niveles) ----
    MPI_Comm comm = act;
    int s;
    MPI_Comm_size(comm, &s);
    while (s > 1) {
        int r;
        MPI_Comm_rank(comm, &r);
        t = MPI_Wtime();

        // Pivote: mediana de las medianas locales. Cada proceso aporta su mediana y su
        // cantidad de elementos; todos calculan el mismo pivote.
        int cnt = (int)D.size();
        Par med = cnt > 0 ? D[cnt / 2] : Par{INT_MAX, INT_MAX};
        std::vector<Par> meds(s);
        std::vector<int> cnts(s);
        MPI_Allgather(&med, 1, MPI_PAR, meds.data(), 1, MPI_PAR, comm);   // (U6.2)
        MPI_Allgather(&cnt, 1, MPI_INT, cnts.data(), 1, MPI_INT, comm);
        std::vector<Par> val;
        for (int u = 0; u < s; ++u) if (cnts[u] > 0) val.push_back(meds[u]);
        Par piv = Par{INT_MAX, INT_MAX};
        if (!val.empty()) {
            std::sort(val.begin(), val.end(), prec);
            piv = val[val.size() / 2];
        }

        // Separacion local:  D[0..corte) < pivote <= D[corte..cnt)
        const int corte = (int)(std::lower_bound(D.begin(), D.end(), piv, prec) - D.begin());
        const int mitad = s / 2;
        const bool bajo = r < mitad;
        const int pareja = bajo ? r + mitad : r - mitad;
        // La mitad inferior envia sus mayores y conserva sus menores; la superior al reves
        const Par* env  = bajo ? D.data() + corte : D.data();
        const int nenv  = bajo ? cnt - corte : corte;
        const int ini_k = bajo ? 0 : corte;
        const int nkeep = bajo ? corte : cnt - corte;

        // (U3.4) MPI_Sendrecv: envio y recepcion simultaneos con la pareja (evita el
        // bloqueo mutuo que tendria un Send seguido de Recv en ambos procesos).
        int nrec = 0;
        MPI_Sendrecv(&nenv, 1, MPI_INT, pareja, 0, &nrec, 1, MPI_INT, pareja, 0,
                     comm, MPI_STATUS_IGNORE);
        std::vector<Par> rec(nrec);
        MPI_Sendrecv(env, nenv, MPI_PAR, pareja, 1, rec.data(), nrec, MPI_PAR, pareja, 1,
                     comm, MPI_STATUS_IGNORE);
        tt[T_EXCH] += MPI_Wtime() - t; t = MPI_Wtime();

        // Mezcla de lo conservado y lo recibido (ambos ordenados): O(n/p) por nivel
        std::vector<Par> nuevo(nkeep + nrec);
        std::merge(D.begin() + ini_k, D.begin() + ini_k + nkeep, rec.begin(), rec.end(),
                   nuevo.begin(), prec);
        D.swap(nuevo);
        tt[T_MERGE] += MPI_Wtime() - t; t = MPI_Wtime();

        // [NUEVO] MPI_Comm_split: el grupo se parte en dos sub-grupos independientes
        // (color 0 = mitad inferior, color 1 = mitad superior) que siguen la recursion.
        MPI_Comm sub;
        MPI_Comm_split(comm, bajo ? 0 : 1, r, &sub);
        if (comm != act) MPI_Comm_free(&comm);   // [NUEVO] libera el comunicador anterior
        comm = sub;
        MPI_Comm_size(comm, &s);
        tt[T_EXCH] += MPI_Wtime() - t;
    }
    if (comm != act) MPI_Comm_free(&comm);
    const double tpar = MPI_Wtime() - tini;

    // ---- 3. Output: Gatherv en orden de rango (post-procesamiento) ----
    const double tp0 = MPI_Wtime();
    int cnt = (int)D.size();
    std::vector<int> gcnt(p), gdsp(p);
    MPI_Gather(&cnt, 1, MPI_INT, gcnt.data(), 1, MPI_INT, 0, act);
    std::vector<Par> B;
    if (k == 0) {
        for (int u = 0, off = 0; u < p; ++u) { gdsp[u] = off; off += gcnt[u]; }
        B.resize(np);
    }
    MPI_Gatherv(D.data(), cnt, MPI_PAR, B.data(), gcnt.data(), gdsp.data(), MPI_PAR, 0, act);  // (U6.2)
    if (k == 0) B.resize(n);                     // los centinelas quedan al final
    const double tpost = MPI_Wtime() - tp0;

    // Verificacion (no cronometrada): orden + estabilidad + mismos pares que A
    bool ok = true;
    if (k == 0) {
        std::vector<char> visto(n, 0);
        for (int u = 0; u < n && ok; ++u) {
            const int idx = B[u].idx;
            if (idx < 0 || idx >= n || visto[idx] || B[u].key != A[idx].key) ok = false;
            else visto[idx] = 1;
        }
        for (int u = 1; u < n && ok; ++u) if (!prec(B[u-1], B[u])) ok = false;
    }

    // Desbalance final: max y min de elementos por proceso
    int cmax, cmin;
    MPI_Reduce(&cnt, &cmax, 1, MPI_INT, MPI_MAX, 0, act);
    MPI_Reduce(&cnt, &cmin, 1, MPI_INT, MPI_MIN, 0, act);
    double ttmax[NT], tparmax;
    MPI_Reduce(tt, ttmax, NT, MPI_DOUBLE, MPI_MAX, 0, act);
    MPI_Reduce(&tpar, &tparmax, 1, MPI_DOUBLE, MPI_MAX, 0, act);

    if (k == 0) {
        const double tcomp = ttmax[T_SORT] + ttmax[T_MERGE];
        const double tcomm = ttmax[T_SCATTER] + ttmax[T_EXCH];
        std::printf("Quicksort paralelo  n=%d  procesos=%d  activos=%d (ociosos=%d)\n",
                    n, wsize, p, wsize - p);
        std::printf("  Scatter=%.6f  SortLocal=%.6f  Intercambio=%.6f  Mezcla=%.6f\n",
                    ttmax[T_SCATTER], ttmax[T_SORT], ttmax[T_EXCH], ttmax[T_MERGE]);
        std::printf("  T_comp=%.6f s  T_comm=%.6f s  T_par=%.6f s  T_post=%.6f s\n",
                    tcomp, tcomm, tparmax, tpost);
        std::printf("  Elementos por proceso al final: min=%d max=%d (ideal %d)\n", cmin, cmax, m);
        std::printf("  Verificacion: %s\n", ok ? "OK" : "FALLO");
        std::printf("CSV,quicksort,%d,%d,%d,%.6f,,,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,,,%s\n",
                    n, p, wsize, ttmax[T_SCATTER], ttmax[T_SORT], ttmax[T_MERGE], ttmax[T_EXCH],
                    tcomp, tcomm, tparmax, tpost, ok ? "OK" : "FALLO");
    }

    MPI_Type_free(&MPI_PAR);
    MPI_Comm_free(&act);
    MPI_Finalize();
    return 0;
}
