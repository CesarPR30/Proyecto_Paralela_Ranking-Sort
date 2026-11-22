/*************************************************
* Proyecto 4 - Ranking Sort (ordenamiento eficiente por ranking)
* CS4052 Computacion Paralela y Distribuida - UTEC 2026-II
* Autores:     O. Chu, C. Rodriguez, C. Pajuelo
* objetivo:    Version SECUENCIAL del Ranking Sort clasico (Enumeration Sort):
*                rank[i] = #{ j : A[j] < A[i] } + #{ j : A[j] = A[i] y j < i }
*                B[rank[i]] = A[i]
*              n^2 comparaciones. Es la referencia T_s^RS del speedup relativo
*              (ec. Srel del informe). Usa el mismo generador y semilla que las
*              versiones MPI, por lo que ordena exactamente los mismos datos.
* compilacion: g++ -O3 -std=c++17 -o rs_sec ranking_sort_sec.cpp
* ejecucion:   ./rs_sec 50000 42
*              (argumentos: n (numero de claves), semilla)
* contenido:   codigo fuente en C++
 *************************************************/
#include <algorithm>
#include <chrono>      // temporizador (ver comentario abajo)
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    const int n = (argc > 1) ? std::atoi(argv[1]) : 50000;
    const unsigned seed = (argc > 2) ? (unsigned)std::atoi(argv[2]) : 42u;
    if (n < 1) { std::fprintf(stderr, "n debe ser >= 1\n"); return 1; }

    // Mismos datos que las versiones MPI: claves en [0, n/2]
    std::vector<int> A(n), rank(n), B(n);
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(0, n / 2);
    for (int t = 0; t < n; ++t) A[t] = dist(gen);

    // [NUEVO] std::chrono::steady_clock: este programa no usa MPI ni OpenMP, asi que no
    // dispone de MPI_Wtime ni de omp_get_wtime (los temporizadores vistos en clase).
    // steady_clock es el reloj monotono estandar de C++ y mide lo mismo (tiempo de pared).
    const auto t0 = std::chrono::steady_clock::now();

    // Fase I: ranking (matriz de comparaciones n x n, recorrida fila por fila)
    for (int i = 0; i < n; ++i) {
        int c = 0;
        const int ai = A[i];
        for (int j = 0; j < n; ++j)
            c += (A[j] < ai) | ((A[j] == ai) & (j < i));
        rank[i] = c;
    }
    // Fase II: colocacion
    for (int i = 0; i < n; ++i) B[rank[i]] = A[i];

    const auto t1 = std::chrono::steady_clock::now();
    const double ts = std::chrono::duration<double>(t1 - t0).count();

    // Verificacion contra std::stable_sort (no cronometrada)
    std::vector<int> ref(A);
    std::stable_sort(ref.begin(), ref.end());
    const bool ok = (B == ref);

    std::printf("Secuencial (Ranking Sort clasico)  n=%d\n", n);
    std::printf("  T_s = %.6f s   comparaciones = %lld\n", ts, (long long)n * n);
    std::printf("  Verificacion: %s\n", ok ? "OK" : "FALLO");
    std::printf("CSV,secuencial,%d,1,1,,,,,,,%.6f,,%.6f,,%lld,%lld,%s\n",
                n, ts, ts, (long long)n * n, (long long)n * n, ok ? "OK" : "FALLO");
    return 0;
}
