# Proyecto 4 — Ranking Sort en paralelo (MPI)

CS4052 Computación Paralela y Distribuida — UTEC, 2026-II. Profesor: José Fiestas.

Integrantes: Oscar Ryan Chu Lao Orrego, Camila Rosa Rodriguez Valverde, Cesar Pajuelo Reyes.

Implementación en C++/MPI del algoritmo de ordenamiento por ranking del enunciado, en una
grilla de `p = q×q` procesos:

```
Input (Scatter) → Gossip (Allgather por columna) → Broadcast (por fila, desde la diagonal)
→ Sort local → Local ranking (búsqueda binaria) → Reduce (suma por fila) → Output (Gather)
```

## Contenido

| Archivo | Descripción |
|---|---|
| `codigo/ranking_sort_beta1.cpp` | **Beta 1**: pipeline completo con colectivas `v` (`Scatterv`, `Allgatherv`, `Gatherv`) para cualquier `n`. |
| `codigo/ranking_sort_beta2.cpp` | **Beta 2**: misma base; la fase Gossip se elige por argumento: `ag` (Allgatherv) o `mb` (Multi-Broadcast). |
| `codigo/ranking_sort_beta3.cpp` | **Beta 3 (versión final)**: colectivas simples, tipo derivado `MPI_PAR`, relleno con centinelas, tiempos por fase y contador de comparaciones. |
| `codigo/quicksort_paralelo.cpp` | Quicksort paralelo (caso `n > p`) usado como comparación. |
| `codigo/ranking_sort_sec.cpp` | Ranking Sort secuencial clásico O(n²), referencia del speedup relativo. |
| `codigo/ranking_sort_hibrido.cpp` | Extensión híbrida MPI+OpenMP de la Beta 3. |
| `codigo/Makefile`, `codigo/compilar_windows.ps1` | Compilación en Linux (`mpic++`) y en Windows (g++ + MS-MPI). |
| `codigo/pruebas_ligeras.ps1` | Pruebas de corrección (67 casos). |
| `codigo/experimentos.py` | Ejecuta todas las mediciones del informe y escribe `resultados.csv`. |
| `codigo/graficas.py` | Calcula medianas, ajusta el modelo teórico y genera las figuras y `resumen.txt`. |
| `codigo/resultados.csv` | Datos crudos: 74 configuraciones × 5 repeticiones (370 ejecuciones, todas verificadas OK). |
| `codigo/resumen.txt` | Medianas y contraste con el modelo teórico. |
| `figuras/` | Gráficas de la sección de Resultados (PDF y PNG). |

## Compilar y ejecutar

Linux / clúster:

```bash
cd codigo
make                                    # binarios en bin/
mpirun -np 4 ./bin/rs_beta3 2000000 42  # n = 2·10^6, semilla 42
mpirun -np 4 ./bin/rs_beta2 2000000 mb  # Gossip con Multi-Broadcast
mpirun -np 4 ./bin/qs_par 2000000 42
OMP_NUM_THREADS=2 mpirun -np 4 ./bin/rs_hibrido 2000000
./bin/rs_sec 50000
```

Windows (MS-MPI y su SDK instalados, g++ en el PATH):

```powershell
cd codigo
powershell -ExecutionPolicy Bypass -File compilar_windows.ps1
mpiexec -n 4 bin\rs_beta3.exe 2000000 42
powershell -ExecutionPolicy Bypass -File pruebas_ligeras.ps1
```

Todos los programas reciben `n` y una semilla; con la misma semilla generan el mismo arreglo,
de modo que las versiones ordenan exactamente los mismos datos. Cada programa verifica su
salida (permutación, orden, estabilidad y pares originales) e imprime una línea `CSV,...`.

## Reproducir los resultados

```bash
cd codigo
python experimentos.py --reps 5   # unos 17 min en un Ryzen 7 5800X (8 núcleos / 16 hilos)
python graficas.py                # figuras en ../figuras y resumen.txt
```

Columnas de `resultados.csv`:
`tag, version, n, p, procesos, t_scatter, t_gossip, t_bcast, t_sort, t_rank, t_reduce, t_comp, t_comm, t_par, t_post, comp_max, comp_total, estado`
(tiempos en segundos, máximo entre procesos; `comp_*` = comparaciones contadas en el código).

## Comandos fuera de lo visto en clase

El código usa los comandos de los ejemplos del curso, las colectivas listadas en la clase de
patrones de comunicación y los tipos derivados / `MPI_Sendrecv` de la clase de tipos MPI. El
único comando nuevo es `MPI_Comm_split` (con `MPI_Comm_free`, `MPI_UNDEFINED` y
`MPI_COMM_NULL`), necesario para crear los comunicadores de fila, columna y diagonal de la
grilla. Está marcado con `[NUEVO]` en el código y justificado en el informe.
