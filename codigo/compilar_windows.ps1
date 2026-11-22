# Compilacion en Windows con g++ (MinGW) + Microsoft MPI (MS-MPI + SDK)
#   powershell -ExecutionPolicy Bypass -File compilar_windows.ps1
# Requiere: g++ en el PATH, MS-MPI (mpiexec) y el SDK de MS-MPI instalados.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$sdk = "C:\Program Files (x86)\Microsoft SDKs\MPI"
$inc = "$sdk\Include"
$lib = "$sdk\Lib\x64"
$flags = @("-O3", "-std=c++17", "-Wall")
New-Item -ItemType Directory -Force bin | Out-Null

$progs = @(
    @{ src = "ranking_sort_sec.cpp";     out = "rs_sec";     mpi = $false; omp = $false },
    @{ src = "ranking_sort_beta1.cpp";   out = "rs_beta1";   mpi = $true;  omp = $false },
    @{ src = "ranking_sort_beta2.cpp";   out = "rs_beta2";   mpi = $true;  omp = $false },
    @{ src = "ranking_sort_beta3.cpp";   out = "rs_beta3";   mpi = $true;  omp = $false },
    @{ src = "quicksort_paralelo.cpp";   out = "qs_par";     mpi = $true;  omp = $false },
    @{ src = "ranking_sort_hibrido.cpp"; out = "rs_hibrido"; mpi = $true;  omp = $true  }
)

foreach ($p in $progs) {
    $gargs = @() + $flags
    if ($p.omp) { $gargs += "-fopenmp" }
    $gargs += $p.src
    if ($p.mpi) { $gargs += @("-I$inc", "-L$lib", "-lmsmpi") }
    $gargs += @("-o", "bin\$($p.out).exe")
    Write-Host "g++ $($p.src) -> bin\$($p.out).exe"
    & g++ @gargs
    if ($LASTEXITCODE -ne 0) { throw "Fallo la compilacion de $($p.src)" }
}
Write-Host "Compilacion terminada."
