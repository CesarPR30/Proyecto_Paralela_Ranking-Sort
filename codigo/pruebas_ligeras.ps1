# Pruebas ligeras de correccion (Windows, MS-MPI). No son las mediciones del informe.
#   powershell -ExecutionPolicy Bypass -File pruebas_ligeras.ps1
# Ejecuta cada programa con varios n y numeros de procesos y revisa que la
# verificacion interna (permutacion, orden, estabilidad, pares originales) diga OK.
Set-Location $PSScriptRoot
& .\compilar_windows.ps1
if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne $null) { exit 1 }

$casos = @()
# n divisible y no divisible entre p, n < p, el ejemplo del enunciado (n=18, p=9)
foreach ($np in 1, 4, 6, 9) {
    foreach ($n in 18, 1000, 100003) {
        $casos += ,@($np, "bin\rs_beta1.exe", "$n")
        $casos += ,@($np, "bin\rs_beta2.exe", "$n ag")
        $casos += ,@($np, "bin\rs_beta2.exe", "$n mb")
        $casos += ,@($np, "bin\rs_beta3.exe", "$n")
    }
}
foreach ($np in 1, 2, 4, 8) { foreach ($n in 3, 1000, 100003) { $casos += ,@($np, "bin\qs_par.exe", "$n") } }
foreach ($np in 1, 4) { foreach ($n in 5, 100003) { $casos += ,@($np, "bin\rs_hibrido.exe", "$n") } }

$fallos = 0
$env:OMP_NUM_THREADS = "2"
foreach ($c in $casos) {
    $np, $exe, $argstr = $c
    $out = & mpiexec -n $np $exe $argstr.Split(" ") 2>&1 | Out-String
    $csv = ($out -split "`n" | Where-Object { $_ -like "CSV,*" }) -join ""
    $estado = if ($csv -match ",OK\s*$") { "OK" } else { $fallos++; "FALLO" }
    "{0,-6} np={1,-2} {2,-22} {3}" -f $estado, $np, $exe, $argstr
}
foreach ($n in 7, 2000, 20001) {
    $out = & bin\rs_sec.exe $n | Out-String
    $estado = if ($out -match "Verificacion: OK") { "OK" } else { $fallos++; "FALLO" }
    "{0,-6} np=1  bin\rs_sec.exe         {1}" -f $estado, $n
}
Write-Host ""
Write-Host "Casos con fallo: $fallos"
exit $fallos
