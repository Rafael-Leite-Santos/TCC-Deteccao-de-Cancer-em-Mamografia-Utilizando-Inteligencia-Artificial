# Ponto 3 do orientador: pacote de aumento de dados (flip + rotacao + escala + translacao +
# brilho + contraste), 3 sementes (~4-5 h no total). Compara com configs/sementes/r50rot_s*.
# Uso (da raiz do clone, com o venv ativo):  .\rodar_aug.ps1
if (-not (Test-Path 'D:\Dataset_VinDR-Mammo')) { Write-Host 'HD D: desconectado. Abortando.'; exit 1 }
$configs = @('configs/sementes/r50aug_s0.py', 'configs/sementes/r50aug_s1.py', 'configs/sementes/r50aug_s2.py')
foreach ($c in $configs) {
  Write-Host "===== $(Get-Date -Format 'yyyy-MM-dd HH:mm')  $c ====="
  python VinDR_train.py $c
}
Write-Host "===== FIM $(Get-Date -Format 'yyyy-MM-dd HH:mm') ====="
