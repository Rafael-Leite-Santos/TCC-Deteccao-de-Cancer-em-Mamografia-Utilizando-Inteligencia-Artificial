# FCOS e ATSS com o pacote de aumento de dados (18/11/2026), 3 sementes cada (6 treinos).
# Ordem intercalada: FCOS s0, ATSS s0, FCOS s1, ATSS s1, ... (pode parar no meio com Ctrl+C).
# Uso (da raiz do clone, com o venv ativo):  .\rodar_fcos_atss_aug.ps1
if (-not (Test-Path 'D:\Dataset_VinDR-Mammo')) { Write-Host 'HD D: desconectado. Abortando.'; exit 1 }
$livre = [math]::Round((Get-PSDrive C).Free / 1GB, 1)
if ($livre -lt 20) { Write-Host "Disco C: com so $livre GB livres. Libere espaco antes (MemoryError)."; exit 1 }
$configs = @(
  'configs/sementes/fcosaug_s0.py', 'configs/sementes/atssaug_s0.py',
  'configs/sementes/fcosaug_s1.py', 'configs/sementes/atssaug_s1.py',
  'configs/sementes/fcosaug_s2.py', 'configs/sementes/atssaug_s2.py'
)
foreach ($c in $configs) {
  Write-Host "===== $(Get-Date -Format 'yyyy-MM-dd HH:mm')  $c ====="
  python VinDR_train.py $c
}
Write-Host "===== FIM $(Get-Date -Format 'yyyy-MM-dd HH:mm') ====="
