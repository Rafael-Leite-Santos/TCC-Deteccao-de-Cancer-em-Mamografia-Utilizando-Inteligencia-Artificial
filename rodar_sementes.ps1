# Ponto 6 do orientador: 3 configuracoes x 3 sementes (9 treinos, ~11-13 h no total).
# Ordem intercalada (todas as configs com semente 0, depois 1, depois 2): se precisar
# interromper, as tres configuracoes ficam com o mesmo numero de rodadas.
# Uso (da raiz do clone, com o venv ativo):  .\rodar_sementes.ps1
# Para recomecar de onde parou, apague da lista os que ja terminaram (tem best_*.pth e teste\).

if (-not (Test-Path 'D:\Dataset_VinDR-Mammo')) { Write-Host 'HD D: desconectado. Abortando.'; exit 1 }

$configs = @(
  'configs/sementes/r50_s0.py', 'configs/sementes/r18_s0.py', 'configs/sementes/r50rot_s0.py',
  'configs/sementes/r50_s1.py', 'configs/sementes/r18_s1.py', 'configs/sementes/r50rot_s1.py',
  'configs/sementes/r50_s2.py', 'configs/sementes/r18_s2.py', 'configs/sementes/r50rot_s2.py'
)
foreach ($c in $configs) {
  Write-Host "===== $(Get-Date -Format 'yyyy-MM-dd HH:mm')  $c ====="
  python VinDR_train.py $c
}
Write-Host "===== FIM $(Get-Date -Format 'yyyy-MM-dd HH:mm') ====="
