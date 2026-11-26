# Teste com imagens sem massa no treino (Faster R-CNN + pacote de aumento, 3 sementes, ~9 h).
# Compara com configs/sementes/r50aug_s*. Antes: python VinDR_prepare_neg.py
# Uso (da raiz do clone, com o venv ativo):  .\rodar_neg.ps1
if (-not (Test-Path 'D:\Dataset_VinDR-Mammo')) { Write-Host 'HD D: desconectado. Abortando.'; exit 1 }
if (-not (Test-Path 'annotations_VinDR_neg1\train.json')) { Write-Host 'Rode antes: python VinDR_prepare_neg.py'; exit 1 }
$livre = (Get-PSDrive C).Free / 1GB
if ($livre -lt 20) { Write-Host "C: com $([math]::Round($livre,1)) GB livres (minimo 20). Abortando."; exit 1 }
$configs = @('configs/sementes/r50augneg_s0.py', 'configs/sementes/r50augneg_s1.py', 'configs/sementes/r50augneg_s2.py')
foreach ($c in $configs) {
  Write-Host "===== $(Get-Date -Format 'yyyy-MM-dd HH:mm')  $c ====="
  python VinDR_train.py $c
}
Write-Host "===== FIM $(Get-Date -Format 'yyyy-MM-dd HH:mm') ====="
