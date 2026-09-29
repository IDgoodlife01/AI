# 데이터 준비 - Discord에서 받은 데이터의 폴더 구조와 학습 가능 여부 확인
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$rawRoot = Join-Path $projectRoot 'raw'
$required = @(
    (Join-Path $rawRoot 'dfire\train\images'),
    (Join-Path $rawRoot 'dfire\train\labels'),
    (Join-Path $rawRoot 'dfire\test\images'),
    (Join-Path $rawRoot 'dfire\test\labels'),
    (Join-Path $rawRoot 'aihub_clean\images'),
    (Join-Path $rawRoot 'aihub_clean\labels')
)
$missing = @($required | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Container) })
if ($missing.Count -gt 0) {
    Write-Host 'Discord 데이터 압축을 저장소의 raw 폴더에 풀어주세요.' -ForegroundColor Yellow
    Write-Host '필요한 폴더:'
    $missing | ForEach-Object { Write-Host "  $_" }
    exit 1
}
$dfireImages = @(Get-ChildItem -LiteralPath (Join-Path $rawRoot 'dfire') -File -Recurse -Include *.jpg,*.jpeg,*.png).Count
$aihubImages = @(Get-ChildItem -LiteralPath (Join-Path $rawRoot 'aihub_clean\images') -File -Recurse -Include *.jpg,*.jpeg,*.png).Count
$aihubLabels = @(Get-ChildItem -LiteralPath (Join-Path $rawRoot 'aihub_clean\labels') -File -Recurse -Filter *.txt).Count
if ($dfireImages -eq 0 -or $aihubImages -eq 0 -or $aihubImages -ne $aihubLabels) {
    throw "데이터 검사 실패: D-Fire=$dfireImages, AI Hub 이미지=$aihubImages, AI Hub 라벨=$aihubLabels"
}
Write-Host "데이터 준비 완료: D-Fire 이미지 $dfireImages, AI Hub 이미지/라벨 ${aihubImages}쌍" -ForegroundColor Green
Write-Host '학습 실행:'
Write-Host '  & .\ai_model\.venv\Scripts\python.exe .\ai_model\src\train_all.py --output .\ai_model\results_v2'
