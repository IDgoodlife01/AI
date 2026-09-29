# AI 모델 최종 학습 - 중단된 기본 학습을 복원하고 6종 최적화와 그래프 생성을 연속 실행
$ErrorActionPreference = 'Stop'
$modelRoot = $PSScriptRoot
$python = Join-Path $modelRoot '.venv\Scripts\python.exe'
$baseline = Join-Path $modelRoot 'results_v2'
$optimized = Join-Path $modelRoot 'results_optimized'
$trainScript = Join-Path $modelRoot 'src\train_all.py'
$optimizeScript = Join-Path $modelRoot 'src\optimize_all.py'

& $python -u $trainScript `
    --output $baseline `
    --image-size 160 `
    --epochs 10 `
    --hard-negative-epochs 2 `
    --batch-size 64 `
    --resume `
    --completed-deep cnn `
    1> (Join-Path $baseline 'final_training.log') `
    2> (Join-Path $baseline 'final_training.error.log')
if ($LASTEXITCODE -ne 0) { throw "기본 6종 학습 실패 (종료 코드: $LASTEXITCODE)" }
if (-not (Test-Path -LiteralPath (Join-Path $baseline 'metrics.json'))) {
    throw '기본 학습 결과 metrics.json이 생성되지 않았습니다.'
}

& $python -u $optimizeScript `
    --source $baseline `
    --output $optimized `
    --epochs 15 `
    --batch-size 32 `
    1> (Join-Path $optimized 'final_training.log') `
    2> (Join-Path $optimized 'final_training.error.log')
if ($LASTEXITCODE -ne 0) { throw "최종 6종 최적화 실패 (종료 코드: $LASTEXITCODE)" }
if (-not (Test-Path -LiteralPath (Join-Path $optimized 'metrics.json'))) {
    throw '최종 결과 metrics.json이 생성되지 않았습니다.'
}
Write-Host '최신 6종 학습·최적화·그래프 생성 완료' -ForegroundColor Green
