# AI 모델 보고서 - 기본 학습 완료 후 results_v2 비교 그래프와 혼동행렬 생성
$ErrorActionPreference = 'Stop'
$results = Join-Path $PSScriptRoot 'results_v2'
$trainingPid = [int](Get-Content -LiteralPath "$results\training.pid")
Wait-Process -Id $trainingPid
if (-not (Test-Path -LiteralPath "$results\model_comparison.csv")) {
    throw 'Training did not create model_comparison.csv.'
}
& (Join-Path $PSScriptRoot '.venv\Scripts\python.exe') `
    (Join-Path $PSScriptRoot 'src\generate_reports.py') --results $results `
    *> "$results\report_generation.log"
if ($LASTEXITCODE -ne 0) { throw "Report generation failed (exit code: $LASTEXITCODE)" }
