# AI 모델 실행 - 기본 학습이 끝나면 6종 추가 학습과 그래프 생성을 시작
$ErrorActionPreference = 'Stop'
$baseline = Join-Path $PSScriptRoot 'results_v2'
$trainingProcessId = [int](Get-Content -LiteralPath "$baseline\training.pid")
if (Get-Process -Id $trainingProcessId -ErrorAction SilentlyContinue) {
    Wait-Process -Id $trainingProcessId
}
if (-not (Test-Path -LiteralPath "$baseline\metrics.json")) {
    throw 'Baseline training did not complete; optimization was not started.'
}
& (Join-Path $PSScriptRoot '.venv\Scripts\python.exe') -u `
    (Join-Path $PSScriptRoot 'src\optimize_all.py')
if ($LASTEXITCODE -ne 0) { throw "Optimization failed (exit code: $LASTEXITCODE)" }
