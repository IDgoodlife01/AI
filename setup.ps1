# 프로젝트 설치 - 가상환경과 의존성을 준비하고 Discord 데이터 구조 확인
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$venvRoot = Join-Path $projectRoot 'ai_model\.venv'
if (-not (Test-Path -LiteralPath (Join-Path $venvRoot 'Scripts\python.exe'))) {
    py -3.13 -m venv $venvRoot
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.13 가상환경 생성 실패' }
}
$python = Join-Path $venvRoot 'Scripts\python.exe'
& $python -m pip install -r (Join-Path $projectRoot 'ai_model\requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'AI 모델 의존성 설치 실패' }
& $python -m pip install -r (Join-Path $projectRoot 'ai_server\requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'AI 서버 의존성 설치 실패' }
& (Join-Path $projectRoot 'setup_data.ps1')
if ($LASTEXITCODE -ne 0) { throw '데이터 구조 검사 실패' }
Write-Host '설치 완료. 서버 실행: .\ai_model\.venv\Scripts\python.exe -m ai_server.run' -ForegroundColor Green
