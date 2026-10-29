<#
    Windows task runner - the PowerShell equivalent of the Makefile.

    Usage:
        .\tasks.ps1            # list tasks
        .\tasks.ps1 install
        .\tasks.ps1 train
        .\tasks.ps1 api
        .\tasks.ps1 test
        .\tasks.ps1 smoke
        .\tasks.ps1 load
#>

param(
    [Parameter(Position = 0)]
    [string]$Task = "help",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

$ErrorActionPreference = "Stop"
$python = if ($env:PYTHON) { $env:PYTHON } else { "python" }

function Show-Help {
    Write-Host "SMS Spam Detector tasks"
    Write-Host ""
    Write-Host "  install      install runtime + dev dependencies"
    Write-Host "  train        download data and train the model"
    Write-Host "  api          run the FastAPI service on :8000"
    Write-Host "  ui           run the Streamlit dashboard on :8501"
    Write-Host "  test         run the test suite"
    Write-Host "  lint         run ruff"
    Write-Host "  smoke        smoke test a running API"
    Write-Host "  load         concurrent load test"
    Write-Host "  docker-build build the production image"
}

switch ($Task) {
    "install"      { & $python -m pip install -r requirements-dev.txt }
    "train"        { & $python main.py train @Rest }
    "api"          { & $python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload @Rest }
    "ui"           { & $python -m streamlit run app.py @Rest }
    "test"         { & $python -m pytest -q @Rest }
    "lint"         { & $python -m ruff check . @Rest }
    "smoke"        { & $python scripts/smoke_test.py @Rest }
    "load"         { if (-not $Rest) { & $python scripts/load_test.py --requests 300 --concurrency 20 } else { & $python scripts/load_test.py @Rest } }
    "docker-build" { docker build -t sms-spam-detector:latest . }
    "docker-up"    { docker compose up --build -d }
    "docker-down"  { docker compose down }
    default        { Show-Help }
}
