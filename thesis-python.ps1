# Run from PowerShell: .\thesis-python.ps1 -m pytest tests -q
# The shared venv's editable installation is left unchanged.
$ErrorActionPreference = 'Stop'
$thesisRoot = $PSScriptRoot
$sharedPython = Join-Path (Split-Path -Parent $thesisRoot) 'venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $sharedPython -PathType Leaf)) {
    throw "Shared Python interpreter not found: $sharedPython"
}

$savedEnvironment = @{}
$thesisEnvironment = @{
    PYTHONPATH = ((Join-Path $thesisRoot 'src'), $thesisRoot -join [IO.Path]::PathSeparator)
    PYTHONDONTWRITEBYTECODE = '1'
    MPLCONFIGDIR = (Join-Path $thesisRoot 'tmp\matplotlib')
    NUMBA_CACHE_DIR = (Join-Path $thesisRoot 'tmp\numba')
}
$thesisExitCode = 1
Push-Location -LiteralPath $thesisRoot
try {
    foreach ($name in $thesisEnvironment.Keys) {
        $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
        [Environment]::SetEnvironmentVariable($name, $thesisEnvironment[$name], 'Process')
    }
    New-Item -ItemType Directory -Force -Path $thesisEnvironment.MPLCONFIGDIR, $thesisEnvironment.NUMBA_CACHE_DIR | Out-Null
    & $sharedPython -B @args
    $thesisExitCode = $LASTEXITCODE
}
finally {
    foreach ($name in $savedEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], 'Process')
    }
    Pop-Location
}
exit $thesisExitCode
