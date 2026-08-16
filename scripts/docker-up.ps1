[CmdletBinding()]
param(
    [switch]$Cpu,
    [switch]$NoBuild,
    [switch]$Detached,
    [switch]$DryRun,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ComposeArgs
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

function Test-NvidiaGpu {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $nvidiaSmi) {
        return $false
    }

    & $nvidiaSmi.Source -L *> $null
    return $LASTEXITCODE -eq 0
}

$useGpu = (-not $Cpu) -and (Test-NvidiaGpu)
$dockerArgs = @("compose", "-f", "docker-compose.yml")

if ($useGpu) {
    if (-not $env:LLAMA_CPP_IMAGE) {
        $env:LLAMA_CPP_IMAGE = "ghcr.io/ggml-org/llama.cpp:server-cuda"
    }
    if (-not $env:LLAMA_N_GPU_LAYERS) {
        $env:LLAMA_N_GPU_LAYERS = "999"
    }

    $dockerArgs += @("-f", "docker-compose.gpu.yml")
    Write-Host "NVIDIA GPU detected. Using $env:LLAMA_CPP_IMAGE with LLAMA_N_GPU_LAYERS=$env:LLAMA_N_GPU_LAYERS."
} else {
    if (-not $env:LLAMA_N_GPU_LAYERS) {
        $env:LLAMA_N_GPU_LAYERS = "0"
    }

    if ($Cpu) {
        Write-Host "CPU mode forced. Using llama.cpp CPU image."
    } else {
        Write-Host "No NVIDIA GPU detected. Using llama.cpp CPU image."
    }
}

$dockerArgs += "up"
if (-not $NoBuild) {
    $dockerArgs += "--build"
}
if ($Detached) {
    $dockerArgs += "-d"
}
if ($ComposeArgs) {
    $dockerArgs += $ComposeArgs
}

if ($DryRun) {
    Write-Host ("docker " + ($dockerArgs -join " "))
    exit 0
}

& docker @dockerArgs
exit $LASTEXITCODE
