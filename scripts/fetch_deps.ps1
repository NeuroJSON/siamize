# Fetch C++ build dependencies into third_party/ on native Windows.
#   - ONNX Runtime prebuilt (Windows x64 zip)
#   - nifti_clib (git clone)
# Re-runnable; skips already-present trees.
#
# Note: CI uses scripts/fetch_deps.sh under Git Bash on all platforms,
# including Windows. This .ps1 is here as a convenience for native Windows
# users who don't have a POSIX shell installed.

$ErrorActionPreference = "Stop"

$OrtVersion = if ($env:ORT_VERSION) { $env:ORT_VERSION } else { "1.26.0" }
$OrtArch    = "win-x64"
$OrtUrl     = "https://github.com/microsoft/onnxruntime/releases/download/v$OrtVersion/onnxruntime-$OrtArch-$OrtVersion.zip"

$Here = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$TP   = Join-Path $Here "third_party"
New-Item -Path $TP -ItemType Directory -Force | Out-Null

$OrtDir = Join-Path $TP "onnxruntime"
if (Test-Path $OrtDir) {
    Write-Host "[fetch_deps] onnxruntime already present, skipping"
} else {
    Write-Host "[fetch_deps] downloading ORT v$OrtVersion ($OrtArch) ..."
    $tmp = Join-Path $env:TEMP "siamize-ort-$([guid]::NewGuid())"
    New-Item -Path $tmp -ItemType Directory -Force | Out-Null
    $zip = Join-Path $tmp "ort.zip"
    Invoke-WebRequest -Uri $OrtUrl -OutFile $zip
    Expand-Archive -Path $zip -DestinationPath $tmp -Force
    $extracted = Join-Path $tmp "onnxruntime-$OrtArch-$OrtVersion"
    Move-Item -Path $extracted -Destination $OrtDir
    Remove-Item -Path $tmp -Recurse -Force
    Write-Host "[fetch_deps] ORT installed at $OrtDir"
}

Write-Host "[fetch_deps] done."
Write-Host "  ORT:  $OrtDir"
Write-Host "  zmat: src\zmat\zmat.h (bundled, dual-licensed Apache-2.0)"
