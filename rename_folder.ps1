# Rename codes/ to code/
if (Test-Path "codes" -PathType Container) {
    Rename-Item -Path "codes" -NewName "code"
    Write-Host "✅ Renamed codes/ -> code/"
} else {
    Write-Host "ℹ️  codes/ folder not found (may already be renamed)"
}
