# setup_anti_spoof_models.ps1
# Script to download official MiniFASNet models for Anti-Spoofing

$ModelsDir = Join-Path $PSScriptRoot "Silent-Face-Anti-Spoofing-master/resources/anti_spoof_models"
if (!(Test-Path $ModelsDir)) {
    New-Item -ItemType Directory -Force -Path $ModelsDir
}

$BaseUrl = "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models"
$Models = @(
    "1.0_120x120_MiniFASNetV2.pth",
    "2.7_80x80_MiniFASNetV2.pth",
    "4_80x80_MiniFASNetV1SE.pth"
)

Write-Host "--- Downloading Anti-Spoofing Models ---" -ForegroundColor Cyan
foreach ($Model in $Models) {
    $DestFile = Join-Path $ModelsDir $Model
    $Url = "$BaseUrl/$Model"
    
    if (Test-Path $DestFile) {
        Write-Host "[SKIP] $Model already exists." -ForegroundColor Gray
    } else {
        Write-Host "[SYNC] Downloading $Model..." -ForegroundColor Yellow
        try {
            Invoke-WebRequest -Uri $Url -OutFile $DestFile -TimeoutSec 60
            Write-Host "[DONE] Successfully downloaded $Model." -ForegroundColor Green
        } catch {
            Write-Error "Failed to download $Model from $Url"
        }
    }
}

Write-Host "`nAll models are ready. Please restart the FaceAttendance AI." -ForegroundColor Cyan
