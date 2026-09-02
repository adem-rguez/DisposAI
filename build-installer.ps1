$ErrorActionPreference = "Stop"

cargo build --release --package daemon-core
if ($LASTEXITCODE -ne 0) { exit 1 }

Copy-Item target\release\daemon-core.exe apps\dispos-studio\resources\daemon-core.exe -Force
Copy-Item scripts apps\dispos-studio\resources\scripts -Recurse -Force

npm install
if ($LASTEXITCODE -ne 0) { exit 1 }

npm run build --prefix apps/dispos-studio
if ($LASTEXITCODE -ne 0) { exit 1 }

Push-Location apps\dispos-studio
npm run dist
if ($LASTEXITCODE -ne 0) { Pop-Location; exit 1 }
Pop-Location

$installer = Get-Item "apps\dispos-studio\dist-installer\DisposAI Setup *.exe" -ErrorAction SilentlyContinue | Select-Object -First 1

if ($installer) {
    Write-Host $installer.FullName
} else {
    Write-Host "ERROR: Installer not found"
    exit 1
}
