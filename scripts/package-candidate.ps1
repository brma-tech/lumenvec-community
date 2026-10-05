param([string]$Version = 'v0.3.0-rc.1')
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$out = Join-Path $root "dist/$Version"
New-Item -ItemType Directory -Force -Path $out | Out-Null
$savedOS = $env:GOOS
$savedArch = $env:GOARCH
$savedCGO = $env:CGO_ENABLED
Push-Location $root
try {
    foreach ($platform in 'windows','linux') {
        $env:GOOS = $platform
        $env:GOARCH = 'amd64'
        $env:CGO_ENABLED = '0'
        $name = "lumenvec-community-$Version-$platform-amd64"
        $bundle = Join-Path $out $name
        New-Item -ItemType Directory -Force -Path $bundle | Out-Null
        $binary = if ($platform -eq 'windows') { 'lumenvec.exe' } else { 'lumenvec' }
        go build -mod=readonly -trimpath -o (Join-Path $bundle $binary) ./cmd/server
        if ($LASTEXITCODE) { throw "Build failed for $platform" }
        Copy-Item LICENSE,RELEASE_NOTES.md -Destination $bundle
        Compress-Archive -Path "$bundle/*" -DestinationPath (Join-Path $out "$name.zip") -Force
    }
    Get-ChildItem $out -Filter '*.zip' | ForEach-Object {
        "{0}  {1}" -f (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant(), $_.Name
    } | Set-Content (Join-Path $out 'SHA256SUMS.txt')
} finally {
    $env:GOOS = $savedOS
    $env:GOARCH = $savedArch
    $env:CGO_ENABLED = $savedCGO
    Pop-Location
}
