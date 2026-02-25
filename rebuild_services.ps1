param (
    [string]$ComposeFile = "webapp/docker-compose.yml"
)

# Ensure Docker is on PATH for this session
$env:PATH = "C:\Program Files\Docker\Docker\resources\bin;$env:PATH"

# Get services from compose
$services = docker compose -f $ComposeFile config --services

if (-not $services) {
    Write-Error "No services found in $ComposeFile"
    exit 1
}

Write-Host "Available services:`n"

$i = 1
foreach ($s in $services) {
    Write-Host "[$i] $s"
    $i++
}

$choice = Read-Host "`nChoose a service number"
$service = $services[$choice - 1]

if (-not $service) {
    Write-Error "Invalid selection"
    exit 1
}

$nocache = Read-Host "Rebuild without cache? (y/N)"

if ($nocache -match '^[Yy]$') {
    docker compose -f $ComposeFile build --no-cache $service
} else {
    docker compose -f $ComposeFile build $service
}

docker compose -f $ComposeFile up -d --no-deps --force-recreate $service

Write-Host "`n✅ Service '$service' rebuilt and restarted"
