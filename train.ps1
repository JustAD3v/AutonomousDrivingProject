#!/usr/bin/env pwsh
# Shortcut for running the training inside the docker compose container.
Write-Host "python src/main.py --container $($args -join ' ')"
docker compose run --rm train python src/main.py --container @args