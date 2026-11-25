#!/usr/bin/env bash
# Shortcut for running the training inside the docker compose container.
echo "python src/main.py --container $*"
docker compose run --rm train python src/main.py --container "$@"
