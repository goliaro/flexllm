#! /usr/bin/env bash
set -euo pipefail

echo "WARNING: This will stop and remove all Docker containers and prune the system."
echo "Please make sure you have saved any important results from the containers to the host."
read -p "Are you sure you want to continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborting."
    exit 1
fi


docker stop $(docker ps -aq) && docker rm $(docker ps -aq)
docker system prune -af