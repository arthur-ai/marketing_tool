#!/bin/bash

# Marketing Tool AWS Deployment Wrapper Script
# This script runs the deployment from the project root

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$SCRIPT_DIR/deploy"

# Check if deploy directory exists
if [[ ! -d "$DEPLOY_DIR" ]]; then
    echo "Error: Deploy directory not found at $DEPLOY_DIR"
    exit 1
fi

# Check if deploy.sh exists
if [[ ! -f "$DEPLOY_DIR/deploy.sh" ]]; then
    echo "Error: deploy.sh not found in $DEPLOY_DIR"
    exit 1
fi

# Make deploy.sh executable
chmod +x "$DEPLOY_DIR/deploy.sh"

# Run the deployment script with all passed arguments
exec "$DEPLOY_DIR/deploy.sh" "$@"
