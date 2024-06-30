#!/bin/sh
# Ensure .credentials directory exists
mkdir -p /home/appuser/.credentials
# Copy credentials.json to the expected directory
cp .credentials/credentials.json /home/appuser/.credentials/credentials.json
