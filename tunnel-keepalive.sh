#!/bin/sh
# Auto-restart apex-tunnel on crash
while true; do
  echo "[$(date -u)] Starting tunnel..."
  apex-tunnel expose 8080 llm-demo 2>&1
  echo "[$(date -u)] Tunnel died, restarting in 2s..."
  sleep 2
done
