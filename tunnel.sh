#!/bin/bash
# Public link for HEADLESS servers only (local users: just open localhost:8000).
# Downloads cloudflared on first use, then prints a *.trycloudflare.com URL.
# Keep the link private — it exposes progress, samples, and LoRA downloads.
cd "$(dirname "$0")"
if [ ! -x ./cloudflared ]; then
  echo "fetching cloudflared..."
  curl -sL -o cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x cloudflared
fi
./cloudflared tunnel --url http://localhost:8000
