#!/bin/bash
# Refresh market prices and update DuckDNS IP.
# Called every 30 minutes by launchd.

cd /Users/lelan/Desktop/claude_code/prediction-markets

# Refresh market prices + resolution status
poetry run python scripts/refresh_dashboard.py >> logs/refresh.log 2>&1

# Update DuckDNS (in case IP changed)
curl -s "https://www.duckdns.org/update?domains=predictionmarkets&token=7ce0e130-86b3-4d15-a2f7-629f63108e82" >> logs/duckdns.log 2>&1

echo "$(date): refresh complete" >> logs/refresh.log
