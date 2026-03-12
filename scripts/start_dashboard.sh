#!/bin/bash
# Start the Streamlit dashboard and keep it running.
# Usage: ./scripts/start_dashboard.sh

cd "$(dirname "$0")/.."

# Kill existing streamlit if running
pkill -f "streamlit run app.py" 2>/dev/null || true
sleep 1

# Start streamlit
poetry run streamlit run app.py \
    --server.port 8501 \
    --server.headless true \
    --server.address 0.0.0.0 \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    >> logs/streamlit.log 2>&1 &

echo "Dashboard started on http://0.0.0.0:8501"
echo "PID: $!"
