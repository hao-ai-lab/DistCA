#!/bin/bash
# Launch the Memory Frontier Explorer webapp
#
# Usage:
#   ./run_memory_frontier.sh
#
# Or specify a custom port:
#   ./run_memory_frontier.sh 8502

CURDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$CURDIR"

PORT=${1:-8501}

echo "🧠 Memory Frontier Explorer"
echo "================================"
echo "Starting webapp on port $PORT..."
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "⚠️  Streamlit not found. Installing dependencies..."
    pip install -r requirements_webapp.txt
fi

# Run the app
streamlit run memory_frontier_app.py --server.port $PORT --server.headless true

