#!/bin/bash

# Clear memory cache (requires sudo)
echo "Clearing memory cache..."
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

# Navigate to project directory
cd /home/mashedpotato/PycharmProjects/PythonProject/ || exit 1

# Activate virtual environment
source .venv/bin/activate

# Start the timer in background
start_time=$(date +%s)

# Dynamic runtime display
(while true; do
    now=$(date +%s)
    elapsed=$((now - start_time))
    printf "\r⏱️  Runtime: %02d:%02d:%02d" $((elapsed/3600)) $(( (elapsed/60)%60 )) $((elapsed%60))
    sleep 1
done) &

# Store background PID
TIMER_PID=$!

# Run the Python script
python uiv2.py

# Kill the timer after script ends
kill $TIMER_PID
echo -e "\n✅ Finished."
