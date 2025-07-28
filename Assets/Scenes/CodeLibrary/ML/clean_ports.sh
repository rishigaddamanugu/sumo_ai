#!/bin/bash

PORTS=(5001)

for PORT in "${PORTS[@]}"
do
  echo "🔍 Checking port $PORT..."
  PIDS=$(lsof -ti :$PORT)

  if [ -z "$PIDS" ]; then
    echo "✅ Port $PORT is already free."
  else
    echo "⚠️  Found processes using port $PORT: $PIDS"
    kill $PIDS
    sleep 1
    STILL_RUNNING=$(lsof -ti :$PORT)
    if [ -n "$STILL_RUNNING" ]; then
      echo "❗ Force killing remaining processes on port $PORT..."
      kill -9 $STILL_RUNNING
    fi
    echo "✅ Port $PORT is now free."
  fi
done

echo "🚀 All done. You can now launch Unity safely."
