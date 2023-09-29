!#/bin/env bash

echo "Inspecting tensor at index $1"
echo "🦀 ========== "
cd ~/gg-rs && cargo run $1
echo "🐍 ========== "
cd ~/tinygrad && source venv/bin/activate && python gg.py $1
