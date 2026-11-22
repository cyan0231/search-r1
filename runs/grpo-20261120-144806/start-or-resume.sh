#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
nohup setsid /usr/bin/python3 -u "$PWD/launch.py" >>"$PWD/status.log" 2>&1 </dev/null &
echo "启动状态见 $PWD/status.log"
