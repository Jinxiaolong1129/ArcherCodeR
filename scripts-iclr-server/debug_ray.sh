#!/usr/bin/env bash
set -xeuo pipefail

echo "🔍 Ray Debugging Script"
echo "======================="

echo "📊 System Information:"
echo "CPU cores: $(nproc)"
echo "Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
echo "Disk space: $(df -h / | tail -1 | awk '{print $4}')"

echo ""
echo "🔍 Checking for existing Ray processes:"
ps aux | grep -i ray || echo "No Ray processes found"

echo ""
echo "🔍 Checking Ray temporary directories:"
ls -la /tmp/ray* 2>/dev/null || echo "No Ray temp directories found"
ls -la ~/.ray 2>/dev/null || echo "No ~/.ray directory found"

echo ""
echo "🔍 Network connectivity check:"
netstat -tuln | grep -E ':(6379|10001|8000)' || echo "No Ray ports in use"

echo ""
echo "🧹 Cleaning up Ray resources:"
pkill -f "ray::" || true
pkill -f "raylet" || true
pkill -f "gcs_server" || true
sleep 3

rm -rf /tmp/ray/session_* || true
rm -rf /tmp/ray/sockets/* || true
rm -rf ~/.ray || true

echo ""
echo "🔧 Setting up Ray environment:"
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_DEDUP_LOGS=0
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
export RAY_raylet_start_wait_time_s=60
export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1
export RAY_DISABLE_STRICT_VERSION_CHECK=1

echo ""
echo "🧪 Testing Ray initialization:"
/home/ec2-user/miniconda3/envs/archer/bin/python -c "
import ray
import time
print('Testing Ray initialization...')
try:
    ray.init(num_cpus=4, object_store_memory=1000000000, _temp_dir='/tmp/ray_temp')
    print('✅ Ray initialization successful!')
    print(f'Ray cluster resources: {ray.cluster_resources()}')
    ray.shutdown()
    print('✅ Ray shutdown successful!')
except Exception as e:
    print(f'❌ Ray initialization failed: {e}')
    import traceback
    traceback.print_exc()
"

echo ""
echo "✅ Ray debugging complete!"
