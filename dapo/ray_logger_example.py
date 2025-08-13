#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ray分布式环境中的正确日志记录方法
"""
import ray
import logging

# 方法1: 使用Ray的内置日志
@ray.remote
class RayWorkerWithLogging:
    def __init__(self):
        # 在worker中配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='[Worker %(process)d] %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def process_data(self, data):
        # 使用logger而不是print
        self.logger.info(f"🔥 Processing data: {data}")
        # 这些日志会出现在Ray的日志文件中
        return f"Processed: {data}"

# 方法2: 强制输出到driver进程（性能较差，但可见）
@ray.remote
class RayWorkerWithForceOutput:
    def process_data_with_output(self, data):
        import sys
        # 强制刷新stdout
        print(f"🚀 Worker processing: {data}", flush=True)
        sys.stdout.flush()
        return f"Processed: {data}"

# 方法3: 返回日志信息到主进程
@ray.remote  
class RayWorkerWithReturnLogs:
    def process_data_with_logs(self, data):
        logs = []
        logs.append(f"📦 Started processing: {data}")
        
        # 模拟处理过程
        result = f"Processed: {data}"
        logs.append(f"✅ Completed processing: {result}")
        
        # 返回结果和日志
        return {"result": result, "logs": logs}

# 使用示例
def main():
    ray.init()
    
    print("=" * 50)
    print("🎯 Ray分布式日志示例")
    print("=" * 50)
    
    # 方法1: 使用logging（推荐）
    print("\n📋 方法1: 使用Ray内置日志系统")
    worker1 = RayWorkerWithLogging.remote()
    result1 = ray.get(worker1.process_data.remote("test_data_1"))
    print(f"Driver收到结果: {result1}")
    print("💡 Worker的日志输出在Ray日志文件中，使用: tail -f /tmp/ray/session_latest/logs/worker-*.out")
    
    # 方法2: 强制输出（不推荐，性能差）
    print("\n📋 方法2: 强制输出到driver")
    worker2 = RayWorkerWithForceOutput.remote()
    result2 = ray.get(worker2.process_data_with_output.remote("test_data_2"))
    print(f"Driver收到结果: {result2}")
    
    # 方法3: 返回日志到主进程（推荐用于重要信息）
    print("\n📋 方法3: 返回日志到主进程")
    worker3 = RayWorkerWithReturnLogs.remote()
    result3 = ray.get(worker3.process_data_with_logs.remote("test_data_3"))
    print(f"Worker日志:")
    for log in result3["logs"]:
        print(f"  {log}")
    print(f"结果: {result3['result']}")
    
    ray.shutdown()

if __name__ == "__main__":
    main() 