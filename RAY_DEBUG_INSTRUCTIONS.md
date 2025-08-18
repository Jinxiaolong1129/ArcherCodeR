# Ray Distributed Debugger 使用指南

## 🎯 重要提醒
**Ray Distributed Debugger 不使用传统的 VSCode launch.json 调试方式！**

## 📋 前提条件
1. ✅ 已安装 Ray Distributed Debugger VSCode 扩展
2. ✅ Ray >= 2.9.1
3. ✅ debugpy >= 1.8.0

## 🚀 调试步骤

### 1. 从命令行启动调试
```bash
# 不要使用 VSCode 的 F5 或 launch.json
# 直接在终端运行调试脚本，它会自动修改并运行您的训练脚本：
python debug_ray_dapo.py
```

### 2. 脚本工作原理
调试脚本会：
- 读取您的 `scripts/train/run_archer_qwen2.5_1.5b_code_single.sh` 脚本
- 自动添加 `export RAY_DEBUG_POST_MORTEM=1` 环境变量
- 修改实验名称为 `Archer-Qwen2.5-1.5B-Single-Debug`
- 将训练轮数改为 1 轮（便于调试）
- 运行修改后的脚本

### 3. 观察输出
程序启动后，会看到类似信息：
```
🚀 Creating Ray DAPO debugging script...
📍 This will run the training with Ray debugging enabled
Ray cluster started at: http://127.0.0.1:8265
Task paused, waiting for debugger to attach...
```

### 4. 配置 Ray Distributed Debugger 扩展
1. 在 VSCode 左侧边栏找到 Ray Distributed Debugger 图标
2. 点击 "Add Cluster"  
3. 输入 Ray dashboard URL：`http://127.0.0.1:8265`
4. 设置 Local Folder 为项目根目录：`/workspace/ArcherCodeR`

### 5. 连接调试器
1. 如果程序在 `# breakpoint()` 处暂停，Ray Distributed Debugger 面板会显示暂停的任务
2. 点击暂停的任务来连接 VSCode 调试器
3. 现在可以使用 VSCode 的所有调试功能：
   - 查看变量
   - 单步执行
   - 设置更多断点
   - 查看调用栈

### 6. 多个断点调试
如果有多个 `# breakpoint()` 调用：
1. 每次断点暂停后，先断开当前调试会话
2. 再次点击 Ray Distributed Debugger 扩展图标
3. 连接到新的暂停任务

## 🔧 断点位置要求
- ⚠️ 断点 `# breakpoint()` **只能** 放在被 `@ray.remote` 装饰的函数内
- ✅ 当前断点位置：`TaskRunner.run()` 方法（被 `@ray.remote` 装饰）

## 📝 调试配置
环境变量已自动设置：
```bash
export RAY_DEBUG_POST_MORTEM=1  # 启用 post-mortem 调试
```

## 🛠️ 故障排除
1. **断点不触发**：确保断点在 `@ray.remote` 函数内
2. **扩展不显示任务**：检查 Ray dashboard URL 是否正确 
3. **无法连接**：确保从命令行启动，不要使用 launch.json
4. **脚本找不到**：确保 `scripts/train/run_archer_qwen2.5_1.5b_code_single.sh` 存在

## 💡 添加自定义断点
如果您想在其他地方添加断点，请在被 `@ray.remote` 装饰的函数中添加 `# breakpoint()` 调用。例如：

```python
@ray.remote
def some_ray_function():
    # 您的代码
    # breakpoint()  # 这里会暂停等待调试器
    # 更多代码
```

## 📚 参考资料
- [Ray Distributed Debugger 官方教程](https://verl.readthedocs.io/en/latest/start/ray_debug_tutorial.html) 