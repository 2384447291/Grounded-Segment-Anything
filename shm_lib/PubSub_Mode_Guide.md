# PubSub 模式选择指南

本文档介绍了 PubSubManager 中两种数据共享模式的区别、使用场景和最佳实践。

## 概述

PubSubManager 支持两种数据共享模式：
- **消费者模式 (Consumer Mode)**: 数据被读取后消失，适用于任务队列
- **广播模式 (Broadcast Mode)**: 数据保留，多个读者可独立读取，适用于状态广播

## 消费者模式 (Consumer Mode)

### 核心特征
- **底层实现**: `SharedMemoryQueue` (FIFO队列)
- **读取行为**: 数据被读取后消失，不可重复读取
- **多读者**: 读者之间**竞争消费**同一份数据
- **数据顺序**: 严格按照 FIFO (先进先出) 顺序

### 数据流示例
```python
# 发布数据
publish(cmd1) → [cmd1]
publish(cmd2) → [cmd1, cmd2]
publish(cmd3) → [cmd1, cmd2, cmd3]

# 消费数据 (多个读者竞争)
reader_A.get() → cmd1  # [cmd2, cmd3] 
reader_B.get() → cmd2  # [cmd3]
reader_A.get() → cmd3  # []
reader_B.get() → Empty # 没有数据了
```

### Buffer 满时行为
```python
队列状态: [cmd1, cmd2, cmd3, cmd4, cmd5] (buffer_size=5, 已满)
新命令到达: cmd6
处理方式: 移除最老的 cmd1 → [cmd2, cmd3, cmd4, cmd5, cmd6]
```
- ✅ 保持 FIFO 顺序
- ✅ 只丢弃最老的一个数据
- ✅ 不会因新数据丢失所有旧数据

### 适用场景
- **命令处理系统**: 确保每个命令只被执行一次
- **任务分发**: 多个工作者竞争处理任务
- **请求/响应**: 分割请求、控制命令等
- **事件处理**: 需要严格顺序处理的事件

### 配置示例
```python
topics_config = {
    'commands': {
        'examples': {
            'command': encode_text_prompt('start'),
            'timestamp': np.float64(0.0)
        },
        'buffer_size': 10,
        'mode': 'consumer'  # 消费者模式
    }
}
```

## 广播模式 (Broadcast Mode)

### 核心特征
- **底层实现**: `SharedMemoryRingBuffer` (环形缓冲区)
- **读取行为**: 数据保留，可重复读取
- **多读者**: 读者**独立读取**同一份最新数据
- **数据顺序**: 总是获取最新数据

### 数据流示例
```python
# 发布数据
publish(pose1) → [pose1, _, _]
publish(pose2) → [pose1, pose2, _]
publish(pose3) → [pose1, pose2, pose3]

# 读取数据 (多个读者独立)
reader_X.get() → pose3  # 最新数据
reader_Y.get() → pose3  # 同样的最新数据
reader_X.get() → pose3  # 还是最新数据
```

### Buffer 满时行为
```python
缓冲区状态: [pose1, pose2, pose3] (buffer_size=3, 已满)
新数据到达: pose4
处理方式: 覆盖最老位置 → [pose4, pose2, pose3]
继续发布: pose5 → [pose4, pose5, pose3]
继续发布: pose6 → [pose4, pose5, pose6]

读取时: 总是获取最新的 pose6
```
- ✅ 自动覆盖最老数据
- ✅ 保持最新的连续数据序列
- ✅ 支持 `get_last_k()` 获取历史数据

### 适用场景
- **状态广播**: 位姿数据、传感器状态等
- **实时监控**: 多个客户端需要相同的最新状态
- **可视化**: 多个显示器显示同一数据
- **数据同步**: 需要保持数据一致性的场景

### 配置示例
```python
topics_config = {
    'poses': {
        'examples': {
            'pose_matrix': np.eye(4, dtype=np.float32),
            'timestamp': np.float64(0.0),
            'frame_idx': np.int32(0)
        },
        'buffer_size': 50,
        'mode': 'broadcast'  # 广播模式
    }
}
```

## 模式对比总结

| 特性 | 消费者模式 | 广播模式 |
|------|------------|----------|
| **底层结构** | SharedMemoryQueue | SharedMemoryRingBuffer |
| **读取语义** | 消费式 (读一个少一个) | 广播式 (可重复读) |
| **数据顺序** | FIFO (先进先出) | LIFO (总是最新) |
| **多读者行为** | 竞争消费 | 独立读取 |
| **数据保持** | 读后消失 | 持续保留 |
| **Buffer满时** | 移除最老的1个 | 覆盖最老数据 |
| **历史数据** | 不支持 | 支持 `get_last_k()` |
| **适用场景** | 任务队列、命令处理 | 状态广播、实时监控 |

## 使用指南

### 选择原则

**选择消费者模式当:**
- 每个数据项需要且仅需要被处理一次
- 数据有明确的处理顺序要求
- 多个工作者需要分担工作负载
- 数据是"命令"或"任务"性质

**选择广播模式当:**
- 多个客户端需要相同的最新数据
- 数据是"状态"或"信息"性质
- 需要支持多个独立的监控/显示客户端
- 数据的时效性比历史重要

### 混合使用示例

在实际项目中，通常会同时使用两种模式：

```python
# FoundationPose 项目的典型配置
topics_config = {
    # 控制命令 - 消费者模式
    'tracking_control': {
        'examples': {
            'command': encode_text_prompt('start'),
            'object_prompt': np.zeros(256, dtype=np.uint8),
            'object_name': encode_text_prompt('object')
        },
        'buffer_size': 5,
        'mode': 'consumer'
    },
    
    # 位姿数据 - 广播模式
    'tracking_poses': {
        'examples': {
            'pose_matrix': np.eye(4, dtype=np.float32),
            'timestamp': np.float64(0.0),
            'frame_idx': np.int32(0),
            'object_name': encode_text_prompt('object')
        },
        'buffer_size': 50,
        'mode': 'broadcast'
    },
    
    # 分割请求 - 消费者模式
    'segmentation_requests': {
        'examples': {
            'rgb': np.zeros((480, 640, 3), dtype=np.uint8),
            'prompt': np.zeros(256, dtype=np.uint8)
        },
        'buffer_size': 10,
        'mode': 'consumer'
    },
    
    # 分割结果 - 消费者模式  
    'segmentation_masks': {
        'examples': {
            'mask': np.zeros((480, 640), dtype=bool)
        },
        'buffer_size': 10,
        'mode': 'consumer'
    }
}
```

## 性能考虑

### Buffer 大小设置

**消费者模式:**
- 设置为 `生产速度 × 最大处理延迟 × 安全系数`
- 例如: 10Hz 生产，最大 0.5s 处理延迟 → buffer_size = 10 × 0.5 × 2 = 10

**广播模式:**
- 设置为支持的最大历史查询数量
- 例如: 需要最近 1 秒的位姿数据，60Hz → buffer_size = 60

### 内存使用

两种模式的内存使用量相近：
```
内存使用 = buffer_size × 数据大小 × 字段数量
```

### 并发性能

- **消费者模式**: 读者间有竞争，但无数据复制开销
- **广播模式**: 读者间无竞争，每个读者独立复制数据

## 最佳实践

1. **模式选择要明确**: 根据数据语义选择，不要混淆
2. **Buffer 大小要合理**: 避免过大浪费内存，过小丢失数据
3. **错误处理要完善**: 处理 `Empty` 异常和超时情况
4. **监控数据流量**: 定期检查生产/消费速度是否匹配
5. **测试并发场景**: 验证多读者/多写者情况下的行为

## 常见问题

**Q: 为什么消费者模式读到的不是最新数据？**
A: 消费者模式是 FIFO 队列，读取的是最老的未处理数据，这确保了处理顺序的公平性。

**Q: 广播模式会丢失历史数据吗？**
A: 会的。当 buffer 满时，最老的数据会被新数据覆盖。如需持久化，应在应用层实现。

**Q: 可以动态切换模式吗？**
A: 不可以。模式在 topic 创建时确定，需要重新创建 topic 才能改变。

**Q: 哪个模式性能更好？**
A: 两者性能相近。消费者模式在高并发读取时可能有轻微优势，广播模式在多读者场景更适合。
