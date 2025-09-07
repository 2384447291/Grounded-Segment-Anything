#!/usr/bin/env python3
"""
PubSub 模式使用示例

演示消费者模式和广播模式的区别：
- 消费者模式：数据被读取后消失，多个订阅者竞争消费
- 广播模式：数据保留，多个订阅者可以独立读取同一份数据
"""

import os
import sys
import time
import threading
import numpy as np
# Add parent directory to path so we can import shm_lib
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from shm_lib.pubsub_manager import PubSubManager
from shm_lib.shared_memory_util import encode_text_prompt

def consumer_mode_example():
    """演示消费者模式：多个订阅者竞争消费数据"""
    print("\n=== 消费者模式示例 ===")
    
    pubsub = PubSubManager(port=10001, authkey=b'consumer_demo')
    pubsub.start(role='both')
    
    # 创建消费者模式的 topic
    topic_name = "consumer_data"
    examples = {
        'frame_id': np.int32(0),
        'data': np.zeros((3, 3), dtype=np.float32),
        'message': encode_text_prompt('test')
    }
    
    pubsub.create_topic(topic_name, examples, buffer_size=5, mode='consumer')
    
    # 发布一些数据
    for i in range(3):
        data = {
            'frame_id': np.int32(i),
            'data': np.random.random((3, 3)).astype(np.float32),
            'message': encode_text_prompt(f'consumer_message_{i}')
        }
        pubsub.publish(topic_name, data)
        print(f"Published consumer data {i}")
        time.sleep(0.1)
    
    # 模拟两个订阅者竞争读取
    def subscriber(name):
        count = 0
        while count < 2:  # 尝试读取2次
            try:
                data = pubsub.get_latest(topic_name, block=False)
                if data is not None:
                    frame_id = data['frame_id']
                    print(f"  {name} got frame {frame_id}")
                    count += 1
                else:
                    print(f"  {name} got no data")
                    break
            except:
                print(f"  {name} got no data (exception)")
                break
            time.sleep(0.05)
    
    # 启动两个订阅者线程
    thread1 = threading.Thread(target=lambda: subscriber("Subscriber-A"))
    thread2 = threading.Thread(target=lambda: subscriber("Subscriber-B"))
    
    thread1.start()
    thread2.start()
    
    thread1.join()
    thread2.join()
    
    pubsub.stop(role='both')
    print("消费者模式：数据被消费后不再可用，订阅者之间竞争数据\n")


def broadcast_mode_example():
    """演示广播模式：多个订阅者可以独立读取同一份数据"""
    print("=== 广播模式示例 ===")
    
    pubsub = PubSubManager(port=10002, authkey=b'broadcast_demo')
    pubsub.start(role='both')
    
    # 创建广播模式的 topic
    topic_name = "broadcast_data"
    examples = {
        'frame_id': np.int32(0),
        'data': np.zeros((2, 2), dtype=np.float32),
        'message': encode_text_prompt('test')
    }
    
    pubsub.create_topic(topic_name, examples, buffer_size=5, mode='broadcast')
    
    # 发布一些数据
    for i in range(3):
        data = {
            'frame_id': np.int32(i),
            'data': np.random.random((2, 2)).astype(np.float32),
            'message': encode_text_prompt(f'broadcast_message_{i}')
        }
        pubsub.publish(topic_name, data)
        print(f"Published broadcast data {i}")
        time.sleep(0.1)
    
    # 模拟两个订阅者独立读取
    def subscriber(name):
        count = 0
        while count < 3:  # 尝试读取3次
            try:
                data = pubsub.get_latest(topic_name, block=False)
                if data is not None:
                    frame_id = data['frame_id']
                    print(f"  {name} got frame {frame_id}")
                    count += 1
                else:
                    print(f"  {name} got no data")
                    break
            except:
                print(f"  {name} got no data (exception)")
                break
            time.sleep(0.05)
    
    # 启动两个订阅者线程
    thread1 = threading.Thread(target=lambda: subscriber("Subscriber-X"))
    thread2 = threading.Thread(target=lambda: subscriber("Subscriber-Y"))
    
    thread1.start()
    thread2.start()
    
    thread1.join()
    thread2.join()
    
    pubsub.stop(role='both')
    print("广播模式：数据保留，多个订阅者可以独立读取最新数据\n")


def mixed_mode_example():
    """演示在同一个 PubSubManager 中使用不同模式的 topic"""
    print("=== 混合模式示例 ===")
    
    pubsub = PubSubManager(port=10003, authkey=b'mixed_demo')
    pubsub.start(role='both')
    
    # 设置配置
    topics_config = {
        'commands': {
            'examples': {
                'command': encode_text_prompt('start'),
                'timestamp': np.float64(0.0)
            },
            'buffer_size': 10,
            'mode': 'consumer'  # 命令使用消费者模式
        },
        'status': {
            'examples': {
                'status': encode_text_prompt('running'),
                'frame_count': np.int32(0),
                'timestamp': np.float64(0.0)
            },
            'buffer_size': 5,
            'mode': 'broadcast'  # 状态使用广播模式
        }
    }
    
    pubsub.setup_subscriber(topics_config)
    
    # 发布一些命令和状态
    pubsub.publish('commands', {
        'command': encode_text_prompt('start'),
        'timestamp': np.float64(time.time())
    })
    
    pubsub.publish('status', {
        'status': encode_text_prompt('running'),
        'frame_count': np.int32(42),
        'timestamp': np.float64(time.time())
    })
    
    print("Published command and status")
    time.sleep(0.1)
    
    # 读取数据
    cmd_data = pubsub.get_latest_data('commands')
    status_data1 = pubsub.get_latest_data('status')
    status_data2 = pubsub.get_latest_data('status')  # 再读一次状态
    
    print(f"Command data: {cmd_data is not None}")
    print(f"Status data (1st read): {status_data1 is not None}")
    print(f"Status data (2nd read): {status_data2 is not None}")
    
    # 再次尝试读取命令（应该已被消费）
    cmd_data2 = pubsub.get_latest_data('commands')
    print(f"Command data (2nd read): {cmd_data2 is not None}")
    
    pubsub.stop(role='both')
    print("混合模式：不同 topic 可以使用不同的数据共享模式\n")


if __name__ == '__main__':
    print("PubSub 模式对比演示")
    print("=" * 50)
    
    try:
        consumer_mode_example()
        time.sleep(1)
        
        broadcast_mode_example()
        time.sleep(1)
        
        mixed_mode_example()
        
    except KeyboardInterrupt:
        print("\n演示被用户中断")
    except Exception as e:
        print(f"\n演示出现错误: {e}")
    
    print("演示完成！")
