import time
from multiprocessing.managers import BaseManager
import numpy as np

# Define the same manager class as in the sender
class QueueManager(BaseManager):
    pass

def main():
    # Register the queue getter with the manager
    QueueManager.register('get_queue')

    # Connect to the manager server
    manager = QueueManager(address=('', 50000), authkey=b'abc')
    manager.connect()
    print("Connected to server")

    # Get the shared memory queue
    queue = manager.get_queue()

    # In a loop, get data from the queue
    frame_count = 0
    start_time = time.time()
    latencies = []
    while True:
        try:
            data = queue.get()
            latency = time.time() - data['timestamp']
            latencies.append(latency)
            frame_count += 1

            if frame_count % 100 == 0:
                end_time = time.time()
                fps = frame_count / (end_time - start_time)
                avg_latency = np.mean(latencies) * 1000
                print(f"Receiver FPS: {fps:.2f}, Avg Latency: {avg_latency:.2f} ms")
                frame_count = 0
                latencies = []
                start_time = time.time()

        except Exception as e:
            # print(f"Queue is empty, waiting.")
            time.sleep(0.001)


if __name__ == '__main__':
    main()