import time
import numpy as np
from multiprocessing.managers import SharedMemoryManager, BaseManager
from shm_lib import SharedMemoryQueue
import multiprocessing as mp

# Define a manager class to host the shared memory queue
class QueueManager(BaseManager):
    pass

def main():
    # Start a shared memory manager
    with SharedMemoryManager() as shm_manager:
        # Create an example piece of data to determine the queue's structure
        example = {
            'rgb': np.zeros((480, 640, 3), dtype=np.uint8),
            'timestamp': 0.0
        }
        
        # Create the shared memory queue
        queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=10
        )

        # Register the queue with our custom manager
        QueueManager.register('get_queue', callable=lambda: queue)
        
        # Start the manager server
        manager = QueueManager(address=('', 50000), authkey=b'abc')
        server = manager.get_server()
        
        # Start the server in a separate process
        server_process = mp.Process(target=server.serve_forever)
        server_process.daemon = True
        server_process.start()
        
        print("Server started at port 50000")
        
        # In a loop, put data into the queue
        frame_count = 0
        start_time = time.time()
        try:
            while True:
                data = {
                    'rgb': np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8),
                    'timestamp': time.time()
                }
                try:
                    queue.put(data)
                    frame_count += 1
                    if frame_count % 100 == 0:
                        end_time = time.time()
                        fps = frame_count / (end_time - start_time)
                        print(f"Sender FPS: {fps:.2f}")
                        frame_count = 0
                        start_time = time.time()

                except Exception as e:
                    # print(f"Queue is full, skipping.")
                    time.sleep(0.001) # wait a bit if queue is full
                
        except KeyboardInterrupt:
            print("Shutting down...")
        finally:
            server_process.terminate()


if __name__ == '__main__':
    main()