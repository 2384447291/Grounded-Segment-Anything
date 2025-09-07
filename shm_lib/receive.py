import os
import sys
import time
import numpy as np
import logging

# Add parent directory to path so we can import shm_lib
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from shm_lib.pubsub_manager import PubSubManager

# Global reference to pubsub for use in handler
pubsub_instance = None

def process_segmentation_request(data):
    """Process segmentation request and send back mask result."""
    global pubsub_instance
    
    # Extract image and prompt info
    rgb_image = data['rgb']
    prompt_encoded = data['prompt']
    
    # Decode prompt
    null_idx = np.where(prompt_encoded == 0)[0]
    if len(null_idx) > 0:
        prompt_encoded = prompt_encoded[:null_idx[0]]
    prompt_text = prompt_encoded.tobytes().decode('utf-8', errors='ignore')
    
    # Process the request (simulate segmentation)
    logging.info(f"Processing request: prompt='{prompt_text}', image_shape={rgb_image.shape}")
    
    # Generate mock mask (in real scenario, this would be actual segmentation)
    mock_mask = np.random.choice([True, False], size=(480, 640), p=[0.1, 0.9])  # 10% pixels are True
    
    # Send result back
    result_data = {'mask': mock_mask}
    success = pubsub_instance.publish("segmentation_masks", result_data)
    
    if success:
        process_segmentation_request.count += 1
        if process_segmentation_request.count % 10 == 0:
            logging.info(f"Processed and sent {process_segmentation_request.count} segmentation results")

# Initialize counter
process_segmentation_request.count = 0

def main():
    global pubsub_instance
    
    # Configuration
    port = 10000
    REQUEST_TOPIC = "segmentation_requests"
    MASK_TOPIC = "segmentation_masks"
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s')
    
    # Create PubSub manager
    pubsub = PubSubManager(port=port, authkey=b'foundationpose')
    pubsub_instance = pubsub  # Store global reference
    
    # Setup as both publisher and subscriber
    pubsub.start(role='both')
    
    # Setup subscriber with topic configurations
    topics_config = {
        REQUEST_TOPIC: {
            'examples': {
                'rgb': np.zeros((480, 640, 3), dtype=np.uint8),
                'prompt': np.zeros(256, dtype=np.uint8)
            },
            'buffer_size': 50
        },
        MASK_TOPIC: {
            'examples': {
                'mask': np.zeros((480, 640), dtype=bool)
            },
            'buffer_size': 50
        }
    }
    
    pubsub.setup_subscriber(topics_config)
    
    # Create topics for publishing
    for topic_name, config in topics_config.items():
        pubsub.create_topic(topic_name, config['examples'], config['buffer_size'])
    
    print("Starting segmentation processor...")
    print("Listening for requests and sending back mask results...")
    
    # Register handler for segmentation requests - PubSub will automatically manage the listener thread!
    pubsub.register_topic_handler(REQUEST_TOPIC, process_segmentation_request, check_interval=0.001)
    
    # Main thread: just wait for KeyboardInterrupt
    try:
        print("Segmentation processor is running. Press Ctrl+C to stop...")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nSegmentation processor shutting down...")
    finally:
        pubsub.stop(role='both')

if __name__ == '__main__':
    main()