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

def encode_text_prompt(text: str, max_length: int = 256) -> np.ndarray:
    """Encode a string into a fixed-size numpy array."""
    encoded = text.encode('utf-8')
    if len(encoded) > max_length:
        raise ValueError("Prompt is too long.")
    
    buffer = np.zeros(max_length, dtype=np.uint8)
    buffer[:len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    return buffer

def generate_request_data():
    """Generate test request data."""
    # Create test image
    rgb_image = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
    
    # Cycle through different prompts
    prompts = [
        "a yellow and black utility knife",
        "red apple", 
        "blue car",
        "green tree"
    ]
    prompt_text = prompts[generate_request_data.counter % len(prompts)]
    prompt_encoded = encode_text_prompt(prompt_text)
    
    generate_request_data.counter += 1
    
    return {
        'rgb': rgb_image,
        'prompt': prompt_encoded
    }

# Initialize counter
generate_request_data.counter = 0

def print_stats(frame_count, start_time, success):
    """Custom stats callback for high-speed testing."""
    if success:
        # Print real-time rate every 10 items
        if frame_count % 10 == 0:
            current_time = time.time()
            fps = frame_count / (current_time - start_time)
            print(f"Rate: {fps:.1f}/s", end='\r', flush=True)
        
        # Print detailed stats every 100 items
        if frame_count % 100 == 0:
            current_time = time.time()
            fps = frame_count / (current_time - start_time)
            print(f"\nPublished {frame_count} items, rate: {fps:.2f}/s")

def handle_mask_result(data):
    """Handle received mask results."""
    mask = data['mask']
    handle_mask_result.count += 1
    
    if handle_mask_result.count % 10 == 0:
        non_zero_pixels = np.sum(mask)
        print(f"Received mask #{handle_mask_result.count}, non-zero pixels: {non_zero_pixels}")

# Initialize counter
handle_mask_result.count = 0

def main():
    # Configuration
    port = 10000
    REQUEST_TOPIC = "segmentation_requests"
    MASK_TOPIC = "segmentation_masks"
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s')
    
    # Create PubSub manager
    pubsub = PubSubManager(port=port, authkey=b'foundationpose')
    
    # Setup as both publisher and subscriber
    pubsub.start(role='both')
    
    # Configure topics for both sending and receiving
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
    
    print("Starting bidirectional communication...")
    print("Sending requests and listening for mask results...")
    
    # Main thread: send requests and get mask results
    try:
        frame_count = 0
        start_time = time.time()
        
        while True:
            # Generate and send request
            request_data = generate_request_data()
            success = pubsub.publish(REQUEST_TOPIC, request_data)
            
            if success:
                frame_count += 1
                print_stats(frame_count, start_time, success)
            
            # Get latest mask result every 1 second
            mask_data = pubsub.get_latest_data(MASK_TOPIC)
            if mask_data is not None:
                handle_mask_result(mask_data)
            
            time.sleep(1.0)  # Check every 1 second
            
    except KeyboardInterrupt:
        print("\nSender shutting down...")
    finally:
        pubsub.stop(role='both')

if __name__ == '__main__':
    main()