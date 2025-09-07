import os
import sys

# Ensure the project root is in the Python path.
# This allows for correct importing of local packages like 'shm_lib'
# and other project modules.
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
from PIL import Image
import logging
import time



# Add project paths - This is no longer needed as the project root is added above
# sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
# sys.path.append(os.path.join(os.getcwd(), "segment_anything"))

# Grounding DINO imports
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict

# Segment Anything imports
from segment_anything import sam_hq_model_registry, SamPredictor

# PubSub Manager
from shm_lib.pubsub_manager import PubSubManager

# --- Helper Functions ---
port = 10000   

def decode_text_prompt(encoded_array: np.ndarray) -> str:
    """Decode a numpy array back to a string."""
    null_idx = np.where(encoded_array == 0)[0]
    if len(null_idx) > 0:
        encoded_array = encoded_array[:null_idx[0]]
    return encoded_array.tobytes().decode('utf-8', errors='ignore')

def prepare_image_for_dino(image_np: np.ndarray, device: str) -> torch.Tensor:
    """Converts a numpy image to a tensor suitable for GroundingDINO."""
    image_pil = Image.fromarray(image_np).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_transformed, _ = transform(image_pil, None)
    return image_transformed.to(device)

def load_grounding_dino_model(model_config_path, model_checkpoint_path, bert_base_uncased_path, device):
    """Loads the GroundingDINO model."""
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    args.bert_base_uncased_path = bert_base_uncased_path
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold, device="cpu"):
    """Runs the GroundingDINO model to get bounding boxes, mirroring test.py logic."""
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."
    
    # Ensure model and image are on the correct device, just like in test.py
    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
        
    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]
    
    # Filter based on box threshold
    filt_mask = logits.max(dim=1)[0] > box_threshold
    boxes_filt = boxes[filt_mask]
    
    # The client doesn't need phrases, so we just return the boxes.
    return boxes_filt

# --- PubSub Topic Names ---
REQUEST_TOPIC = "segmentation_requests"
MASK_TOPIC = "segmentation_masks"

# Global references for use in handler
pubsub_instance = None
dino_model_instance = None
predictor_instance = None
device_instance = None
box_threshold_instance = None

def process_segmentation_request(data):
    """Process segmentation request and send back mask result."""
    global pubsub_instance, dino_model_instance, predictor_instance, device_instance, box_threshold_instance
    
    try:
        # Extract image and prompt info
        rgb_image_np = data['rgb']
        text_prompt = decode_text_prompt(data['prompt'])
        logging.info(f"Processing request with prompt: '{text_prompt}', image_shape={rgb_image_np.shape}")
        
        H, W, _ = rgb_image_np.shape

        # Run GroundingDINO to get boxes
        image_tensor = prepare_image_for_dino(rgb_image_np, device_instance)
        text_threshold = 0.25  # Value from test.py
        boxes = get_grounding_output(
            dino_model_instance, image_tensor, text_prompt, box_threshold_instance, text_threshold, device=device_instance
        )

        if boxes.size(0) == 0:
            logging.warning("No objects detected. Returning an empty mask.")
            empty_mask = np.zeros((H, W), dtype=bool)
            pubsub_instance.publish(MASK_TOPIC, {'mask': empty_mask})
            return
        
        # Prepare boxes for SAM (cxcywh -> xyxy format)
        boxes_xyxy = boxes.clone()
        boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * W
        boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * H
        boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * W
        boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * H
        
        # Run SAM predictor to get masks
        predictor_instance.set_image(rgb_image_np)
        transformed_boxes = predictor_instance.transform.apply_boxes_torch(boxes_xyxy, (H, W)).to(device_instance)

        masks, _, _ = predictor_instance.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        
        # Combine masks and send the result back
        final_mask = torch.any(masks, dim=0).squeeze(0).cpu().numpy()
        success = pubsub_instance.publish(MASK_TOPIC, {'mask': final_mask})
        
        if success:
            process_segmentation_request.count += 1
            if process_segmentation_request.count % 10 == 0:
                logging.info(f"Processed and sent {process_segmentation_request.count} segmentation results")
        
        logging.info("Processed request and sent segmentation mask.")
        
    except Exception as e:
        logging.error(f"Error processing segmentation request: {e}", exc_info=True)
        # Send empty mask on error
        try:
            H, W = data['rgb'].shape[:2]
            empty_mask = np.zeros((H, W), dtype=bool)
            pubsub_instance.publish(MASK_TOPIC, {'mask': empty_mask})
        except:
            pass

# Initialize counter
process_segmentation_request.count = 0

# --- Main Client Logic ---

def main():
    global pubsub_instance, dino_model_instance, predictor_instance, device_instance, box_threshold_instance
    
    # --- Configuration ---
    config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    grounded_checkpoint = "groundingdino_swint_ogc.pth"
    sam_version = "vit_b"
    sam_hq_checkpoint = "sam_hq_vit_b.pth"
    box_threshold = 0.3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bert_base_uncased_path = None 

    # --- Load Models ---
    logging.info("Loading models...")
    dino_model = load_grounding_dino_model(config_file, grounded_checkpoint, bert_base_uncased_path, device=device)
    predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device))
    logging.info("Models loaded successfully.")
    
    # Store global references for handler
    dino_model_instance = dino_model
    predictor_instance = predictor
    device_instance = device
    box_threshold_instance = box_threshold

    # --- Setup PubSub Manager ---
    logging.info(f"Setting up PubSub manager on port {port}...")
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
            'buffer_size': 50,
            'mode': 'consumer'  # 请求使用消费者模式，确保每个请求只被处理一次
        },
        MASK_TOPIC: {
            'examples': {
                'mask': np.zeros((480, 640), dtype=bool)
            },
            'buffer_size': 50,
            'mode': 'consumer'  # 结果使用消费者模式，确保每个结果只被读取一次
        }
    }
    
    # Setup subscriber with topic configurations
    pubsub.setup_subscriber(topics_config)
    
    logging.info("PubSub topics created and ready.")

    print("Starting segmentation processor...")
    print("Listening for requests and sending back mask results...")
    
    # Register handler for segmentation requests - PubSub will automatically manage the listener thread!
    pubsub.register_topic_handler(REQUEST_TOPIC, process_segmentation_request, check_interval=0.001)
    
    # --- Main Processing Loop ---
    logging.info("Client is ready. Waiting for segmentation requests...")
    try:
        print("Segmentation processor is running. Press Ctrl+C to stop...")
        while True:
            time.sleep(1.0)
                
    except KeyboardInterrupt:
        logging.info("Client shutting down by user.")
    finally:
        pubsub.stop(role='both')
        logging.info("Client has shut down.")

if __name__ == '__main__':
    import os
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s')
    main()