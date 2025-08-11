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
import cv2
import logging
import time
from multiprocessing.managers import BaseManager
from queue import Empty



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

# Shared Memory Queue
from shm_lib.shared_memory_queue import SharedMemoryQueue

# --- Helper Functions ---
port = 5000   

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

# --- Shared Memory Manager ---

class QueueManager(BaseManager):
    """Manager to handle shared memory queues over the network."""
    pass

# --- Main Client Logic ---

def main():
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

    # --- Connect to Shared Memory Server ---
    logging.info(f"Connecting to Shared Memory server at 127.0.0.1:{port}...")
    QueueManager.register('get_req_queue')
    QueueManager.register('get_res_queue')
    
    manager = QueueManager(address=('127.0.0.1', port), authkey=b'foundationpose')
    manager.connect()
    logging.info("Connected to server.")

    req_queue = manager.get_req_queue()
    res_queue = manager.get_res_queue()

    # --- Main Processing Loop ---
    logging.info("Client is ready. Waiting for segmentation requests...")
    while True:
        try:
            # Get request from the server
            req_data = req_queue.get()
            
            rgb_image_np = req_data['rgb']
            text_prompt = decode_text_prompt(req_data['prompt'])
            logging.info(f"Received request with prompt: '{text_prompt}'")
            H, W, _ = rgb_image_np.shape

            # Run GroundingDINO to get boxes
            image_tensor = prepare_image_for_dino(rgb_image_np, device)
            text_threshold = 0.25 # Value from test.py
            boxes = get_grounding_output(
                dino_model, image_tensor, text_prompt, box_threshold, text_threshold, device=device
            )

            if boxes.size(0) == 0:
                logging.warning("No objects detected. Returning an empty mask.")
                empty_mask = np.zeros((H, W), dtype=bool)
                res_queue.put({'mask': empty_mask})
                continue
            
            # Prepare boxes for SAM (cxcywh -> xyxy format)
            boxes_xyxy = boxes.clone()
            boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * W
            boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * H
            boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * W
            boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * H
            
            # Run SAM predictor to get masks
            predictor.set_image(rgb_image_np)
            transformed_boxes = predictor.transform.apply_boxes_torch(boxes_xyxy, (H, W)).to(device)

            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            
            # Combine masks and send the result back
            final_mask = torch.any(masks, dim=0).squeeze(0).cpu().numpy()
            res_queue.put({'mask': final_mask})
            logging.info("Processed request and sent segmentation mask.")

        except (ConnectionResetError, BrokenPipeError, EOFError):
            logging.warning("Connection to the server was lost. Shutting down client.")
            break
        except (KeyboardInterrupt, SystemExit):
            logging.info("Client shutting down by user.")
            break
        except Exception as e:
            logging.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
            break # Exit on any other critical error
            
    logging.info("Client has shut down.")

if __name__ == '__main__':
    import os
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s')
    main()