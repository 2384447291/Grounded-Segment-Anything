import argparse
import os
import sys
import numpy as np
import json
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)

def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image_pil, None)
    return image_pil, image

def load_model(model_config_path, model_checkpoint_path, bert_base_uncased_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    args.bert_base_uncased_path = bert_base_uncased_path
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]
    # Filter by box_threshold
    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]
    boxes_filt = boxes[filt_mask]
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    pred_phrases = []
    for logit in logits_filt:
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)
    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1,1,-1)
    ax.imshow(mask_image)

def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2]-box[0], box[3]-box[1]
    ax.add_patch(plt.Rectangle((x0,y0), w,h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)

def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0
    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10,10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    save_path = os.path.join(output_dir, 'mask.jpg')
    plt.savefig(save_path, bbox_inches="tight", dpi=300, pad_inches=0.0)
    plt.close()

    json_data = [{'value': 0, 'label': 'background'}]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1]
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)

def resize_and_convert_to_png(input_path, output_path, width=640, height=480):
    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: 无法读取图片 {input_path}")
        return False
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    output_path = os.path.splitext(output_path)[0] + ".png"
    cv2.imwrite(output_path, resized_image)
    print(f"已保存调整大小的 PNG 图片: {output_path}")
    return output_path

def extract_yellow_and_save(input_path, output_gray_path):
    image = cv2.imread(input_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    result = np.zeros_like(image)
    result[mask > 0] = [255,255,255]
    result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(output_gray_path, result_gray)
    print(f"已保存二值灰度图: {output_gray_path}")
    plt.imshow(result_gray, cmap='gray')
    plt.axis('off')
    plt.show()

def main():
    config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    grounded_checkpoint = "groundingdino_swint_ogc.pth"
    sam_version = "vit_b"
    sam_hq_checkpoint = "sam_hq_vit_b.pth"
    use_sam_hq = True
    image_path = "test.png"
    text_prompt = "yellow and black utility knife"
    output_dir = "outputs"
    box_threshold = 0.3
    text_threshold = 0.25
    device = "cuda"
    bert_base_uncased_path = None

    os.makedirs(output_dir, exist_ok=True)

    # 加载图像和模型
    image_pil, image = load_image(image_path)
    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    model = load_model(config_file, grounded_checkpoint, bert_base_uncased_path, device=device)

    # image = image.half()
    # model = model.half()

    boxes_filt, pred_phrases = get_grounding_output(model, image, text_prompt, box_threshold, text_threshold, device=device)

    print(f"使用 SAM-HQ 模型: {sam_hq_checkpoint}")
    predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device))

    image_cv = cv2.imread(image_path)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_cv)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image_cv.shape[:2]).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(device),
        multimask_output=False,
    )

    target_size = (480, 640)
    masks_resized = torch.nn.functional.interpolate(masks.float(), size=target_size, mode="nearest")

    plt.figure(figsize=(10, 10))
    plt.imshow(image_cv)
    for mask in masks_resized:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.numpy(), plt.gca(), label)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "grounded_sam_output_hexagona_brick.jpg"),
                bbox_inches="tight", dpi=300, pad_inches=0.0)
    plt.close()

    print("成功保存分割后的数据")
    save_mask_data(output_dir, masks_resized, boxes_filt, pred_phrases)

    # 后续处理：调整大小、转png、提取黄色区域
    mask_jpg_path = os.path.join(output_dir, "mask.jpg")
    resized_png_path = resize_and_convert_to_png(mask_jpg_path, os.path.join(output_dir, "mask_out.png"))
    if resized_png_path:
        gray_output_path = os.path.join(output_dir, "extracted_yellow_mask.png")
        extract_yellow_and_save(resized_png_path, gray_output_path)

if __name__ == "__main__":
    import os
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
    main()
