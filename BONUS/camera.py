import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torchvision

import sys
import subprocess

try:
    import pytorch_lightning as pl
    import timm
    import cv2
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "pytorch-lightning", "timm", "opencv-python"])
    import pytorch_lightning as pl
    import timm
    import cv2

GRID_SIZE          = 14
IMG_SIZE           = 224
NUM_BOXES_PER_CELL = 2


class YOLOLoss(nn.Module):
    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse = nn.MSELoss(reduction='sum')
        self.bce = nn.BCELoss(reduction='sum')

    def forward(self, predictions, targets):
        batch_size      = predictions.shape[0]
        coord_loss      = 0.0
        conf_loss_obj   = 0.0
        conf_loss_noobj = 0.0

        for b in range(NUM_BOXES_PER_CELL):
            s = b * 5
            pred_xy   = predictions[..., s:s+2]
            pred_wh   = predictions[..., s+2:s+4]
            pred_conf = predictions[..., s+4:s+5]
            tgt_xy    = targets[..., s:s+2]
            tgt_wh    = targets[..., s+2:s+4]
            tgt_conf  = targets[..., s+4:s+5]

            obj_mask   = tgt_conf
            noobj_mask = 1 - obj_mask

            coord_loss += self.mse(pred_xy * obj_mask, tgt_xy * obj_mask)
            coord_loss += self.mse(
                torch.sqrt(pred_wh  + 1e-6) * obj_mask,
                torch.sqrt(tgt_wh  + 1e-6) * obj_mask)

            conf_loss_obj   += self.bce(
                (pred_conf * obj_mask).clamp(1e-6, 1-1e-6),
                tgt_conf * obj_mask)
            conf_loss_noobj += self.bce(
                (pred_conf * noobj_mask).clamp(1e-6, 1-1e-6),
                tgt_conf * noobj_mask)

        total = (self.lambda_coord * coord_loss
                 + conf_loss_obj
                 + self.lambda_noobj * conf_loss_noobj)
        return total / batch_size


class ViTYOLODetector(pl.LightningModule):

    def __init__(self, lr=3e-5, weight_decay=1e-4,
                 lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.save_hyperparameters()
        self.criterion = YOLOLoss(lambda_coord=lambda_coord,
                                  lambda_noobj=lambda_noobj)

        self.backbone = timm.create_model(
            'deit_small_patch16_224',
            pretrained=False, num_classes=0)
        embed_dim = self.backbone.num_features  # 384

        self.yolo_head = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.Conv2d(128, 5 * NUM_BOXES_PER_CELL, kernel_size=1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feats    = self.backbone.forward_features(x)   # (B, 197, 384)
        patches  = feats[:, 1:, :]                     # (B, 196, 384)
        B        = patches.shape[0]
        feat_map = patches.permute(0, 2, 1).reshape(B, 384, GRID_SIZE, GRID_SIZE)
        out      = self.sigmoid(self.yolo_head(feat_map))  # (B, 10, 14, 14)
        out      = out.permute(0, 2, 3, 1).contiguous()   # (B, 14, 14, 10)
        return out

    def configure_optimizers(self):
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay)

DROPOUT       = 0.3
FC_DROPOUT    = 0.5
WIDTHS        = [64, 128, 256, 512]
N_CLASSES     = 7
EMOTION_SIZE  = 48

def conv_block(cin, cout, p):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(True),
        nn.MaxPool2d(2), nn.Dropout(p),
    )

class EmotionCNN(nn.Module):
    def __init__(self, dropout=DROPOUT):
        super().__init__()
        ch = [1, *WIDTHS]
        self.features = nn.Sequential(
            *[conv_block(ch[i], ch[i+1], dropout) for i in range(len(WIDTHS))])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(WIDTHS[-1], 256), nn.BatchNorm1d(256),
            nn.ReLU(True), nn.Dropout(FC_DROPOUT),
            nn.Linear(256, N_CLASSES),
        )
    def forward(self, x):
        return self.head(self.features(x))


def decode_predictions(preds, conf_thresh, W_orig, H_orig):
    boxes  = []
    scores = []

    for cy in range(GRID_SIZE):
        for cx in range(GRID_SIZE):
            for b in range(NUM_BOXES_PER_CELL):
                s    = b * 5
                conf = preds[cy, cx, s + 4].item()
                if conf < conf_thresh:
                    continue

                rel_x  = preds[cy, cx, s + 0].item()
                rel_y  = preds[cy, cx, s + 1].item()
                w_norm = preds[cy, cx, s + 2].item()
                h_norm = preds[cy, cx, s + 3].item()

                # μετατροπή σε pixel coords
                cx_abs = ((cx + rel_x) / GRID_SIZE) * W_orig
                cy_abs = ((cy + rel_y) / GRID_SIZE) * H_orig
                w      = w_norm * W_orig
                h      = h_norm * H_orig

                x1 = cx_abs - w / 2
                y1 = cy_abs - h / 2
                x2 = cx_abs + w / 2
                y2 = cy_abs + h / 2

                boxes.append([x1, y1, x2, y2])
                scores.append(conf)

    return boxes, scores


def predict_emotion(face_bgr, emotion_model, device):
    if emotion_model is None:
        return ""
    EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    img_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    pil_img  = Image.fromarray(img_gray)
    w, h     = pil_img.size
    side     = max(w, h)
    padded   = Image.new('L', (side, side), 0)
    padded.paste(pil_img, ((side - w) // 2, (side - h) // 2))
    tf = transforms.Compose([
        transforms.Resize((EMOTION_SIZE, EMOTION_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    inp = tf(padded).unsqueeze(0).to(device)
    with torch.no_grad():
        _, pred = torch.max(emotion_model(inp), 1)
    idx = pred.item()
    return EMOTIONS[idx] if idx < len(EMOTIONS) else f"Class {idx}"



def run_webcam():
    FACE_MODEL_WEIGHTS    = "final_vit_yolo.pth"
    EMOTION_MODEL_WEIGHTS = "best_model.pt"
    CONF_THRESH           = 0.3
    IOU_THRESH            = 0.2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    face_model = ViTYOLODetector()
    try:
        state = torch.load(FACE_MODEL_WEIGHTS, map_location=device,
                           weights_only=False)
        # αν είναι Lightning checkpoint
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        face_model.load_state_dict(state)
        face_model.to(device).eval()
    except Exception as e:
        print("error")
        return

    emotion_model = None
    try:
        loaded = torch.load(EMOTION_MODEL_WEIGHTS, map_location=device,
                            weights_only=False)
        emotion_model = EmotionCNN()
        if isinstance(loaded, dict):
            key = 'model' if 'model' in loaded else \
                  'state_dict' if 'state_dict' in loaded else None
            emotion_model.load_state_dict(loaded[key] if key else loaded)
        else:
            emotion_model = loaded
        emotion_model.to(device).eval()
    except Exception as e:
        print(f"error")

    face_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No camera")
        return
    print("ΟΚ — 'q' for exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        H_orig, W_orig = frame.shape[:2]
        pil_img    = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = face_tf(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = face_model(img_tensor)[0].cpu()  # (14, 14, 10)

        boxes, scores = decode_predictions(preds, CONF_THRESH, W_orig, H_orig)

        if boxes:
            boxes_t  = torch.tensor(boxes,  dtype=torch.float32)
            scores_t = torch.tensor(scores, dtype=torch.float32)
            keep     = torchvision.ops.nms(boxes_t, scores_t, IOU_THRESH)

            for idx in keep:
                x1, y1, x2, y2 = map(int, boxes[idx])
                conf = scores[idx]

                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(W_orig, x2); y2 = min(H_orig, y2)

                emotion_text = ""
                face_roi = frame[y1:y2, x1:x2]
                if face_roi.size > 0 and emotion_model is not None:
                    emotion_text = predict_emotion(face_roi, emotion_model, device)

                color = (255, 200, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"{emotion_text} ({conf:.2f})" if emotion_text \
                        else f"Face ({conf:.2f})"
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1-th-10), (x1+tw, y1), color, -1)
                cv2.putText(frame, label, (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        cv2.imshow('ViT-YOLO + Emotion Detection  [Q = exit]', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_webcam()