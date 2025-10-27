import io, json, torch
from PIL import Image
from torchvision import transforms
from src.model import build_model

MODELS = {}
CLASSES = {}
TFM = transforms.Compose([transforms.Resize((224,224)),
                          transforms.ToTensor(),
                          transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

def _load_one(task):
    state = torch.load(f"models/{task}/model.pt", map_location="cpu")
    classes = state.get("classes") or json.load(open(f"models/{task}/labels.json"))
    m = build_model(num_classes=len(classes), pretrained=False)
    m.load_state_dict(state["state_dict"])
    m.eval()
    return m, classes

def init():
    for task in ["condition", "pressure"]:
        try:
            m, cls = _load_one(task)
            MODELS[task], CLASSES[task] = m, cls
        except Exception:
            # Task not packaged is fine for POC
            pass

def run(raw):
    try:
        body = json.loads(raw)
        task = body.get("task", "condition")
        if task not in MODELS:
            return {"error": f"task '{task}' not available"}
        if "image_base64" in body:
            import base64
            img_bytes = base64.b64decode(body["image_base64"])
        else:
            img_bytes = bytes(body["image_bytes"])
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        x = TFM(img).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(MODELS[task](x), dim=1).flatten().tolist()
        idx = int(torch.tensor(probs).argmax().item())
        return {"task": task, "pred_class": CLASSES[task][idx], "probs": probs}
    except Exception as e:
        return {"error": str(e)}
