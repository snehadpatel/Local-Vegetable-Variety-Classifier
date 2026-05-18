import os
import io
import torch
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from PIL import Image
from torchvision import transforms
from models import CustomGAPCNN, CustomMultiScaleCNN, CustomResidualCNN

# 1. Initialize FastAPI Server
app = FastAPI(
    title="Antigravity Vegetable Variety Classifier",
    description="Serving wholly trained custom PyTorch CNN models on Apple Silicon MPS GPU.",
    version="1.0.0"
)

# 2. Mount Static Files and Templates
# This allows serving custom CSS styles and raw images using absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# 3. Model Configurations & Classes
CLASSES = ['Green Chilli (Marcha)', 'Ladies finger', 'Pointed gourd', 'ivy guard', 'peas']
GAP_WEIGHTS_PATH = "gap_model.pth"
RESIDUAL_WEIGHTS_PATH = "residual_model.pth"
MULTISCALE_WEIGHTS_PATH = "multiscale_model.pth"

# 4. Device Setup (Leveraging hardware acceleration if available)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("[FastAPI Init] Hardware Acceleration: Active Apple Silicon GPU ('mps').")
else:
    device = torch.device("cpu")
    print("[FastAPI Init] Hardware Acceleration: Active CPU.")

# 5. Load Ensemble Neural Network weights
def load_model(model_class, weights_path):
    try:
        if os.path.exists(weights_path):
            model = model_class(num_classes=len(CLASSES))
            model.load_state_dict(torch.load(weights_path, map_location=device))
            model.to(device)
            model.eval()
            print(f"[FastAPI Init] Loaded {model_class.__name__} successfully from '{weights_path}'!")
            return model
        else:
            print(f"[FastAPI Init] WARNING: Weight file '{weights_path}' not found! Ensemble will be incomplete.")
            return None
    except Exception as e:
        print(f"[FastAPI Init] ERROR loading {model_class.__name__}: {e}")
        return None

print("[FastAPI Init] Initializing Softmax Ensemble (GAP, Multi-Scale, Residual)...")
model_gap = load_model(CustomGAPCNN, GAP_WEIGHTS_PATH)
model_residual = load_model(CustomResidualCNN, RESIDUAL_WEIGHTS_PATH)
model_multiscale = load_model(CustomMultiScaleCNN, MULTISCALE_WEIGHTS_PATH)

ensemble_models = [m for m in [model_gap, model_residual, model_multiscale] if m is not None]

# 6. Prediction Preprocessing Transforms
# Standardized resizing and ImageNet mean/std scaling
predict_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# =====================================================================
# GET Route: Serves the Web Dashboard Home Page
# =====================================================================
@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    """
    Renders and serves the interactive dark glassmorphism dashboard UI.
    """
    return templates.TemplateResponse(request=request, name="index.html")


# =====================================================================
# POST Route: Predict Class Probabilities for Uploaded Image
# =====================================================================
@app.post("/predict")
async def predict_vegetable(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file, processes it, runs inference through
    our Custom Residual CNN model, and returns classification probabilities.
    """
    # 1. Model Validation Check
    if not ensemble_models:
        raise HTTPException(
            status_code=503,
            detail="Classification ensemble is not loaded. Please make sure the .pth weight files are present on the server."
        )

    # 2. File Content Type Validation
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only standard image files (JPEG, PNG, WEBP) are supported."
        )

    try:
        # 3. Read File Bytes and Open Image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 4. Apply Transforms and Construct Batch Dimension
        tensor = predict_transforms(image).unsqueeze(0).to(device)
        
        # 5. Run Softmax Ensemble Inference
        with torch.no_grad():
            ensemble_probs = []
            for model in ensemble_models:
                outputs = model(tensor)
                probs = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
                ensemble_probs.append(probs)
                
            # Average the probabilities across all active models in the ensemble
            probabilities = sum(ensemble_probs) / len(ensemble_probs)
            
        # 6. Organize Response Data
        prediction_index = probabilities.argmax()
        prediction = CLASSES[prediction_index]
        confidence = float(probabilities[prediction_index])
        
        # Compile dictionary of individual class percentages
        probs_dict = {CLASSES[i]: float(probabilities[i]) for i in range(len(CLASSES))}
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probs_dict
        }

    except Exception as e:
        print(f"[FastAPI Predict] Error during processing: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while running model classification: {str(e)}"
        )


# =====================================================================
# Server Startup Check
# =====================================================================
if __name__ == "__main__":
    import uvicorn
    # Start ASGI Web Server locally on port 8000
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
