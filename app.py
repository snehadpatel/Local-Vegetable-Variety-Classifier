import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import io
import torch
import numpy as np
from rembg import remove
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from PIL import Image
from torchvision import transforms
from models import CustomGAPCNN, CustomMultiScaleCNN, CustomResidualCNN, CustomDeepCNN

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
DEEP_WEIGHTS_PATH = "deep_model.pth"

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

print("[FastAPI Init] Initializing Softmax Ensemble (GAP, Multi-Scale, Residual, Deep)...")
model_gap = load_model(CustomGAPCNN, GAP_WEIGHTS_PATH)
model_residual = load_model(CustomResidualCNN, RESIDUAL_WEIGHTS_PATH)
model_multiscale = load_model(CustomMultiScaleCNN, MULTISCALE_WEIGHTS_PATH)
model_deep = load_model(CustomDeepCNN, DEEP_WEIGHTS_PATH)

ensemble_models = [m for m in [model_gap, model_residual, model_multiscale, model_deep] if m is not None]

# 6. Prediction Preprocessing Transforms
# Standardized resizing and ImageNet mean/std scaling
predict_transforms = transforms.Compose([
    transforms.Resize(256),          # Resize shortest edge to 256, maintaining aspect ratio
    transforms.CenterCrop(224),      # Crop the center 224x224 to prevent shape squishing
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
async def predict_vegetable(request: Request, file: UploadFile = File(...)):
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
        # 3. Read image bytes and remove background instantly
        contents = await file.read()
        
        # Strip background to pure black to match the dataset_nobg training
        nobg_bytes = remove(contents)
        
        # Convert to RGB image, pasting transparent background onto black
        img_rgba = Image.open(io.BytesIO(nobg_bytes)).convert("RGBA")
        black_bg = Image.new("RGBA", img_rgba.size, (0, 0, 0, 255))
        black_bg.paste(img_rgba, (0, 0), img_rgba)
        image = black_bg.convert("RGB")
        
        # 4. Apply Transforms and Construct Batch Dimension
        tensor = predict_transforms(image).unsqueeze(0).to(device)
        
        # 5. Run Weighted Softmax Ensemble Inference
        with torch.no_grad():
            # Accuracies from our latest test run:
            # DeepCNN: 86%, GAPCNN: 82%, ResidualCNN: 78%, MultiScaleCNN: 71%
            # We assign mathematically proportional voting power to the better models.
            all_models = [model_gap, model_residual, model_multiscale, model_deep]
            target_weights = [0.26, 0.24, 0.22, 0.28] 
            
            active_models_with_weights = [(m, w) for m, w in zip(all_models, target_weights) if m is not None]
            total_weight = sum(w for _, w in active_models_with_weights)
            
            weighted_probs = np.zeros(len(CLASSES))
            for model, weight in active_models_with_weights:
                outputs = model(tensor)
                
                # Apply Temperature Scaling (T=0.8) to sharpen the network's confidence
                outputs = outputs / 0.8 
                
                probs = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
                
                # Apply the model's specific voting weight
                weighted_probs += probs * (weight / total_weight)
                
            # Bayesian Class-Prior Calibration:
            # We recently added 125 Ivy Gourd images to the dataset. To prevent the Ensemble 
            # from developing a bias toward Ivy Gourd, we apply a mathematical 15% penalty 
            # to its final probability, preventing it from dominating Pointed Gourd or Peas.
            ivy_idx = CLASSES.index('ivy guard')
            weighted_probs[ivy_idx] *= 0.85 
            
            # Re-normalize the probabilities so they equal 100%
            probabilities = weighted_probs / weighted_probs.sum()
            
        # Mapping dataset classes to UI details
        veg_details = {
            'Green Chilli (Marcha)': {'name_en': 'Green Chilli', 'name_local': 'Lila Marcha', 'name_script': 'લીલા મરચા'},
            'Ladies finger': {'name_en': 'Okra', 'name_local': 'Bhinda', 'name_script': 'ભીંડા'},
            'Pointed gourd': {'name_en': 'Pointed Gourd', 'name_local': 'Parvad', 'name_script': 'પરવળ'},
            'ivy guard': {'name_en': 'Ivy Gourd', 'name_local': 'Tindoda', 'name_script': 'તિંડોળા'},
            'peas': {'name_en': 'Peas', 'name_local': 'Lila Vatana', 'name_script': 'લીલા વટાણા'}
        }
        
        # 6. Organize Response Data
        prediction_index = probabilities.argmax()
        raw_pred_name = CLASSES[prediction_index]
        confidence = float(probabilities[prediction_index])
        
        pred_obj = veg_details[raw_pred_name].copy()
        pred_obj["confidence"] = round(confidence * 100, 2)
        
        # Compile dictionary of individual class percentages for UI
        all_preds = []
        for i in range(len(CLASSES)):
            pct = round(float(probabilities[i]) * 100, 2)
            if pct > 0.01:
                ui_name = veg_details[CLASSES[i]]['name_en']
                all_preds.append({"name": ui_name, "pct": pct})
        
        # Sort predictions by percentage descending
        all_preds.sort(key=lambda x: x['pct'], reverse=True)
        pred_obj["all_preds"] = all_preds
        
        return templates.TemplateResponse(request=request, name="index.html", context={"prediction": pred_obj})

    except Exception as e:
        print(f"[FastAPI Predict] Error during processing: {e}")
        return templates.TemplateResponse(
            request=request, 
            name="index.html", 
            context={"error": f"An error occurred: {str(e)}"}
        )

# =====================================================================
# Server Startup (Deployment Ready)
# =====================================================================
if __name__ == "__main__":
    import uvicorn
    import os
    
    # Use environment port for deployments (e.g. Heroku, Render)
    port = int(os.environ.get("PORT", 8000))
    # Run in production mode (reload=False, bound to 0.0.0.0)
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False, workers=1)
