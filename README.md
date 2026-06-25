# VeggieSense — Local Vegetable Variety Classifier

VeggieSense is a state-of-the-art, hardware-accelerated computer vision platform designed to classify regional Gujarati vegetable varieties with expert-level precision. Powered by a **weighted Softmax Ensemble of four custom Convolutional Neural Networks (CNNs)** built in PyTorch, the system utilizes real-time background removal and Bayesian class-prior calibration to achieve robust, high-confidence classifications.

---

## 🌟 Key Features
- **Intelligent Preprocessing Pipeline:** Automatically balances dataset splits, applies targeted image augmentation for minority classes, and resizes inputs to standard $224 \times 224$ pixels.
- **Dynamic Background Elimination:** Integrates a parallel CPU-accelerated `rembg` (U-2-Net) pipeline to strip out cluttered environments, pasting the foreground vegetable onto pure black to maximize model focus.
- **Weighted Neural Ensemble:** Combines prediction vectors from 4 customized architectures:
  - **Global Average Pooling (GAP) CNN:** Parameter-efficient network utilizing global averages rather than massive fully-connected weights.
  - **Multi-Scale CNN:** Inception-inspired parallel convolutional kernels ($3\times3$, $5\times5$, $7\times7$) for multi-scale feature extraction.
  - **Residual CNN:** Custom scratch network utilizing residual shortcut bypass links to prevent vanishing gradients.
  - **Deep CNN:** 5-layer classic deep convolutional network with batch normalization and dropout layers.
- **Apple Silicon MPS Acceleration:** Preconfigured to run model training and backend inference using native Apple Silicon GPU (`mps`) acceleration, falling back automatically to CPU.
- **Bayesian Prior Calibration:** Employs temperature scaling to sharpen confidence and class-prior probability scaling to prevent classifier bias on highly represented categories (e.g., Ivy Gourd).
- **Glassmorphic Web Dashboard:** Fast, intuitive user dashboard built on FastAPI, supporting file drag-and-drop, sample quick-loads, and live webcam captures.

---

## 🥦 Supported Vegetable Varieties

| English Name | Gujarati Name (Transliterated) | Gujarati Script | Special Processing |
| :--- | :--- | :--- | :--- |
| **Ivy Gourd** | Tindoda | તિંડોળા | 15% Bayesian probability penalty (prior calibration) |
| **Pointed Gourd** | Parvad | પરવળ | Standard class weights |
| **Green Chilli** | Lila Marcha | લીલા મરચા | Target weight scaling |
| **Okra (Ladies finger)** | Bhinda | ભીંડા | Standard class weights |
| **Peas** | Lila Vatana | લીલા વટાણા | Standard class weights |

---

## 📂 Project Structure

```directory
├── app.py                      # FastAPI web server and Softmax Ensemble inference logic
├── models.py                   # PyTorch classes for all custom CNN architectures
├── utils.py                    # Helper dataset loaders, early stopping, and class weights
├── preprocess_and_balance.py   # Dataset splitter (70/15/15) & class balancer script
├── remove_backgrounds.py       # rembg-based parallel background removal pipeline
├── evaluate_all.py             # Script to evaluate model accuracies across all weights
├── calculate_neurons.py        # Math helper script for calculating dense layer sizes
├── create_report.py            # Generates comprehensive training progress reports
├── run_inference_samples.py    # Console script to test inference on sample images
├── templates/
│   └── index.html              # Dashboard HTML template (Forest Emerald light theme)
├── static/
│   ├── css/
│   │   └── style.css           # Glassmorphic layout custom stylesheet
│   └── samples/
│       └── [veggies].jpg       # Standard high-quality sample images for instant UI testing
└── dataset/                    # Root directory for raw source images (user-supplied)
```

---

## ⚙️ Setup and Installation

### 1. Prerequisite Environments
Ensure you have **Python 3.9+** and `pip` installed.

### 2. Create and Activate Virtual Environment
```bash
# Create the virtual environment
python3 -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install torch torchvision fastapi uvicorn rembg pillow numpy jinja2 python-multipart matplotlib
```

---

## 🚀 Execution & Training Pipeline

### Step 1: Preprocess & Balance Raw Data
Place raw photos inside folder paths matching their class names in `dataset/` (e.g. `dataset/peas/image_1.jpg`). Then run:
```bash
python preprocess_and_balance.py
```
This splits the dataset into `dataset_processed/{train,val,test}` and scales minority classes to balance class distributions.

### Step 2: Remove Image Backgrounds
Standardize the training dataset by removing messy backdrops:
```bash
python remove_backgrounds.py
```
This outputs processed black-background images into `dataset_nobg/` using parallel CPU cores.

### Step 3: Train Neural Network Models
To train the individual custom CNN models:
```bash
# Train the Residual model
python train_residual_cnn.py

# Train the GAP model
python train_gap_cnn.py

# Train the Multi-Scale model
python train_multiscale_cnn.py

# Train the Deep model
python train_deep_cnn.py
```
Training runs natively on your macOS GPU using MPS (`mps`) for rapid acceleration. Outputs save as `.pth` files.

### Step 4: Run the Web Server
Launch the interactive VeggieSense dashboard:
```bash
python app.py
```
The server will bind to `http://localhost:8000/`. You can upload local images, snap photos using your webcam, or click the preloaded sample cards to see live classification probabilities!

---

## ☁️ Cloud Deployment

VeggieSense is fully containerized and ready for cloud deployment.

### Option A: Hugging Face Spaces (Recommended - Free 16GB RAM)
Hugging Face Spaces provides high-performance CPU instances ideal for running PyTorch models without memory crashes.
1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space).
2. Choose **Docker** as the SDK, then select the **Blank** template.
3. Clone your newly created Space repository locally.
4. Copy all VeggieSense files (including `app.py`, `models.py`, `utils.py`, `templates/`, `static/`, `requirements.txt`, `Dockerfile`, and all `.pth` weights) into the Space folder.
5. Push the changes to Hugging Face:
   ```bash
   git add .
   git commit -m "Deploy VeggieSense Ensemble App"
   git push
   ```
6. Hugging Face will automatically build and run the Docker container.

### Option B: Render Web Service
1. Link your GitHub repository containing the VeggieSense files to [Render](https://render.com/).
2. Create a new **Web Service** on Render and select this repository.
3. Render will auto-detect the `Dockerfile` and configure the build process automatically.
4. Note: On Render's Free tier, the 512MB RAM limit may occasionally result in Out-Of-Memory (OOM) failures under heavy `rembg` processing. A paid Starter tier or Option A is recommended.

---

## 🧠 Ensemble Voting Mechanics

During prediction, the model feeds the input through all loaded networks. The resulting output logits are divided by a temperature factor of $T = 0.8$ to sharpen confidence, converted to probabilities via Softmax, and combined using a weighted sum:

$$\text{P}_{\text{ensemble}}(C) = \sum_{m} w_m \cdot \text{P}_m(C)$$

Where model weights are mathematically proportional to individual test accuracies:
- **Deep CNN:** $w = 0.28$
- **GAP CNN:** $w = 0.26$
- **Residual CNN:** $w = 0.24$
- **Multi-Scale CNN:** $w = 0.22$

---

## 👥 Engineering Team
- **Sneha Patel** — Project Lead & Core AI/CV Engineer (Designed & trained all custom CNN architectures; engineered the hardware acceleration pipeline)
- **Hardi Patel** — AI Research Collaborator (Contributed to neural architecture research and model validation design)
- **Tithi Patel** — UI/UX Design Collaborator (Assisted with the interactive layouts and aesthetic design of the dashboard)
- **Ayush Singh** — Backend API Collaborator (Supported the FastAPI backend integration and server deployment logistics)
- **Amaan Malik** — Data Engineering Collaborator (Helped with raw image dataset collection and structuring)
