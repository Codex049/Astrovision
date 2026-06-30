# 🔭 AstroVision AI

**AstroVision** is an intelligent astrophysics analysis tool that combines computer vision and large language models to interpret astronomical images. Upload images of celestial objects, and AstroVision will classify them, extract photometric properties, analyze their morphology, and provide AI-powered insights.

This project was made to put newly learnt machine learning skills to use and acts as a demonstration. However, with a little bit of refining (such as with the accuracy of visual models and presentation of the data), it could possibly aid amateur astronomers and help contextualize findings at least at a basic level. Originally built near the end of the summer of 2025, recieved a polished presentable look & published June 29th 2026

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [CNN Model Specifications](#cnn-model-specifications)
4. [How the Processing Pipeline Works](#how-the-processing-pipeline-works)
5. [Server.py Documentation](#serverpy-documentation)
6. [LLM Chat System](#llm-chat-system)
7. [Frontend & UI](#frontend--ui)
8. [Setup & Installation](#setup--installation)
9. [Running the Application](#running-the-application)
10. [Data Files](#data-files)

---

## 📖 Project Overview

AstroVision is built to analyze astronomical images using three-stage deep learning pipeline:

1. **Star/Galaxy Classification** — Determines if an object is a star or galaxy
2. **UGRIZ Photometry Regression** — Predicts magnitude values in 5 photometric bands
3. **Galaxy Morphology Classification** — If a galaxy is detected, predicts its structural type (elliptical, spiral, edge-on, or merger)

The system then uses an LLM powered by Ollama to provide educational, accessible explanations of the astronomical data to users.

### Key Features

- ✨ **Multi-stage CNN pipeline** for comprehensive object analysis
- 🧠 **AI-powered chat** using Ollama for natural language insights
- 📊 **Photometric analysis** with UGRIZ magnitude extraction
- 🔬 **Galaxy morphology classification** for structural analysis
- 🌐 **Web-based interface** with real-time results
- 📍 **Runs Locally** on nearly any device

---

## 🏗️ Architecture

### System Flow

```
┌──────────────────────────────────┐
│   User uploads astronomical      │
│   image via web interface        │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│  server.py receives upload       │
│  (Flask backend)                 │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│  Architecture.py process()       │
│  3-stage ML pipeline             │
└────────────┬─────────────────────┘
             │
    ┌────────┼────────┐
    │        │        │
    ▼        ▼        ▼
  [1]      [2]      [3]
  Star/    UGRIZ    Morphology
  Galaxy   Regress  (if galaxy)
  Classify         Classify
    │        │        │
    └────────┼────────┘
             │
             ▼
┌──────────────────────────────────┐
│  Results returned to frontend    │
│  - object_type (star/galaxy)     │
│  - star_galaxy_probs             │
│  - ugriz magnitudes              │
│  - morphology_probs (if galaxy)  │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│  Frontend displays results &     │
│  shows chat interface            │
└────────────┬─────────────────────┘
             │
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
  Results        Chat with
  Dashboard      LLM
  (charts,       (Ollama)
   magnitudes)   for insights
```

### Tech Stack

- **Backend**: Flask (Python web framework)
- **Deep Learning**: PyTorch, TorchVision
- **Models**: EfficientNet, ConvNeXt
- **LLM**: Ollama (local LLM inference)
- **Frontend**: HTML5, Bootstrap 5, Chart.js, Vanilla JavaScript
- **Image Processing**: Pillow, OpenCV, NumPy

---

## 🧠 CNN Model Specifications

The system uses **4 trained PyTorch models** for end-to-end astronomical analysis.

### 1. **Star/Galaxy Classifier** (EfficientNet-B0)
- **Path**: `Models/StarGalaxy_CLassifier/checkpoint.pth`
- **Architecture**: EfficientNet-B0 with custom classification head
- **Input**: 256×256 RGB images (normalized with ImageNet statistics)
- **Output**: 2 logits → softmax → [galaxy_prob, star_prob]
- **Classes**: 2 (Galaxy, Star)
- **Purpose**: Binary classification to distinguish stars from galaxies
- **Class Definition**:
  - Class 0: **Galaxy** (extended, structured objects)
  - Class 1: **Star** (point-like sources)

**Model Architecture** (`StarGalaxyEffNet`):
```python
- EfficientNet-B0 features (pretrained)
- AdaptiveAvgPool2d(1)
- Dropout(0.2)
- Linear(base_classifier_in_features → 2)
```

---

### 2. **UGRIZ Photometry Regressor** (ConvNeXt-Base)
- **Path**: `Models/UGRIZ/model.pth`
- **Architecture**: ConvNeXt-Base with regression head + center-weighted mask
- **Input**: 256×256 RGB images
- **Output**: 5 continuous values (u, g, r, i, z magnitudes)
- **Output Range**: Typical SDSS magnitudes (≈12-25, but can exceed)
- **Bands**:
  - **u (ultraviolet)**: 300-400 nm, sensitive to hot, young stars
  - **g (green)**: 400-550 nm, visible blue-green
  - **r (red)**: 550-700 nm, visible red
  - **i (near-infrared)**: 700-900 nm, near-infrared
  - **z (far-infrared)**: 900-1100 nm, far-infrared

**Key Feature: Center-Weight Mask**
- Gaussian weighting emphasizes the central region of feature maps (to minimize distractions in a compact region)
- Default mask size: 7×7, sigma (how much surrounding pixels affect the focus, with higher values leading to a larger focus area): 1.5
- Resized to match feature map dimensions during forward pass
- Reduces noise from edges, focusing on core object

**Model Architecture** (`GalaxyConvNeXt`):
```python
- ConvNeXt-Base features
- Gaussian center-weight mask (registered as buffer)
- AdaptiveAvgPool2d(1)
- Regression head:
  - Linear(in_features → 512)
  - ReLU
  - Dropout(0.3)
  - Linear(512 → 5)  (5 magnitude outputs)
```

---

### 3. **Galaxy Morphology Classifier (4-class)** (EfficientNet-B0)
- **Path**: `Models/EfficientNet_Morphology2/model.pth`
- **Architecture**: EfficientNet-B0 with 4-class classification head
- **Input**: 256×256 RGB images
- **Output**: 4 sigmoid probabilities (multi-class, can overlap)
- **Classes**: 4
- **Purpose**: Classify galaxy structure when a galaxy is detected
- **Class Definitions**:
  - **Elliptical**: Smooth, featureless, rounded/oval shapes
  - **Spiral**: Disk-like with spiral arms (Sa, Sb, Sc types)
  - **Edge-on**: Disk galaxies viewed edge-on (appear as thin lines)
  - **Merger**: Distorted/interacting galaxies, asymmetric shapes

**Model Architecture** (`GalaxyEfficientNet4`):
```python
- EfficientNet-B0 features (pretrained)
- Gaussian center-weight mask (7×7, sigma 1.5)
- AdaptiveAvgPool2d(1)
- Classification head:
  - Linear(base_classifier_in_features → 256)
  - ReLU
  - Dropout(0.4)
  - Linear(256 → 4)
  - Sigmoid  (Each class independent probability)
```

---

### 4. **Galaxy Morphology Classifier (6-class)** (EfficientNet-B0)
- **Path**: `Models/EfficientNet_Morphology1/model.pth`
- **Architecture**: EfficientNet-B0 with 6-class classification head
- **Input**: 256×256 RGB images
- **Output**: 6 sigmoid probabilities
- **Classes**: 6 (Extended Hubble classification)
- **Purpose**: Alternative morphology classifier with finer granularity
- **Class Definitions**:
  - **E0-E3**: Elliptical galaxies (increasingly elongated)
  - **S0**: Lenticular galaxies (disk without arms)
  - **Sa-Sc**: Spiral galaxies (tight to loose arms)
  - **Irregular**: Chaotic morphology

**Model Architecture** (`GalaxyEfficientNet6`):
```python
- EfficientNet-B0 features (pretrained)
- Gaussian center-weight mask (7×7, sigma 1.5)
- AdaptiveAvgPool2d(1)
- Classification head:
  - Linear(base_classifier_in_features → 256)
  - ReLU
  - Dropout(0.3)
  - Linear(256 → 6)
  - Sigmoid
```

---

### Shared Preprocessing

All models use **ImageNet normalization** and **center-weight masking**:

```python
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

**Center-Weight Mask**:
- Generated as a Gaussian distribution centered on the feature map
- Applied multiplicatively to feature maps before pooling
- Helps the model focus on the central galaxy/star
- Sigma default: 1.5 (determines spread; larger σ = broader weighting)

---

## ⚙️ How the Processing Pipeline Works

### Image Processing in Simple Terms

When you upload an image to AstroVision, it goes through a 3-stage analysis process. Think of it like a pipeline where each stage asks a specific question about the object in your image.

#### Stage 1: What Is It? (Star or Galaxy?)

**What Happens**: The first neural network looks at your image and asks: "Is this a star or a galaxy?"

- A **star** is a point of light — like a bright dot
- A **galaxy** is a large structure with shape — like a spiral or oval blob

**How It Works**: The AI has learned patterns by studying thousands of examples. It recognizes the difference between round bright points (stars) and larger structured shapes (galaxies). It gives you confidence percentages for each.

**Output**: 
- "This is a **Galaxy** (95% confident)"
- "This is a **Star** (88% confident)"

---

#### Stage 2: How Bright Is It? (UGRIZ Magnitudes)

**What Happens**: The second neural network measures how bright the object appears through 5 different colored filters. This happens **regardless of whether it's a star or galaxy**.

Think of it like looking at the same object through 5 different colored glasses and writing down how bright it looks through each one:
- **U (ultraviolet)** — Ultra-violet colored glass (blue/purple)
- **G (green)** — Green colored glass
- **R (red)** — Red colored glass  
- **I (near-infrared)** — Deep red/infrared
- **Z (far-infrared)** — Far infrared (heat-sensing)

**Why This Matters**: These 5 measurements tell scientists a lot about the object:
- If it's brighter in the blue filters, it's hot and young
- If it's brighter in the red/infrared filters, it's cooler and older
- The difference between filters (called "color index") reveals temperature, composition, and distance

**Output**: 
```
u=19.5  g=18.2  r=17.8  i=17.3  z=16.9
```

---

#### Stage 3: What Shape Is It? (Morphology — Galaxy Only)

**What Happens**: If the first stage identified a **galaxy**, a third neural network analyzes its shape.

The AI classifies the galaxy into 4 common types based on its visual appearance:

1. **Elliptical** — Smooth, featureless, round or egg-shaped
2. **Spiral** — Disk with beautiful spiral arms (like a pinwheel)
3. **Edge-on** — A disk galaxy viewed from the side (looks like a thin line)
4. **Merger** — A chaotic, distorted galaxy (usually from two galaxies colliding)

**Why This Matters**: Galaxy shape tells astronomers about:
- How old the galaxy is
- What it's been through (collisions, interactions)
- How fast it's rotating
- Its history over billions of years

**Output** (if galaxy):
```
Elliptical: 15%  |  Spiral: 78%  |  Edge-on: 5%  |  Merger: 2%
Most likely: Spiral Galaxy
```

**Output** (if star):
```
No morphology data (only galaxies have shape)
```

---

### Final Results

After all 3 stages, you get a complete analysis:

```
Object Type: Galaxy
Star/Galaxy Confidence: Galaxy 95% | Star 5%

Brightness (UGRIZ magnitudes):
  u=19.5  g=18.2  r=17.8  i=17.3  z=16.9
  Color index (g-r): 0.4

Galaxy Shape:
  Elliptical: 15%  Spiral: 78%  Edge-on: 5%  Merger: 2%
  Most likely: Spiral
```

This rich data is then sent to the AI chatbot, which uses it to answer your questions in plain English.

---

## 🌐 Server.py Documentation

### Overview
`server.py` is a **Flask-based REST API backend** that:
1. Serves the web interface
2. Handles image uploads and runs inference
3. Manages Ollama LLM communication
4. Provides health checks and debugging endpoints

### Key Components

#### Cross-Origin Support (CORS)
The server is configured to accept requests from any domain, so the web interface can communicate with the backend from different machines or networks.

#### Ollama LLM Configuration
The server connects to Ollama (a local AI language model engine) to power the chat feature. You can configure three settings:

- **Where Ollama runs** (default: `localhost:11434`)  
  If Ollama is on a different machine, change this address
- **Which model to use** (default: `llama3.2`)  
  You can swap in different language models
- **Response timeout** (default: 120 seconds)  
  How long the server waits for AI responses

To customize, set environment variables before starting:
```bash
export OLLAMA_MODEL="mistral"  # Use a different model
export OLLAMA_TIMEOUT="180"    # Wait longer
python server.py
```

#### System Prompt
The LLM is guided by a system prompt that establishes its role as an astrophysicist:
```
You are AstroVision AI, an expert astrophysicist and data scientist 
specializing in the Sloan Digital Sky Survey (SDSS)...
```
Key responsibilities defined:
- Interpret UGRIZ photometric magnitudes
- Explain galaxy morphology
- Discuss star vs. galaxy classification
- Discuss color-magnitude diagrams and what they reveal
- Explain redshift and stellar classification

### Reference Galaxy Data
The server loads a database of ~500 known galaxies from the Sloan Digital Sky Survey (SDSS). This data is used to show context — when you analyze a galaxy, the frontend displays it on a chart alongside thousands of other known galaxies, so you can see where your galaxy falls in the bigger picture.

### How It Works: The Main Endpoints

**`GET /`** — Serves the main web interface when you visit the app

**`GET /example images/<filename>`** — Provides example astronomical images for the gallery demo

**`POST /upload`**
1. You upload an image
2. Server saves it temporarily
3. Runs all 3 stages of the ML pipeline (star/galaxy, magnitude, morphology)
4. Returns all the analysis results + reference galaxy data to display on the chart

**Response includes**:
- Object type (star or galaxy)
- Confidence percentages
- UGRIZ magnitude values
- Galaxy morphology (if galaxy)
- Reference data for context

**`POST /chat`** — Sends your question to the AI chatbot:
1. You type a question (e.g., "What does this galaxy tell us?")
2. Server bundles your message with the analysis results from the image
3. Sends it to Ollama (the local LLM) 
4. Ollama generates a response
5. Chat displays the AI's answer

The AI has access to the UGRIZ values, morphology data, etc, so it can give informed answers grounded in the actual measurements.

**Error Handling**: If Ollama is offline or times out, the chat gracefully tells you what happened instead of crashing.

**`GET /health`** — Quick check to see if the server and Ollama are running. Returns status info.

**`GET /debug`** — Shows detailed configuration information for troubleshooting (useful if something goes wrong)

### Startup Output
```
============================================================
AstroVision AI is running
============================================================
Local access:     http://127.0.0.1:5001
Network access:   http://192.168.1.100:5001
Debug info:       http://192.168.1.100:5001/debug
Health check:     http://192.168.1.100:5001/health

💡 CORS enabled for cross-origin requests
============================================================
```

---

## 💬 LLM Chat System

### Overview
The chat system enables users to ask questions about uploaded astronomical images using natural language. Powered by **Ollama**, an LLM framework, it provides accessible, educational explanations without requiring an internet connection or API keys.

### How It Works

#### 1. Ollama Setup
Ollama is a local LLM inference engine that runs on your machine. The system assumes:
- Ollama is installed and running (`ollama serve` in a terminal)
- A language model is downloaded (default: `llama3.2`)

**Install Ollama**:
```bash
# macOS (brew)
brew install ollama

# Linux / manual
curl -fsSL https://ollama.ai/install.sh | sh
```

**Start Ollama**:
```bash
ollama serve
# Listens on http://localhost:11434
```

**Download a Model**:
```bash
ollama pull llama3.2
# Or use another model: llama2, mistral, neural-chat, etc.
```

#### 2. Message Flow

```
User Message
    ↓
[Frontend builds request]
    ↓
{
  "message": "What does this galaxy tell us?",
  "astro_context": {...},
  "history": [...]
}
    ↓
[POST to /chat endpoint]
    ↓
[server.py builds message array]
    ↓
[System Prompt]
  ↓
[Context Block]
  ↓
[Conversation History]
  ↓
[Current User Message]
    ↓
[Call Ollama API: POST /api/chat]
    ↓
[Ollama generates response]
    ↓
[Return to frontend]
    ↓
[Display in chat]
```

#### 3. How the AI Gets Context

When you ask a question about your image it sends:

1. **Instructions** — "You are an astrophysicist expert on galaxy analysis...."
2. **The data** — All the measurements predicted by the visual models (star/galaxy classification, UGRIZ magnitudes, morphology)
3. **Your question** — What you actually typed in the chat
4. **Conversation history** — What you asked before and what the AI said (context length depends on chosen LLM)

This way, the AI has real data to reference when answering. It's not guessing — it's working with actual measurements from the neural networks. This makes the responses grounded in science rather than speculation.

#### 4. Error Handling

If something goes wrong (Ollama isn't running, times out, etc.), the system handles it gracefully:
- Instead of crashing, it shows you a helpful message in the chat
- You can see what the problem is and how to fix it
- The app continues to work (you can still view your analysis results)

### Configuring the LLM

#### Change Model
```bash
# Download a different model
ollama pull mistral
ollama pull neural-chat
ollama pull dolphin-mixtral
...

# Use it with AstroVision
export OLLAMA_MODEL="mistral"
python server.py
```

#### Change Ollama URL
If Ollama is running on a different machine:
```bash
export OLLAMA_BASE_URL="http://192.168.1.50:11434"
python server.py
```

#### Adjust Timeout
For slower machines or longer responses:
```bash
export OLLAMA_TIMEOUT="300"  # 5 minutes
python server.py
```

#### Customize System Prompt
Edit the `SYSTEM_PROMPT` variable in `server.py` to change the LLM's personality, instructions, or expertise areas.

---

## 🎨 Frontend & UI

### Technology Stack
- **HTML5** + **CSS3**
- **Bootstrap 5** for responsive grid layout
- **Chart.js** for UGRIZ magnitude visualization
- **Vanilla JavaScript** for interactivity

### Key UI Components within analysis page

**Top Section - Object Classification**:
- Large classification result: "🌀 GALAXY" or "⭐ STAR"
- Confidence percentages (galaxy %, star %)
- Donut chart showing probabilities

**Middle Section - UGRIZ Magnitudes**:
- 5 bar chart displaying magnitude values for u, g, r, i, z bands
- Each band labeled with wavelength info
- Color-coded bars (UV blue, green, red, IR red, FIR red)

**Bottom Section - Morphology** (if galaxy):
- 4 bars showing morphology probabilities
- Classes: Elliptical, Spiral, Edge-on, Merger
- Dominant class highlighted

**Context: CMD (Color Magnitude Diagram) Background**:
- Scatter plot showing reference galaxy positions
- User's galaxy plotted as larger point
- Axes: g magnitude (x), (g-r) color index (y)
- Shows how object compares to SDSS dataset

#### 5. Chat Interface
- **Chat History Panel**: Shows conversation messages
  - User messages: Right-aligned, blue bubbles
  - Assistant messages: Left-aligned, gray bubbles
- **Input Box**: Text input with send button
- **Status Indicator**: Shows if Ollama is available

### Responsive Design
- Mobile: Single-column layout, touch-friendly buttons
- Tablet: Two-column (image/results left, chat right)
- Desktop: Full layout with larger charts

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.8+
- Ollama (for LLM chat feature)

### Step 1: Clone/Download Repository
```bash
cd /path/to/Astrovision
```

### Step 2: Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 3: Install Python Dependencies
```bash
pip install -r requirements.txt
```

**Dependencies**:
- Flask — web framework
- flask-cors — cross-origin support
- torch, torchvision — deep learning
- Pillow — image processing
- numpy, opencv-python — array/image operations
- scikit-learn, pandas, matplotlib — data analysis
- requests — HTTP client for Ollama
- werkzeug — file upload handling

### Step 4: Install Ollama (Optional, for Chat)
```bash
# macOS
brew install ollama

# Or visit: https://ollama.ai
```

### Step 5: Download Models
```bash
# Ollama model
ollama pull llama3.2

# PyTorch models are already in Models/ directory
```

### Step 6: Verify Models Directory
Ensure this structure exists:
```
Models/
├── StarGalaxy_CLassifier/
│   └── checkpoint.pth
├── UGRIZ/
│   └── model.pth
├── EfficientNet_Morphology2/
│   └── model.pth
└── EfficientNet_Morphology1/
    └── model.pth
```

### Step 7: (Optional) Ensure references.csv
For CMD background visualization, the file should exist:
```
references.csv  (SDSS reference data)
```
If missing, the app will warn but still work.

---

## ▶️ Running the Application

### Terminal 1: Start Ollama (if using chat)
```bash
ollama serve
# Listens on http://localhost:11434
```

### Terminal 2: Start Flask Server
```bash
cd /path/to/Astrovision
source .venv/bin/activate
python server.py
```

**Output**:
```
============================================================
AstroVision AI is running
============================================================
Local access:     http://127.0.0.1:5001
Network access:   http://192.168.1.100:5001
...
```

### Terminal 3 (Optional): Open in Browser
```bash
open http://127.0.0.1:5001
# Or: http://localhost:5001
```

### Usage Flow
1. **Upload**: Click "Upload Your Image" or drag-drop
2. **Wait**: ML pipeline processes (~5-30 seconds depending on hardware)
3. **View Results**: See classification, magnitudes, morphology
4. **Chat**: Ask questions about the object in the chat panel
5. **Learn**: Read AI-generated insights

---

## Training the Models

### Data Collection
Images and photometry were retrieved from the Sloan Digital Sky Survey (SDSS) using their SQL database. A Python script queries the database, downloads 256×256 pixel image cutouts by coordinates (RA/Dec), and retrieves associated data including UGRIZ magnitudes and object IDs. All data is stored in CSV format with image filenames mapped to their corresponding astronomical properties.

### Training Configuration

All three CNN models were fine-tuned using the same data splitting approach:

**Data Split**: 80% training, 20% validation (no separate test set)  
**Splitting Method**: PyTorch's `random_split()` for random shuffling  
**Data Augmentation**: Random rotation (±20°), horizontal flip, color jitter (brightness, contrast, saturation)

| Model | Architecture | Batch Size | Epochs | Optimizer | Learning Rate | Loss Function |
|-------|--------------|-----------|--------|-----------|---------------|---------------|
| Star/Galaxy Classifier | EfficientNet-B0 | 32 | 15 | Adam | 1e-3 | CrossEntropyLoss |
| UGRIZ Regressor | ConvNeXt-Base | 32 | 30-45 | Adam | 1e-4 | MSELoss + BCELoss |
| Morphology (4-class) | EfficientNet-B0 | 64 | 30 | Adam | 5e-5 | BCELoss (weighted) |

The training loop loaded batches from the DataLoader, computed losses, backpropagated gradients, and saved checkpoints at each epoch. Early stopping was implemented with patience of 7 epochs to prevent overfitting. Model checkpoints were saved at each epoch with validation metrics tracked.

### Training Challenges

The models were prone to overfitting despite showing high validation accuracy, particularly the morphology classifier. The models learned specific patterns in the training data but didn't generalize well to new images with different conditions or viewing angles. This made them "easy to trick" — they would fail on images that differed even slightly from the training distribution.

Resolution came through iterative experimentation: testing multiple architectures, hyperparameter combinations, and data augmentation strategies. The final models chosen were those that performed best across diverse test scenarios rather than highest validation accuracy alone.

Files regarding training and the dataset used can be found under TestingImages/ExtraPyFiles.zip

The dataset retrieved and cleaned is high-quality for tasks like this, feel free to check it out.

---

## 📊 Data Files

### `references.csv`
SDSS reference galaxy data used for CMD background.

**Columns**: u, g, r, i, z (magnitude values)
**Rows**: ~500 reference galaxies
**Generated**: Color index (g-r) computed server-side
**Location**: Project root

### Model Checkpoints
Pre-trained PyTorch models stored in `Models/`:

| Model | Path | Type | Input | Output |
|-------|------|------|-------|--------|
| Star/Galaxy | `StarGalaxy_CLassifier/checkpoint.pth` | Classifier | 256×256 RGB | 2 classes |
| UGRIZ | `UGRIZ/model.pth` | Regressor | 256×256 RGB | 5 values |
| Morphology 4-class | `EfficientNet_Morphology2/model.pth` | Classifier | 256×256 RGB | 4 classes |
| Morphology 6-class | `EfficientNet_Morphology1/model.pth` | Classifier | 256×256 RGB | 6 classes |

### `uploads/`
Temporary directory storing uploaded images (auto-created by Flask).

### `example images/`
Sample astronomical images for gallery demo.

---

## 🎓 Technical Deep Dives

### Color-Magnitude Diagram (CMD)
Used by the UGRIZ regressor to understand magnitude relationships:
- **X-axis**: g magnitude (brightness in green band)
- **Y-axis**: g-r color index (temperature/age proxy)
- **Pattern**: Main sequence stars cluster in specific region
- **Application**: Context for user's analyzed object

### Center-Weight Mask
A technique used in two of the neural networks to focus on what matters. Imagine putting a special filter over the image that says "pay more attention to the center" and "ignore the edges." This helps the models focus on the actual galaxy or star rather than getting distracted by noise or artifacts at the edges of the image.

### Photometric Bands (UGRIZ)
From Sloan Digital Sky Survey (SDSS):
- **u** (3400Å): Ultraviolet, sensitive to hot stars
- **g** (4770Å): Green, visual band
- **r** (6230Å): Red, visual band
- **i** (7625Å): Near-infrared
- **z** (9134Å): Far-infrared

**Common Uses**:
- Color index (g-r): Indicates stellar temperature/age
- Magnitude ratios: Distinguish object types
- SED (spectral energy distribution) modeling

### Galaxy Morphology Types
Hubble sequence of galaxy classification:
- **Elliptical**: Smooth, featureless, shapes from circular to elongated (E0-E7)
- **Spiral**: Rotating disk with spiral arms (Sa, Sb, Sc)
- **Edge-on**: Spiral/disk galaxy viewed edge-on, appears as thin line
- **Merger/Irregular**: Distorted by tidal interactions or chaotic morphology

### Pipeline Workflow
The system follows a simple decision tree:

1. **First**: Ask "Is this a star or galaxy?" (always)
2. **Second**: Measure the brightness in 5 filters (always)
3. **Third**: If it's a galaxy, analyze its shape. If it's a star, skip this step.

This makes sense because stars don't have shape — they're just points of light — so there's no point analyzing morphology for a star.

---

## ✅ Checklist: Before Deployment

- [ ] Ollama installed and `ollama serve` tested
- [ ] Model downloaded: `ollama pull llama3.2`
- [ ] PyTorch models present in `Models/` directory
- [ ] `references.csv` file present (or commented out)
- [ ] Python dependencies installed: `pip install -r requirements.txt`
- [ ] Port 5001 available (or change in `server.py` line 321)
- [ ] Flask server starts without errors: `python server.py`
- [ ] Frontend loads at `http://localhost:5001`
- [ ] Test upload with sample image
- [ ] Test chat with Ollama running
- [ ] Check `/health` endpoint returns `ollama_available: true`

---

## 🤝 Contributing

To add features or improve AstroVision:

1. **New Model**: Add to `Models/` with checkpoint, update `test.py`
2. **New Endpoint**: Add route in `server.py`, update frontend
3. **UI Changes**: Modify `site.html` CSS/JS
4. **System Prompt**: Edit `SYSTEM_PROMPT` in `server.py` for LLM behavior

---

## Support

For issues or questions:
- Check `/debug` endpoint for configuration details
- Review error messages in browser console
- Ensure Ollama is running: `curl http://localhost:11434/api/tags`
- Check model availability: `ollama list`

---
