# Cortical Explorer

Cortical Explorer is an interactive neuroimaging application for exploring regional cortical-thickness patterns and generating neural-network predictions across cognitive trajectories.

The project began as a three-dimensional data viewer for Homework 1 and was extended into an AI-enabled application for Homework 2. Users can explore regional cortical-thickness summaries, save research annotations, select an aggregate demonstration profile, and obtain a prediction from a trained PyTorch neural network.

> **Important:** This application is an educational and research prototype. Its predictions are not medical diagnoses or clinical recommendations.

## Research Motivation

Mild cognitive impairment is a heterogeneous clinical state. Some individuals remain cognitively stable during observed follow-up, while others subsequently progress to dementia.

The application organizes the available observations into three cognitive-trajectory groups:

- **CN:** Cognitively normal at baseline and not observed to progress to dementia in the available data.
- **MCI:** Mild cognitive impairment at baseline and not observed to progress to dementia in the available data.
- **AD Trajectory:** Dementia at baseline or subsequent progression to dementia.

The application examines whether regional cortical-thickness measurements contain enough information for a small neural network to distinguish these three trajectories.

## Data Being Visualized

The application visualizes regional cortical-thickness measurements derived from structural magnetic resonance imaging.

The prepared viewer and modeling cohort contains:

- 905 observations
- 68 cortical regions
- 34 regions in the left hemisphere
- 34 regions in the right hemisphere
- Cortical thickness measured in millimeters

### Cohort Counts

| Group | Observations |
|---|---:|
| CN | 477 |
| MCI | 261 |
| AD Trajectory | 167 |
| **Total** | **905** |

The cortical measurements follow the FreeSurfer `aparc` Desikan-Killiany cortical parcellation, consisting of 34 regions per hemisphere.

One observation containing a zero cortical-thickness measurement was excluded during quality control.

## Neural-Network Task

The prediction task is three-class classification:

```text
68 cortical-thickness measurements
                ↓
        PyTorch neural network
                ↓
       CN, MCI, or AD Trajectory
```

The model receives one value for each of the 68 cortical regions and returns probabilities for the three cognitive-trajectory groups.

## Model Architecture

The model is a small multilayer perceptron implemented with PyTorch.

```text
Input layer:        68 cortical-thickness features
First hidden layer: 32 units + ReLU + dropout
Second hidden layer:16 units + ReLU + dropout
Output layer:       3 class logits
```

Additional model details:

- **Trainable parameters:** 2,787
- **Activation function:** ReLU
- **Dropout:** 0.20
- **Weight initialization:** Xavier uniform initialization
- **Loss function:** Cross-entropy loss
- **Optimizer:** AdamW
- **Learning rate:** 0.001
- **Batch size:** 64
- **Maximum epochs:** 300
- **Early-stopping patience:** 40 epochs
- **Random seed:** 42

PyTorch automatically calculates parameter gradients through backpropagation. AdamW uses those gradients to update the model parameters after each mini-batch.

## Training and Evaluation

The 905 observations were separated using stratified sampling:

| Dataset | Observations |
|---|---:|
| Training | 633 |
| Validation | 136 |
| Test | 136 |

The feature scaler was fitted using only the training data. The same fitted scaling parameters were then applied to the validation and test sets to avoid preprocessing leakage.

Early stopping monitored validation loss. The best model occurred at epoch 36, and those weights were restored for the final evaluation.

## Model Performance

| Model | Test Accuracy | Test Error |
|---|---:|---:|
| Majority-class baseline | 52.94% | 47.06% |
| Balanced logistic-regression benchmark | 47.79% | 52.21% |
| **PyTorch MLP** | **54.41%** | **45.59%** |

Additional MLP results:

| Metric | Result |
|---|---:|
| Training accuracy | 73.46% |
| Validation accuracy | 60.29% |
| Test accuracy | 54.41% |
| Test macro F1 | 0.430 |
| Best epoch | 36 |

The MLP achieved a lower test error than the logistic-regression benchmark. However, its improvement over the majority-class baseline was small.

### Test Confusion Matrix

Rows represent the observed class, and columns represent the predicted class.

| Observed \ Predicted | CN | MCI | AD Trajectory |
|---|---:|---:|---:|
| CN | 60 | 8 | 4 |
| MCI | 30 | 4 | 5 |
| AD Trajectory | 14 | 1 | 10 |

### Class-Specific Performance

| Group | Precision | Recall | F1 |
|---|---:|---:|---:|
| CN | 0.577 | 0.833 | 0.682 |
| MCI | 0.308 | 0.103 | 0.154 |
| AD Trajectory | 0.526 | 0.400 | 0.455 |

The model identifies CN most reliably and detects some AD Trajectory patterns. It performs poorly on MCI, which frequently overlaps with CN.

This limitation is scientifically plausible because MCI is heterogeneous, the class distribution is imbalanced, and cortical thickness alone may not provide enough information to cleanly distinguish stable MCI from normal aging or dementia trajectories.

## Application Workflow

The complete application pipeline is:

```text
User selects an aggregate profile
                ↓
React sends 68 measurements through HTTP
                ↓
FastAPI validates and standardizes the features
                ↓
PyTorch performs forward propagation
                ↓
Softmax converts the logits into probabilities
                ↓
React displays the prediction and probabilities
```

The user does not need to manually run a training script or open a Jupyter notebook to obtain a prediction.

## Implemented Features

### Interactive Cortical Viewer

- Three-dimensional cortical scaffold
- Rotation, zooming, and panning controls
- Markers representing 68 cortical regions
- Selection of individual cortical regions
- Cortical-region dropdown
- Switching between CN, MCI, and AD Trajectory groups
- Color encoding based on mean cortical thickness
- Regional mean, standard deviation, and sample-size summaries
- Group-comparison charts
- Pairwise regional differences

The current three-dimensional representation is a cortical scaffold rather than an anatomically exact cortical surface. The 68 markers correspond to the measurements used by the application.

### Neural Prediction System

- Three aggregate demonstration profiles
- FastAPI prediction endpoint
- PyTorch model inference
- Predicted cognitive-trajectory class
- CN, MCI, and AD Trajectory probabilities
- Validation of all 68 required inputs
- Research-use disclaimer
- Model-performance dashboard

The demonstration profiles are class-level aggregate prototypes constructed from training-set means. They are not real participant records.

### Training Monitor

The graphical interface displays:

- Training-loss curve
- Validation-loss curve
- Training-accuracy curve
- Validation-accuracy curve
- MLP test accuracy
- Logistic-regression benchmark accuracy
- Validation accuracy
- Test macro F1
- Best training epoch
- Prediction results

The displayed training history is saved from the completed training run rather than generated through live retraining in the browser.

### Visual Design

The application uses a PlayStation 2-era system-interface aesthetic with:

- Indigo fog
- Floating translucent cubes
- Scanlines
- Colored signal lights
- Translucent interface panels
- Neon interface accents

## Annotation

**Annotation was implemented.**

Users can select a cortical region, assign a tag, write a research note, and save or delete the annotation.

Annotations are stored using browser `localStorage`. They persist after refreshing the application when it is opened using the same browser and device.

## Backend and Database

**Backend implemented:** Yes  
**External database implemented:** No

The FastAPI backend:

- Loads the saved PyTorch model
- Loads the saved feature order and scaling parameters
- Validates the submitted cortical-thickness profile
- Standardizes the 68 measurements
- Performs neural-network inference
- Returns the predicted class and class probabilities

The project does not require a database. Annotations remain in browser `localStorage`, and predictions are not permanently stored.

## Privacy

The public application and GitHub repository do not contain participant identifiers or participant-level clinical records.

The following are excluded from version control:

- Original clinical records
- Original cortical-thickness files
- Participant identifiers
- Participant-level modeling tables
- Private research-data directories
- Temporary training checkpoints

The public application includes:

- Anonymous regional summaries
- Aggregate model-performance metrics
- Aggregate demonstration profiles
- Feature-scaling parameters
- Trained model weights
- Training and application source code

Public JSON files were checked for ADNI-style participant identifiers before publication.

## Major Libraries and Frameworks

### Frontend

- React
- Vite
- Three.js
- React Three Fiber
- Drei
- Recharts

### Machine Learning and Data Preparation

- Python
- PyTorch
- pandas
- NumPy
- scikit-learn
- Matplotlib

### Backend

- FastAPI
- Uvicorn
- Pydantic

## Repository Structure

```text
cortical-thickness-explorer/
├── backend/
│   └── app.py
├── data-preparation/
│   └── prepare_viewer_data.py
├── models/
│   ├── cortical_mlp.pt
│   └── preprocessing.json
├── private_data/                 # Ignored by Git
├── public/
│   └── data/
│       ├── cortical_summary.json
│       ├── demo_profiles.json
│       ├── model_metrics.json
│       └── training_curves.png
├── src/
│   ├── components/
│   │   ├── AnnotationPanel.jsx
│   │   ├── PredictionPanel.jsx
│   │   └── SystemBackdrop.jsx
│   └── App.jsx
├── training/
│   ├── data.py
│   ├── model.py
│   └── train.py
├── package.json
├── requirements-hw2.txt
└── README.md
```

## Running the Complete Application

The application requires two local servers:

1. The React frontend
2. The FastAPI prediction backend

### Requirements

Install:

- Git
- Node.js 20 or newer
- npm
- Conda or another Python environment manager

Python 3.11 is recommended for the backend.

### 1. Clone the Repository

```bash
git clone https://github.com/ACastellana0629/cortical-thickness-explorer.git
cd cortical-thickness-explorer
```

### 2. Install the Frontend Dependencies

```bash
npm install
```

### 3. Create the Python Environment

Using Conda:

```bash
conda create -n cortical_hw2 python=3.11 -y
conda activate cortical_hw2
```

Install the required Python packages:

```bash
python -m pip install -r requirements-hw2.txt
```

### 4. Start the FastAPI Backend

In the first terminal:

```bash
conda activate cortical_hw2
uvicorn backend.app:app --reload --port 8000
```

The backend will run at:

```text
http://127.0.0.1:8000
```

Useful backend pages:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
```

### 5. Start the React Frontend

Open a second terminal in the project directory:

```bash
npm run dev
```

Open the address displayed by Vite, normally:

```text
http://localhost:5173/
```

Both servers must remain running for neural-network predictions to work.

The cortical viewer and saved model-performance dashboard use static public files, but the prediction button requires the FastAPI backend.

## Additional Commands

Run the frontend linter:

```bash
npm run lint
```

Create a production frontend build:

```bash
npm run build
```

Preview the production build:

```bash
npm run preview
```

Test the private training-data loader:

```bash
conda activate cortical_hw2
python training/data.py
```

Retrain the model:

```bash
conda activate cortical_hw2
python training/train.py
```

Retraining requires authorized access to the original source data inside the ignored `private_data/` directory.

## Data Preparation

The viewer-data preparation script is:

```text
data-preparation/prepare_viewer_data.py
```

For authorized users with the original source files in `private_data/`, the aggregate viewer dataset can be regenerated with:

```bash
python data-preparation/prepare_viewer_data.py
```

The script:

1. Loads the left- and right-hemisphere cortical-thickness files.
2. Matches the measurements with the clinical data.
3. Assigns observations to CN, MCI, or AD Trajectory.
4. Excludes observations containing zero cortical-thickness values.
5. Calculates aggregate statistics for all 68 regions.
6. Exports the anonymous results to `public/data/cortical_summary.json`.

The public application does not require the private source data because the trained model, scaling parameters, and safe demonstration profiles are included in the repository.

## Limitations

- The three-dimensional visualization does not show exact anatomical parcel boundaries.
- The model was evaluated on a relatively small and imbalanced dataset.
- The model performs poorly on the MCI class.
- Cortical thickness alone does not capture every biological or clinical factor related to cognitive decline.
- The aggregate demonstration profiles are simplified class prototypes.
- The predictions are not calibrated or validated for clinical use.
- The application is not a diagnostic device.
- Annotations are browser-specific and do not synchronize across devices.
- The FastAPI backend must be started separately from the React frontend.

## Future Work

Potential improvements include:

- Adding intrinsic neural timescale measurements
- Comparing cortical thickness, neural timescales, and combined features
- Improving class balance during training
- Evaluating class-weighted loss or sampling strategies
- Performing repeated cross-validation
- Calibrating predicted probabilities
- Testing demographic performance differences
- Replacing the scaffold with an anatomically accurate cortical surface
- Adding a secure database for authenticated annotations
- Deploying the frontend and backend as a hosted application

## AI-Assisted Development Disclosure

ChatGPT was used to assist with:

- Planning the application architecture
- Drafting portions of the PyTorch, FastAPI, React, and CSS implementations
- Debugging environment and dependency issues
- Organizing the training and evaluation pipeline
- Reviewing privacy protections
- Drafting documentation

The project author selected the research task, supplied and interpreted the data, executed and tested the code, evaluated the model, reviewed the generated implementation, and is responsible for understanding and explaining the final system.

## Author

**Andres Castellanos**

Ph.D. Student in Data Science  
University of Virginia