# Cortical Explorer

Cortical Explorer is an interactive web application for exploring regional cortical-thickness patterns across cognitive trajectories. It was developed as the foundation for a future deep-learning application studying Alzheimer’s disease and mild cognitive impairment.

The application allows users to rotate and explore a three-dimensional cortical representation, switch between cognitive groups, select cortical regions, compare group-level measurements, and save research annotations.

## Research Motivation

Mild cognitive impairment is a heterogeneous clinical state. Some individuals remain cognitively stable during observed follow-up, while others subsequently progress to dementia.

This application organizes participants into three groups:

- **CN:** Cognitively normal at baseline and not observed to progress to dementia in the available data.
- **MCI:** Mild cognitive impairment at baseline and not observed to progress to dementia in the available data.
- **AD Trajectory:** Dementia at baseline or subsequent progression to dementia.

The current version focuses on visual exploration rather than prediction. Future versions may incorporate deep-learning models for cognitive-status classification and progression-risk analysis.

## Data Being Visualized

The application visualizes regional cortical-thickness measurements derived from structural magnetic resonance imaging (MRI).

The prepared viewer dataset contains:

- 905 participants
- 68 cortical regions
- 34 regions in the left hemisphere
- 34 regions in the right hemisphere
- Cortical thickness measured in millimeters
- Aggregate regional statistics only

### Viewer Cohort

| Group | Participants |
|---|---:|
| CN | 477 |
| MCI | 261 |
| AD Trajectory | 167 |
| **Total** | **905** |

The cortical measurements follow the FreeSurfer `aparc` Desikan–Killiany cortical parcellation, consisting of 34 regions per hemisphere.

One observation containing a zero cortical-thickness measurement was excluded during data preparation.

## Privacy

The public application and GitHub repository do not contain participant identifiers or participant-level clinical records.

Only anonymous, aggregate regional statistics are included in:

`public/data/cortical_summary.json`

The original participant-level data are intentionally excluded from version control and are not required to run the application.

## Implemented Features

The application includes:

- An interactive three-dimensional cortical visualization
- Rotation, zooming, and panning controls
- Markers representing 68 cortical regions
- Selection of individual cortical regions
- A searchable cortical-region selector
- Switching between CN, MCI, and AD Trajectory groups
- Color encoding based on mean cortical thickness
- Regional mean, standard deviation, and sample-size summaries
- Charts comparing cortical thickness across the three groups
- Pairwise regional differences between cognitive groups
- Persistent region-specific research annotations
- A PlayStation 2-era visual interface with animated background elements

The current three-dimensional representation is an interactive cortical scaffold rather than an anatomically exact cortical surface. The 68 selectable markers correspond to the regional cortical-thickness measurements in the prepared dataset.

## Annotation

**Yes, annotation functionality was implemented.**

Users can select a cortical region, assign a tag, write a research note, and save or delete the annotation.

Annotations are stored in the browser using `localStorage`. This means that annotations persist after refreshing the application when it is opened using the same browser and device.

## Backend and Database

**No backend server or external database was implemented.**

The application is a client-side React application. Aggregate cortical-thickness data are loaded from a static JSON file, while annotations are stored locally in the user’s browser.

## Major Libraries and Frameworks

The project uses:

- **React** for the user interface
- **Vite** for development and production builds
- **Three.js** for three-dimensional graphics
- **React Three Fiber** for integrating Three.js with React
- **Drei** for Three.js helpers and camera controls
- **Recharts** for group-comparison charts
- **Python** for data preparation
- **pandas** for cleaning, merging, and aggregating the source data

## Running the Application

### Requirements

Before running the application, install:

- Node.js
- npm
- Git

### 1. Clone the repository

```bash
git clone https://github.com/ACastellana0629/cortical-thickness-explorer.git
```

### 2. Enter the project directory

```bash
cd cortical-thickness-explorer
```

### 3. Install the dependencies

```bash
npm install
```

### 4. Start the development server

```bash
npm run dev
```

Open the local address displayed in the terminal. The default Vite address is usually:

```text
http://localhost:5173/
```

To stop the development server, press `Control + C` in the terminal running the server.

## Additional Commands

Create a production build:

```bash
npm run build
```

Check the project with the configured linter:

```bash
npm run lint
```

Preview the production build locally:

```bash
npm run preview
```

## Data Preparation

The repository includes the anonymous, aggregate JSON file required to run the viewer. Access to the original participant-level data is not required to launch the application.

The data-preparation script is located at:

`data-preparation/prepare_viewer_data.py`

If an authorized user has the original source files inside the ignored `private_data/` directory, the aggregate viewer dataset can be regenerated with:

```bash
python data-preparation/prepare_viewer_data.py
```

The script:

1. Loads the left- and right-hemisphere cortical-thickness files.
2. Merges the measurements with the clinical data.
3. Assigns participants to the CN, MCI, or AD Trajectory group.
4. Excludes observations containing zero cortical-thickness measurements.
5. Calculates aggregate statistics for all 68 cortical regions.
6. Exports the anonymous results to `public/data/cortical_summary.json`.

The `private_data/` directory and original participant-level datasets are excluded from GitHub through `.gitignore`.

## Limitations

- The three-dimensional visualization is a prototype cortical scaffold and does not show exact anatomical parcel boundaries.
- The displayed comparisons are descriptive and should not be interpreted as clinical diagnoses.
- The public viewer contains aggregate statistics rather than participant-level records.
- Annotations are browser-specific and are not synchronized across devices.
- The application does not currently include a trained predictive model.

## Future Work

Possible future developments include:

- Replacing the cortical scaffold with an anatomically accurate brain surface
- Mapping each measurement to its exact cortical parcel
- Adding cortical-thickness and intrinsic neural timescale comparisons
- Incorporating deep-learning classification models
- Studying progression risk among participants with mild cognitive impairment
- Adding a secure backend for authenticated annotation storage
- Supporting annotation export and collaboration across devices

## Author

**Andres Castellanos**

Ph.D. Student in Data Science  
University of Virginia