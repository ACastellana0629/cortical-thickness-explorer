# Cortical Explorer

Cortical Explorer is an interactive web application for exploring regional cortical-thickness patterns across cognitive trajectories. It was developed as the foundation for a future deep-learning application studying Alzheimer’s disease and mild cognitive impairment.

The application allows users to rotate and explore a three-dimensional cortical representation, switch between cognitive groups, select cortical regions, compare group-level measurements, and save research annotations.

## Research Motivation

Mild cognitive impairment is a heterogeneous clinical state. Some individuals remain cognitively stable during observed follow-up, while others subsequently progress to dementia.

This application organizes participants into three groups:

- **CN:** Cognitively normal at baseline and not observed to progress to dementia in the available data.
- **MCI:** Mild cognitive impairment at baseline and not observed to progress to dementia in the available data.
- **AD Trajectory:** Dementia at baseline or subsequent progression to dementia.

The current version focuses on visual exploration rather than prediction. Future versions will incorporate deep-learning models for cognitive-status classification and progression-risk analysis.

## Data

The application visualizes regional cortical-thickness measurements derived from structural MRI.

The prepared viewer dataset contains:

- 905 participants
- 68 cortical regions
- 34 regions in the left hemisphere
- 34 regions in the right hemisphere
- Cortical thickness measured in millimeters
- Aggregate statistics only

Viewer cohort counts:

| Group | Participants |
| --- | ---: |
| CN | 477 |
| MCI | 261 |
| AD Trajectory | 167 |
| **Total** | **905** |

The cortical measurements follow the FreeSurfer `aparc` cortical parcellation, with 34 regions per hemisphere.

## As for Privacy

The public application and GitHub repository do not contain participant identifiers or participant-level clinical records.

Only aggregate regional statistics are included in:

```text
public/data/cortical_summary.json