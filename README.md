
# EEG Motor Imagery Decoding with MNE-Python

An exploratory analysis of noninvasive EEG motor imagery using the public [PhysioNet EEG Motor Movement/Imagery dataset](https://physionet.org/content/eegmmidb/1.0.0/).

This project implements a multi-participant EEG preprocessing and analysis pipeline to investigate sensorimotor oscillations and classify imagined left- versus right-fist movements using Common Spatial Patterns (CSP) and Linear Discriminant Analysis (LDA).

The project includes a reproducible Python pipeline and a Jupyter Notebook for step-by-step exploration and visualization.

## Methods

### 1. Dataset

- **Source:** PhysioNet EEG Motor Movement/Imagery Dataset (EEGBCI)
- **Participants analyzed:** 30
- **Recordings:** Runs 4, 8, and 12 for each participant (90 EDF files)
- **Task:** Imagined left- and right-fist movements
- **Total trials:** 1,350

Runs 4, 8, and 12 contain repeated motor imagery tasks. Runs 3, 7, and 11 correspond to actual motor execution and are not analyzed.

### 2. EEG Preprocessing

Preprocessing is implemented using MNE-Python:

- Standardized electrode names and assigned electrode locations
- Applied a 1–40 Hz FIR bandpass filter
- Applied average EEG referencing
- Extracted event-locked epochs from −1 to 4 seconds relative to movement imagery cues
- Implemented optional independent component analysis (ICA) for ocular artifact removal

ICA uses frontal EEG electrodes (Fp1/Fp2) as proxies for detecting eye-related components because dedicated EOG recordings are unavailable. This method is heuristic and requires validation.

### 3. Time-Frequency Analysis

- Applied Morlet wavelets to characterize mu (8–12 Hz) and beta (13–30 Hz) oscillatory activity
- Calculated power changes relative to a pre-cue baseline (−0.8 to −0.2 seconds)
- Investigated event-related desynchronization (ERD) over sensorimotor regions
- Compared contralateral and ipsilateral oscillatory activity during motor imagery

### 4. Motor Imagery Classification

- Filtered EEG signals to 8–30 Hz
- Extracted activity from 0.5–3.5 seconds after movement imagery cues
- Applied CSP with four components for spatial feature extraction
- Used LDA to classify imagined left- versus right-fist movements
- Evaluated classification using leave-one-run-out cross-validation within each participant

CSP is fitted separately within each training fold. Each recording run serves as the held-out test set once.

This evaluates generalization to unseen recording runs from the same participant, not to previously unseen participants.

## Results

### Decoding Performance

Across 30 participants, the pipeline achieved:

| Metric | Result |
|---|---|
| Mean decoding accuracy | 60.4% |
| Median decoding accuracy | 56.7% |
| Accuracy range | 44.4%–100.0% |
| Participants above 50% accuracy | 25/30 |
| Total correctly classified trials | 816/1,350 |

These results were obtained using the standard preprocessing pipeline without ICA artifact removal.

![Participant-level decoding accuracy](figures/held_out_run_accuracy.png)

![Combined decoding confusion matrix](figures/held_out_run_confusion.png)

Classification performance varied substantially between participants. The results demonstrate the feasibility of decoding motor imagery from EEG recordings using CSP-LDA, although performance was near chance for many individuals.

The 50% reference represents balanced-class chance performance and does not establish statistical significance for individual participants.

### ICA Comparison

The analysis was repeated with optional ICA artifact removal enabled.

| Preprocessing | Mean decoding accuracy |
|---|---|
| Without ICA | 60.4% |
| With ICA | 59.9% |

Overall decoding performance changed only slightly after automated ICA cleaning, although individual participants showed larger differences.

The ICA analysis is preliminary. Some recordings produced convergence warnings, and automatically selected components have not been independently validated. These results should not be interpreted as evidence that all non-neural artifacts were removed.

### Sensorimotor Oscillations

Time-frequency analysis revealed changes in mu/beta power during motor imagery. Exploratory laterality analysis showed a mean contralateral-minus-ipsilateral power change of approximately −5.83 percentage points across participants.

Negative values indicate relatively stronger contralateral power suppression. These exploratory observations do not establish that EEG lateralization explains individual differences in decoding accuracy.

## Repository Structure

```text
eeg-motor-imagery-decoding/
├── eeg_pipeline.py
├── eeg_analysis.ipynb
├── download_eeg_data.py
├── README.md
├── requirements.txt
├── requirements-notebook.txt
├── figures/
└── tests/
```

The `data/`, `results/`, and `results_ica/` folders are excluded from version control. Raw EDF recordings are not included in the repository.

## Running the Pipeline

### Installation

Python 3.10 or later is recommended.

From the repository folder, create a Python virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Alternatively, activate an existing compatible Conda environment.

### Downloading EEG Data

The pipeline checks the `data/` directory for existing EDF recordings before downloading missing files.

For example, a participant's recordings can be organized as:

```text
data/
└── S001/
    ├── S001R04.edf
    ├── S001R08.edf
    └── S001R12.edf
```

To download recordings for multiple participants in parallel:

```bash
python download_eeg_data.py --start 1 --end 30 --workers 5
```

Already-downloaded files are skipped.

### Analyze 30 Participants

To analyze participants 1–30 using locally downloaded recordings:

```bash
python eeg_pipeline.py --subjects $(seq 1 30) --local-only
```

To download any missing files automatically, omit `--local-only`.

### Run With ICA Artifact Removal

```bash
python eeg_pipeline.py --subjects $(seq 1 30) --local-only --ica --output-dir results_ica
```

The default pipeline does not apply ICA. The `--ica` option enables automated candidate ocular component removal.

### Run Tests

```bash
python -m pip install pytest
python -m pytest -q
```

## Jupyter Notebook

The accompanying [eeg_analysis.ipynb](eeg_analysis.ipynb) provides a step-by-step exploration of the analysis.

The notebook includes:

1. Raw EEG visualization
2. Filtering and preprocessing
3. Power spectral density analysis
4. Morlet wavelet time-frequency analysis
5. CSP-LDA motor imagery classification
6. Multi-participant decoding comparisons
7. Exploratory analysis of ERD lateralization and decoding performance

**Participant 2 (S002)** is used as the illustrative example for individual EEG visualizations. This participant achieved 91.1% decoding accuracy without ICA, substantially above the group mean, and should not be considered representative of the full cohort.

The notebook's group-level sections use the results generated by the Python pipeline for all 30 participants.

### Launch Jupyter

```bash
python -m pip install -r requirements-notebook.txt
jupyter lab eeg_analysis.ipynb
```

The notebook requires the dataset files for individual recordings and the pipeline-generated results for group comparisons. Saved notebook outputs can be viewed directly on GitHub.

## Outputs

Running the pipeline generates:

```text
results/
├── subject_results.csv
├── summary.json
└── figures/
    ├── held_out_run_accuracy.png
    ├── held_out_run_confusion.png
    ├── example_motor_imagery_tfr.png
    └── contralateral_erd.png
```

These files contain participant-level decoding results, group summaries, and EEG visualizations. The output directory is excluded from GitHub; selected figures are provided separately in `figures/`.

## Limitations

- This project analyzes offline motor imagery EEG recordings, not real-time BCI control.
- Classification is performed within participants and does not demonstrate generalization to new individuals.
- Automated ocular artifact detection uses frontal EEG channels rather than dedicated EOG recordings.
- Some ICA fits produced convergence warnings, and rejected components require further validation.
- Participants with unusually high decoding accuracy may require additional checks for artifacts or other confounds.
- The ERD and laterality analyses are exploratory and should not be interpreted as causal explanations of decoding performance.

This project is intended as an exploratory analysis and demonstration of EEG signal processing and decoding methods, not a clinically validated BCI.

## Data Source and References

**Dataset:** [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/)

**Related references:**

- Schalk G, McFarland DJ, Hinterberger T, Birbaumer N, Wolpaw JR. BCI2000: a general-purpose brain-computer interface (BCI) system. *IEEE Trans Biomed Eng.* 2004;51(6):1034–1043. [DOI](https://doi.org/10.1109/TBME.2004.827072)
- Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. *Circulation.* 2000. [DOI](https://doi.org/10.1161/01.CIR.101.23.e215)
- [MNE-Python Motor Imagery CSP Example](https://mne.tools/stable/auto_examples/decoding/decoding_csp_eeg.html)

## Development Note

This project involved AI-assisted coding and documentation. Scientific methods, preprocessing choices, and interpretations require independent review before use in formal research.

## License

No software license has currently been assigned.
