"""Reproducible exploratory motor-imagery EEG analysis (PhysioNet EEGBCI).

The default analysis uses runs 4, 8, 12: *imagined* left/right fist movement.
All decoding scores use held-out recording runs within each participant.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Safe for command-line/headless execution.
import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.datasets import eegbci
from mne.decoding import CSP
from mne.preprocessing import ICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import Pipeline

LOGGER = logging.getLogger("eeg_motor_imagery")
RUNS = (4, 8, 12)  # Imagined left/right fist; 3/7/11 are actual movements.
FREQS = np.arange(8.0, 31.0, 2.0)


def fetch_recordings(subject: int, data_dir: Path, local_only: bool = False) -> list[tuple[int, Path]]:
    """Find previously downloaded EEGBCI files, then optionally fetch missing runs.

    Accepted layouts (and arbitrarily nested folders below data_dir):
      data/S004/S004R04.edf
      data/S004R04.edf
      data/MNE-eegbci-data/files/eegmmidb/1.0.0/S004/S004R04.edf
    """
    required = {run: f"S{subject:03d}R{run:02d}.edf" for run in RUNS}
    existing: dict[int, Path] = {}
    if data_dir.is_dir():
        # Search once for all runs; match filenames exactly (not partial IDs).
        wanted = {name.casefold(): run for run, name in required.items()}
        for path in sorted(data_dir.rglob("*")):
            if not path.is_file():
                continue
            run = wanted.get(path.name.casefold())
            if run is not None:
                if run in existing:
                    LOGGER.warning(
                        "Multiple copies of %s found; using %s (ignoring %s)",
                        required[run], existing[run], path,
                    )
                else:
                    existing[run] = path
                    LOGGER.info("Using existing EDF: %s", path)

    missing = [run for run in RUNS if run not in existing]
    if missing and local_only:
        missing_names = ", ".join(required[run] for run in missing)
        raise FileNotFoundError(
            f"Missing EDF files for subject {subject}: {missing_names}. "
            f"Place them anywhere inside {data_dir} or omit --local-only "
            "to download the missing runs from PhysioNet."
        )

    if missing:
        LOGGER.info("Missing runs for participant %03d: %s; downloading", subject, missing)
        paths = eegbci.load_data(
            subject,
            missing,
            path=str(data_dir.resolve()),
            update_path=False,
            verbose="ERROR",
        )
        for path in paths:
            match = re.search(r"R(\d{2})\.edf$", str(path), flags=re.IGNORECASE)
            if not match:
                raise ValueError(f"Cannot determine run number from {path}")
            existing[int(match.group(1))] = Path(path)

    if set(existing) != set(RUNS):
        raise ValueError(f"Expected runs {RUNS}; found {sorted(existing)}")
    return [(run, existing[run]) for run in RUNS]


def read_recording(path: Path) -> mne.io.BaseRaw:
    """Read and standardize EEGBCI electrodes to standard scalp positions."""
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    eegbci.standardize(raw)
    raw.set_montage("standard_1005", match_case=False, on_missing="warn", verbose="ERROR")
    return raw


def preprocess(raw: mne.io.BaseRaw, use_ica: bool = False) -> mne.Epochs:
    """Filter continuous signals; optionally reject candidate ocular ICs; epoch.

    ICA uses Fp1/Fp2 *EEG* channels as imperfect ocular proxies. Without EOG
    ground truth these exclusions are heuristic, not validated artifact labels.
    """
    raw = raw.copy()
    raw.filter(1.0, 40.0, fir_design="firwin", verbose="ERROR")
    # A separate 60-Hz notch is unnecessary with a 40-Hz low-pass cutoff.
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")

    if use_ica:
        ica = ICA(
            n_components=0.99,
            method="fastica",  # Does not require the optional Picard package.
            max_iter=500,
            random_state=42,
            verbose="ERROR",
        )
        ica.fit(raw, decim=3, verbose="ERROR")
        bad_components: set[int] = set()
        for channel in ("Fp1", "Fp2"):
            if channel in raw.ch_names:
                candidate_indices, _ = ica.find_bads_eog(
                    raw, ch_name=channel, verbose="ERROR"
                )
                bad_components.update(candidate_indices)
        if not ("Fp1" in raw.ch_names or "Fp2" in raw.ch_names):
            LOGGER.warning("No frontal proxy channels found: ICA did not remove components")
        ica.exclude = sorted(bad_components)
        LOGGER.info("Removed candidate ocular ICA components: %s", ica.exclude)
        if ica.exclude:
            ica.apply(raw, verbose="ERROR")

    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    for label in ("T1", "T2"):
        if label not in event_id:
            raise ValueError(f"Missing event {label}; available events: {list(event_id)}")
    epochs = mne.Epochs(
        raw,
        events,
        event_id={label: event_id[label] for label in ("T1", "T2")},
        tmin=-1.0,
        tmax=4.0,
        baseline=None,  # Preserve continuous spectral power for CSP/TFR.
        preload=True,
        reject_by_annotation=True,
        verbose="ERROR",
    )
    if len(epochs) == 0:
        raise ValueError("No usable trials remain after epoching")
    return epochs


def load_subject(subject: int, data_dir: Path, use_ica: bool, local_only: bool = False) -> tuple[mne.Epochs, np.ndarray]:
    """Pool trials from all three runs, preserving run labels for validation."""
    run_epochs = []
    groups = []
    for run, file_path in fetch_recordings(subject, data_dir, local_only=local_only):
        ep = preprocess(read_recording(file_path), use_ica=use_ica)
        run_epochs.append(ep)
        groups.extend([run] * len(ep))
        LOGGER.info("Subject %03d run %02d: %d epochs", subject, run, len(ep))
    return mne.concatenate_epochs(run_epochs, verbose="ERROR"), np.asarray(groups)


def condition_tfr(epochs: mne.Epochs, condition: str) -> mne.time_frequency.AverageTFR:
    """Morlet power, in percentage points relative to pre-cue baseline."""
    tfr = epochs[condition].compute_tfr(
        method="morlet",
        freqs=FREQS,
        n_cycles=FREQS / 2,
        picks=["C3", "C4"],
        average=True,
        return_itc=False,
        verbose="ERROR",
    )
    tfr.apply_baseline(baseline=(-0.8, -0.2), mode="percent", verbose="ERROR")
    # MNE's 'percent' mode returns a *fractional* change (0.2 means +20%).
    # Convert to actual percentage points before plotting/reporting ERD.
    tfr.data *= 100.0
    return tfr


def mean_erd(tfr: mne.time_frequency.AverageTFR, channel: str) -> float:
    """Mean mu/beta percent power change, 0.5 to 3.5 s after cue onset."""
    ch = tfr.ch_names.index(channel)
    time_mask = (tfr.times >= 0.5) & (tfr.times <= 3.5)
    freq_mask = (tfr.freqs >= 8) & (tfr.freqs <= 30)
    if not np.any(time_mask) or not np.any(freq_mask):
        raise ValueError("ERD analysis window is empty")
    return float(tfr.data[ch][np.ix_(freq_mask, time_mask)].mean())


def laterality_effect(left_tfr: mne.time_frequency.AverageTFR,
                       right_tfr: mne.time_frequency.AverageTFR) -> float:
    """Contra-minus-ipsi mean ERD; negative means stronger contralateral ERD."""
    left = mean_erd(left_tfr, "C4") - mean_erd(left_tfr, "C3")
    right = mean_erd(right_tfr, "C3") - mean_erd(right_tfr, "C4")
    return (left + right) / 2.0


def sign_flip_pvalue(effects: np.ndarray, seed: int = 42) -> tuple[float | None, str]:
    """One-sided sign-flip test on subject-level effects (exploratory)."""
    effects = np.asarray(effects, dtype=float)
    if len(effects) < 2:
        return None, "not estimated (fewer than two participants)"
    observed = effects.mean()
    if len(effects) <= 16:
        null_means = np.fromiter(
            (np.mean(effects * signs)
             for signs in itertools.product((-1.0, 1.0), repeat=len(effects))),
            dtype=float,
        )
        return float(np.mean(null_means <= observed + 1e-12)), "exact sign-flip"
    rng = np.random.default_rng(seed)
    simulated = (rng.choice([-1.0, 1.0], size=(20_000, len(effects))) * effects).mean(axis=1)
    p = (np.count_nonzero(simulated <= observed + 1e-12) + 1) / (len(simulated) + 1)
    return float(p), "Monte Carlo sign-flip (20,000 samples)"


def make_decoder() -> Pipeline:
    """CSP is fitted *within each training fold* to avoid label leakage."""
    return Pipeline([
        ("csp", CSP(n_components=4, reg=None, log=True, norm_trace=False)),
        ("lda", LinearDiscriminantAnalysis()),
    ])


def prepare_decoding_data(epochs: mne.Epochs) -> tuple[np.ndarray, np.ndarray]:
    """Mu/beta-band trials, without the pre-cue baseline period."""
    band_epochs = epochs.copy().filter(8.0, 30.0, fir_design="firwin", verbose="ERROR")
    band_epochs.crop(tmin=0.5, tmax=3.5)
    x = band_epochs.get_data(picks="eeg", copy=True)
    mapping = {band_epochs.event_id["T1"]: 0, band_epochs.event_id["T2"]: 1}
    y = np.asarray([mapping[int(code)] for code in band_epochs.events[:, -1]])
    return x, y


def decode_held_out_runs(epochs: mne.Epochs, groups: np.ndarray
                         ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Each test fold is a complete *unseen run* for that subject."""
    x, y = prepare_decoding_data(epochs)
    if len(x) != len(groups):
        raise ValueError("Mismatch between epoch count and run groups")
    if len(np.unique(groups)) < 2:
        raise ValueError("At least two different recording runs are needed")
    cv = LeaveOneGroupOut()
    for train, test in cv.split(x, y, groups=groups):
        if len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2:
            raise ValueError("Every training/test run split must contain both classes")
    predictions = cross_val_predict(make_decoder(), x, y, cv=cv, groups=groups)
    return accuracy_score(y, predictions), confusion_matrix(y, predictions, labels=[0, 1]), y, predictions


def save_tfr_figure(left: mne.time_frequency.AverageTFR,
                    right: mne.time_frequency.AverageTFR, destination: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    # A shared robust scale keeps both conditions comparable without hiding
    # reasonable variation behind a fixed, oversized (or tiny) range.
    values = np.concatenate([np.abs(left.data).ravel(), np.abs(right.data).ravel()])
    scale = max(20.0, float(np.percentile(values, 95)))
    for ax, (tfr, channel, label) in zip(axes, [
        (left, "C4", "Imagine left fist · C4 (contralateral)"),
        (right, "C3", "Imagine right fist · C3 (contralateral)"),
    ]):
        data = tfr.data[tfr.ch_names.index(channel)]
        mesh = ax.pcolormesh(tfr.times, tfr.freqs, data, shading="auto", cmap="RdBu_r",
                             vmin=-scale, vmax=scale)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set(title=label, xlabel="Time relative to cue (s)", ylabel="Frequency (Hz)")
        fig.colorbar(mesh, ax=ax, label="Power change vs baseline (%)")
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def save_accuracy_figure(rows: list[dict], destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ids = [str(row["subject"]) for row in rows]
    values = [row["accuracy"] for row in rows]
    ax.bar(ids, values)
    ax.axhline(0.5, linewidth=1, linestyle="--", color="gray", label="Balanced chance (50%)")
    ax.set(xlabel="Participant ID", ylabel="Held-out-run accuracy", ylim=(0, 1),
           title="Within-subject CSP + LDA decoding")
    ax.legend()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def save_confusion_figure(cm: np.ndarray, destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    image = ax.imshow(cm, cmap="Blues")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(int(value)), ha="center", va="center")
    ax.set(xticks=[0, 1], xticklabels=["Left", "Right"],
           yticks=[0, 1], yticklabels=["Left", "Right"],
           xlabel="Predicted imagery", ylabel="True imagery",
           title="Combined held-out-run predictions")
    fig.colorbar(image, ax=ax, label="Trials")
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def save_erd_figure(rows: list[dict], destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    effects = [r["laterality_erd_pct"] for r in rows]
    ax.scatter(range(1, len(effects) + 1), effects, s=50)
    ax.axhline(0, linestyle="--", color="gray")
    ax.set(xticks=range(1, len(rows) + 1), xticklabels=[str(r["subject"]) for r in rows],
           xlabel="Participant ID", ylabel="Contra − ipsi ERD (% baseline)",
           title="Exploratory hemispheric laterality")
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def run_analysis(subjects: list[int], data_dir: Path, output_dir: Path,
                 use_ica: bool = False, local_only: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    rows = []
    all_confusions = np.zeros((2, 2), dtype=int)
    example_saved = False
    for subject in subjects:
        LOGGER.info("Processing participant %s", subject)
        epochs, groups = load_subject(subject, data_dir, use_ica, local_only=local_only)
        left_tfr = condition_tfr(epochs, "T1")
        right_tfr = condition_tfr(epochs, "T2")
        effect = laterality_effect(left_tfr, right_tfr)
        accuracy, confusion, y, predicted = decode_held_out_runs(epochs, groups)
        all_confusions += confusion
        row = {
            "subject": subject,
            "n_trials": len(y),
            "n_runs": len(np.unique(groups)),
            "left_trials": int(np.count_nonzero(y == 0)),
            "right_trials": int(np.count_nonzero(y == 1)),
            "accuracy": round(float(accuracy), 5),
            "majority_class_baseline": round(float(max(np.mean(y == 0), np.mean(y == 1))), 5),
            "laterality_erd_pct": round(effect, 5),
        }
        rows.append(row)
        LOGGER.info("Subject %s: held-out-run decoding %.1f%% (%d trials)",
                    subject, 100 * accuracy, len(y))
        if not example_saved:
            save_tfr_figure(left_tfr, right_tfr, figures / "example_motor_imagery_tfr.png")
            example_saved = True

    with (output_dir / "subject_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    effects = np.array([r["laterality_erd_pct"] for r in rows], dtype=float)
    pvalue, test_type = sign_flip_pvalue(effects)
    report = {
        "dataset": "PhysioNet EEG Motor Movement/Imagery (EEGBCI)",
        "task": "Left-versus-right fist motor imagery (runs 4, 8, 12)",
        "subjects": subjects,
        "ica_enabled": use_ica,
        "preprocessing": "1–40 Hz FIR bandpass, average reference; optional heuristic ICA",
        "decoding": "8–30 Hz, 0.5–3.5 s; CSP(4) + LDA; leave-one-run-out per subject",
        "mean_subject_accuracy": float(np.mean([r["accuracy"] for r in rows])),
        "mean_laterality_effect_pct": float(np.mean(effects)),
        "laterality_p_one_sided": pvalue,
        "laterality_test": test_type,
        "caveats": [
            "Small exploratory sample; outcomes are not evidence of a clinical BCI.",
            "Held-out runs test within-subject, not new-subject generalization.",
            "The exploratory ERD test is low powered; IC selection uses an imperfect EEG eye proxy.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    save_accuracy_figure(rows, figures / "held_out_run_accuracy.png")
    save_confusion_figure(all_confusions, figures / "held_out_run_confusion.png")
    save_erd_figure(rows, figures / "contralateral_erd.png")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                        help="EEGBCI subject IDs (1–109; default: 1 2 3 4 5)")
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="Dataset cache (downloaded on first execution)")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--local-only", action="store_true",
                        help="Use only existing EDF recordings; never download missing files")
    parser.add_argument("--ica", action="store_true",
                        help="Experimental automatic ocular-ICA rejection using frontal EEG proxies")
    args = parser.parse_args()
    if any(subject < 1 or subject > 109 for subject in args.subjects):
        parser.error("EEGBCI subject IDs must be between 1 and 109")
    if len(set(args.subjects)) != len(args.subjects):
        parser.error("Duplicate subject IDs are not permitted")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    mne.set_log_level("WARNING")
    report = run_analysis(args.subjects, args.data_dir, args.output_dir, args.ica,
                          local_only=args.local_only)
    print(f"\nCompleted {len(args.subjects)} subjects. Mean held-out-run accuracy: "
          f"{100 * report['mean_subject_accuracy']:.1f}%")
    print(f"Results written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
