"""Synthetic-data smoke tests; no internet/dataset download required."""

import numpy as np
import mne

from eeg_pipeline import (
    decode_held_out_runs,
    mean_erd,
    preprocess,
    sign_flip_pvalue,
    condition_tfr,
)


def make_fake_raw(seed=0):
    rng = np.random.default_rng(seed)
    sfreq = 160.0
    ch_names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4"]
    n_times = int(39 * sfreq)
    times = np.arange(n_times) / sfreq
    data = rng.normal(scale=1.0e-6, size=(len(ch_names), n_times))
    events = []
    for i in range(6):
        onset = 2.0 + 6 * i
        left = i % 2 == 0
        events.append((onset, "T1" if left else "T2"))
        active = (times >= onset + 0.5) & (times < onset + 3.5)
        data[4 if left else 5, active] += 2e-6 * np.sin(2 * np.pi * 10 * times[active])
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.set_montage("standard_1005", verbose="ERROR")
    raw.set_annotations(mne.Annotations(
        onset=[e[0] for e in events],
        duration=[0] * len(events),
        description=[e[1] for e in events],
    ))
    return raw


def test_preprocess_and_tfr():
    epochs = preprocess(make_fake_raw(), use_ica=False)
    assert len(epochs) == 6
    tfr = condition_tfr(epochs, "T1")
    assert np.isfinite(mean_erd(tfr, "C3"))


def test_cross_validation_holds_out_runs():
    epo = [preprocess(make_fake_raw(i), use_ica=False) for i in (0, 1, 2)]
    groups = np.repeat([4, 8, 12], [len(e) for e in epo])
    epochs = mne.concatenate_epochs(epo, verbose="ERROR")
    accuracy, confusion, y, pred = decode_held_out_runs(epochs, groups)
    assert len(pred) == len(y) == len(epochs)
    assert confusion.sum() == len(epochs)
    assert 0 <= accuracy <= 1


def test_exact_permutation_test():
    pvalue, method = sign_flip_pvalue(np.array([-1.0, -2.0]))
    assert pvalue == 0.25
    assert method == "exact sign-flip"


def test_full_output_generation_with_synthetic_data(tmp_path, monkeypatch):
    import eeg_pipeline as pipeline

    def fake_loader(subject, data_dir, use_ica, local_only=False):
        runs = [preprocess(make_fake_raw(subject * 10 + i)) for i in range(3)]
        return (
            mne.concatenate_epochs(runs, verbose="ERROR"),
            np.repeat([4, 8, 12], [len(epochs) for epochs in runs]),
        )

    monkeypatch.setattr(pipeline, "load_subject", fake_loader)
    output_dir = tmp_path / "results"
    report = pipeline.run_analysis([1, 2], tmp_path / "data", output_dir)
    assert report["task"].startswith("Left-versus-right fist motor imagery")
    assert (output_dir / "subject_results.csv").is_file()
    assert (output_dir / "summary.json").is_file()
    for name in (
        "example_motor_imagery_tfr.png",
        "held_out_run_accuracy.png",
        "held_out_run_confusion.png",
        "contralateral_erd.png",
    ):
        assert (output_dir / "figures" / name).stat().st_size > 10_000


def test_local_edf_discovery_without_network(tmp_path, monkeypatch):
    import eeg_pipeline as pipeline

    # Support both nested and flat files (without reading dummy EDF content).
    nested = tmp_path / "S004"
    nested.mkdir()
    (nested / "S004R04.edf").write_bytes(b"fake")
    (tmp_path / "S004R08.edf").write_bytes(b"fake")
    deep = tmp_path / "more" / "folders"
    deep.mkdir(parents=True)
    (deep / "S004R12.edf").write_bytes(b"fake")

    def no_network(*args, **kwargs):
        raise AssertionError("Existing local EDF recordings should not be downloaded")

    monkeypatch.setattr(pipeline.eegbci, "load_data", no_network)
    files = pipeline.fetch_recordings(4, tmp_path, local_only=True)
    assert [run for run, _ in files] == [4, 8, 12]
    assert [path.name for _, path in files] == ["S004R04.edf", "S004R08.edf", "S004R12.edf"]


def test_local_only_explains_missing_files(tmp_path):
    import pytest
    import eeg_pipeline as pipeline

    (tmp_path / "S004R04.edf").write_bytes(b"fake")
    with pytest.raises(FileNotFoundError, match="S004R08.edf, S004R12.edf"):
        pipeline.fetch_recordings(4, tmp_path, local_only=True)


def test_tfr_uses_real_percentage_points():
    """Guard against MNE's fraction-named-'percent' convention."""
    epo = preprocess(make_fake_raw(seed=88))
    expected_fraction = epo['T1'].compute_tfr(
        method='morlet', freqs=np.arange(8.0, 31.0, 2.0),
        n_cycles=np.arange(8.0, 31.0, 2.0) / 2,
        picks=['C3', 'C4'], average=True, return_itc=False, verbose='ERROR',
    )
    expected_fraction.apply_baseline(baseline=(-0.8, -0.2), mode='percent', verbose='ERROR')
    actual_percent = condition_tfr(epo, 'T1')
    np.testing.assert_allclose(actual_percent.data, expected_fraction.data * 100, rtol=1e-7, atol=1e-7)
