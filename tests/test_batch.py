"""
Test script for batch_runner module.
Verifies directory discovery, validation, config parsing, and master output generation.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from runners.batch_runner import (
    discover_participants,
    validate_participant_folder,
    parse_master_csv,
    build_participant_config,
    generate_master_output,
)

TEST_DIR = Path(__file__).parent.parent / "data" / "test_batch"
OUT_DIR = Path(__file__).parent.parent / "data" / "test_batch_output"

def test_discover():
    folders = discover_participants(TEST_DIR)
    assert len(folders) == 2, f"Expected 2 folders, got {len(folders)}"
    assert folders[0].name == "01"
    assert folders[1].name == "02"
    print("PASS: discover_participants")

def test_validate():
    missing = validate_participant_folder(TEST_DIR / "01")
    assert missing == [], f"Expected no missing files, got {missing}"
    print("PASS: validate_participant_folder (all present)")

    # Test with a folder missing files
    import tempfile, os
    tmp = Path(tempfile.mkdtemp())
    missing = validate_participant_folder(tmp)
    assert len(missing) == 5, f"Expected 5 missing, got {len(missing)}"
    os.rmdir(tmp)
    print("PASS: validate_participant_folder (missing files)")

def test_parse_master():
    meta = parse_master_csv(TEST_DIR / "01" / "master.csv")
    assert meta["height"] == 1.72, f"Expected 1.72, got {meta['height']}"
    assert meta["fps"] == 30.0, f"Expected 30.0, got {meta['fps']}"
    assert meta["speed_factor"] == 8.0, f"Expected 8.0, got {meta['speed_factor']}"
    print("PASS: parse_master_csv")

def test_build_config():
    cfg = build_participant_config(TEST_DIR / "01")
    assert cfg.participant_id == "sub_01"
    assert cfg.height_m == 1.72
    assert cfg.fps == 30.0
    assert cfg.speed_factor == 8.0
    assert cfg.st_video == TEST_DIR / "01" / "single.mp4"
    assert cfg.dt_video == TEST_DIR / "01" / "dual.mp4"
    assert cfg.st_boundaries_csv == TEST_DIR / "01" / "single.csv"
    assert cfg.dt_boundaries_csv == TEST_DIR / "01" / "dual.csv"
    print("PASS: build_participant_config")

def test_master_output():
    """Test master CSV generation with mock aggregated results."""
    from runners.batch_runner import ParticipantConfig

    # Create mock aggregated data for 2 participants
    configs = [
        ParticipantConfig("sub_01", TEST_DIR/"01", TEST_DIR/"01"/"single.mp4",
                          TEST_DIR/"01"/"dual.mp4", TEST_DIR/"01"/"single.csv",
                          TEST_DIR/"01"/"dual.csv", 1.72, 30, 8.0),
        ParticipantConfig("sub_02", TEST_DIR/"02", TEST_DIR/"02"/"single.mp4",
                          TEST_DIR/"02"/"dual.mp4", TEST_DIR/"02"/"single.csv",
                          TEST_DIR/"02"/"dual.csv", 1.68, 30, 8.0),
    ]

    mock_results = []
    for i, cfg in enumerate(configs):
        st_agg = pd.DataFrame([{
            "stride_lengths_avg": 1.2 + i*0.1, "stride_times_avg": 1.0 + i*0.05,
            "cadence_avg": 120.0 - i*5, "speed_avg": 1.2 + i*0.1,
            "stride_lengths_CV": 0.05, "stride_times_CV": 0.04,
            "stride_lengths_SI": 0.02, "stride_times_SI": 0.01,
            "stride_lengths_avg_left": 1.15, "stride_lengths_avg_right": 1.25,
            "sub": cfg.participant_id, "condition": "st",
        }])
        dt_agg = pd.DataFrame([{
            "stride_lengths_avg": 1.1 + i*0.1, "stride_times_avg": 1.1 + i*0.05,
            "cadence_avg": 110.0 - i*5, "speed_avg": 1.0 + i*0.1,
            "stride_lengths_CV": 0.07, "stride_times_CV": 0.06,
            "stride_lengths_SI": 0.03, "stride_times_SI": 0.02,
            "stride_lengths_avg_left": 1.05, "stride_lengths_avg_right": 1.15,
            "sub": cfg.participant_id, "condition": "dt",
        }])
        dtc_df = pd.DataFrame([{
            "sub": cfg.participant_id, "condition": "st",
            "stride_lengths_avg_DTC": (1.2+i*0.1 - (1.1+i*0.1)) / (1.2+i*0.1) * 100,
            "stride_times_avg_DTC": ((1.0+i*0.05) - (1.1+i*0.05)) / (1.0+i*0.05) * 100,
            "cadence_avg_DTC": ((120-i*5) - (110-i*5)) / (120-i*5) * 100,
            "speed_avg_DTC": ((1.2+i*0.1) - (1.0+i*0.1)) / (1.2+i*0.1) * 100,
        }])
        mock_results.append({
            "aggregate_st": st_agg,
            "aggregate_dt": dt_agg,
            "dtc": {"dtc": dtc_df, "dtc_summary": pd.DataFrame()},
        })

    master_dir = OUT_DIR / "master"
    master_df = generate_master_output(mock_results, configs, master_dir)

    # Verify structure
    assert len(master_df) == 3, f"Expected 3 rows (2 participants + average), got {len(master_df)}"
    assert master_df.iloc[-1]["sub"] == "AVERAGE"

    # Verify average DTC is recomputed, not averaged
    avg_row = master_df[master_df["sub"] == "AVERAGE"].iloc[0]
    avg_st_stride = master_df[master_df["sub"] != "AVERAGE"]["stride_lengths_avg_st"].mean()
    avg_dt_stride = master_df[master_df["sub"] != "AVERAGE"]["stride_lengths_avg_dt"].mean()
    expected_dtc = (avg_st_stride - avg_dt_stride) / avg_st_stride * 100
    actual_dtc = avg_row["stride_lengths_avg_DTC"]
    assert abs(actual_dtc - expected_dtc) < 0.001, \
        f"DTC mismatch: expected {expected_dtc:.3f}, got {actual_dtc:.3f}"

    # Verify CSV was saved
    assert (master_dir / "master.csv").exists(), "master.csv not created"
    print(f"PASS: generate_master_output")
    print(f"  Master CSV: {master_dir / 'master.csv'}")
    print(f"  Graphs: {list(master_dir.glob('*.png'))}")

    # Print the master CSV for inspection
    print(f"\nMaster CSV contents:")
    print(master_df.to_string(index=False))

if __name__ == "__main__":
    print("=" * 60)
    print("  Batch Runner Module Tests")
    print("=" * 60)
    test_discover()
    test_validate()
    test_parse_master()
    test_build_config()
    test_master_output()
    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)
