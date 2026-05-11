# -*- coding: utf-8 -*-
"""
run_all.py  —  Single entry point for the full gait reliability pipeline.

Usage
-----
  python run_all.py                                    # opens folder picker, runs both methods
  python run_all.py path/to/output_folder              # CLI mode, runs both methods
  python run_all.py path/to/output_folder --method ap  # CLI mode, AP only

All results are saved to:  reliability/out/<input_folder_name>/
"""
import sys, os, webbrowser, time, argparse

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

# Add the scripts dir so imports work from anywhere
sys.path.insert(0, SCRIPTS_DIR)

from gait_reliability import run as run_reliability, _method_suffix
from generate_dashboard import generate as generate_dashboard


def _pick_folder_gui() -> str | None:
    """
    Open a native folder-picker dialog using tkinter.
    Returns the selected folder path, or None if cancelled.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()       # hide the root window
        root.attributes("-topmost", True)
        folder = filedialog.askdirectory(
            title="Select the pipeline output folder (contains sub_01/, sub_03/, …)",
        )
        root.destroy()
        return folder if folder else None
    except Exception as e:
        print(f"[ERROR] Could not open folder picker: {e}")
        return None


def _build_output_dir(input_dir: str) -> str:
    """
    Build the output path:  reliability/out/<input_folder_name>/

    The 'out' folder lives alongside the reliability scripts.
    """
    input_name = os.path.basename(os.path.normpath(input_dir))
    out_dir = os.path.join(SCRIPTS_DIR, "out", input_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def main():
    parser = argparse.ArgumentParser(
        description="Gait Reliability Pipeline — run analysis + dashboard.\n"
                    "When run without arguments, opens a folder picker.",
    )
    parser.add_argument(
        "data_dir", nargs="?", default=None,
        help="Path to the output folder containing volunteer subdirectories "
             "(e.g. sub_01/, sub_03/).  If omitted, a folder picker opens."
    )
    parser.add_argument(
        "--method", choices=["vertical", "ap", "both"], default="both",
        help="Event detection method to analyse. Default: 'both'."
    )
    args = parser.parse_args()

    # ── Resolve input directory ───────────────────────────────────────────
    if args.data_dir:
        data_dir = os.path.abspath(args.data_dir)
    else:
        data_dir_pick = _pick_folder_gui()
        if not data_dir_pick:
            print("No folder selected — exiting.")
            sys.exit(0)
        data_dir = os.path.abspath(data_dir_pick)

    if not os.path.isdir(data_dir):
        print(f"[ERROR] Folder not found: {data_dir}")
        sys.exit(1)

    # ── Build output directory ────────────────────────────────────────────
    output_dir = _build_output_dir(data_dir)
    input_label = os.path.basename(os.path.normpath(data_dir))

    methods = ["vertical", "ap"] if args.method == "both" else [args.method]

    print("\n" + "#"*55)
    print("  GAIT RELIABILITY PIPELINE")
    print(f"  Input:  {data_dir}")
    print(f"  Output: {output_dir}")
    print("#"*55)

    any_success = False
    for method in methods:
        rel_df, volunteers = run_reliability(
            data_dir=data_dir, output_dir=output_dir, method=method
        )
        if rel_df is None:
            print(f"[SKIP] No data for method '{method}'.")
            continue
        any_success = True

    # Generate a single combined dashboard after all methods finish
    if any_success:
        html_path = generate_dashboard(output_dir, data_dir_label=input_label)
        if html_path and os.path.exists(html_path):
            url = "file:///" + html_path.replace("\\", "/")
            print(f"\n  Opening: {url}")
            time.sleep(0.5)
            webbrowser.open(url)

    print("\n  Pipeline complete!")
    print(f"  All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
