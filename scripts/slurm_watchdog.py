# Copyright 2026 UIUC SSAIL
# Part of MegaFold project
#
# Licensed under the MIT License. See LICENSE file in the project root
# for full license text.
#
# If you use this code, please cite:
#
#   La, H., Gupta, A., Morehead, A., Cheng, J., & Zhang, M. (2025).
#   MegaFold: System-Level Optimizations for Accelerating Protein
#   Structure Prediction Models. arXiv:2506.20686
#   https://arxiv.org/abs/2506.20686
#
# BibTeX:
#   @misc{la2025megafoldsystemleveloptimizationsaccelerating,
#       title={MegaFold: System-Level Optimizations for Accelerating
#              Protein Structure Prediction Models},
#       author={Hoa La and Ahan Gupta and Alex Morehead
#               and Jianlin Cheng and Minjia Zhang},
#       year={2025},
#       eprint={2506.20686},
#       archivePrefix={arXiv},
#       primaryClass={q-bio.BM},
#       url={https://arxiv.org/abs/2506.20686},
#   }

import os
import subprocess  # nosec
import time
from pathlib import Path
from typing import Optional

# Parameters
log_dir = Path(
    "/scratch/pawsey1018/amorehead/af3-pytorch-lightning-hydra"
)  # Replace with your job log directory
check_interval = 60  # Check every 60 seconds
timeout_duration = 1080  # 18 minutes in seconds


def get_latest_log_file(directory: str) -> Optional[Path]:
    """Return the latest .out file in the directory based on name order."""
    log_files = sorted(directory.glob("*.out"))
    return log_files[-1] if log_files else None


def get_job_id_from_filename(filename: Optional[Path]) -> Optional[str]:
    """Extract the SLURM job ID from the filename using the pattern 'J-%x.%j.out'."""
    return filename.stem.split(".")[-1] if filename else None


def get_file_modification_time(filepath: Path) -> Optional[float]:
    """Return the last modification time of a file."""
    try:
        return os.path.getmtime(filepath)
    except FileNotFoundError:
        return None


def kill_slurm_job(job_id: str):
    """Kill the SLURM job with the given job_id."""
    subprocess.run(["scancel", job_id], check=True)  # nosec


def main():
    """Monitor the latest job log file and kill the job if no activity is detected."""
    current_log_file = get_latest_log_file(log_dir)
    if not current_log_file:
        print(f"No job log files found in the directory {log_dir}.")
        return

    job_id = get_job_id_from_filename(current_log_file)
    if not job_id:
        print(f"Unable to extract job ID from the filename {current_log_file}.")
        return
    print(f"Monitoring job {job_id} in file {current_log_file}")
    last_mod_time = get_file_modification_time(current_log_file)
    start_time = time.time()

    while True:
        time.sleep(check_interval)

        latest_log_file = get_latest_log_file(log_dir)
        if not latest_log_file:
            print(f"No new job log files found in the directory {log_dir}.")
            continue

        # Switch to a new log file if a newer job has started
        if latest_log_file != current_log_file:
            current_log_file = latest_log_file
            job_id = get_job_id_from_filename(current_log_file)
            last_mod_time = get_file_modification_time(current_log_file)
            start_time = time.time()
            print(f"Switching to monitor new job {job_id} in file {current_log_file}")
            continue

        # Check the current file's modification time
        current_mod_time = get_file_modification_time(current_log_file)

        # If the file has been updated, reset the timeout
        if current_mod_time and current_mod_time > last_mod_time:
            last_mod_time = current_mod_time
            start_time = time.time()

        # If no update within the timeout duration, kill the job and reset
        elif time.time() - start_time > timeout_duration:
            print(
                f"No activity detected for {timeout_duration // 60} minutes. Killing job {job_id}."
            )
            kill_slurm_job(job_id)
            start_time = time.time()  # Reset start time to monitor the next job
            current_log_file = None
            job_id = None  # Reset to pick up the next job log


if __name__ == "__main__":
    main()
