# LLM Response Export Updates

This document tracks the changes made to implement the optional LLM response saving feature.

### 1. New Command Line Flag
**File:** `main.py` (Line 36-37)
*   **Change:** Added `--save-output` to the `argparse` configuration.
*   **Purpose:** Allows the user to choose whether they want to export the actual text responses or just the performance metrics.

### 2. Capture Logic in Distributed & Hybrid Mode
**File:** `main.py` (Lines 68, 102, 105, 142, 146, 151)
*   **Change:** Initialized `responses_data = []` and added code inside `wrap_task` and `wrap_request` to append the LLM's `result` text and the `worker_id` to this list.
*   **Purpose:** Since these tasks run in parallel, we need a thread-safe way to collect all individual worker responses as they come back.

### 3. Capture Logic in Simulation Mode
**File:** `main.py` (Lines 185-188)
*   **Change:** Modified the simulation loop to append `resp.result` to the `final_outputs` list.
*   **Purpose:** Ensures the feature works perfectly even when running locally without Docker.

### 4. File Writing Routine
**File:** `main.py` (Lines 210-218)
*   **Change:** Added a block that checks `if args.save_output`. It sorts the collected results by `request_id` and writes them into a new CSV file in the `results/` folder.
*   **Output Format:** `results/[mode]_[strategy]_[users]users_outputs.csv`
*   **Purpose:** Creates a permanent record of what the LLM actually said for every question in the test.

---

### How to use:
To run a test and save the actual answers, add the flag:
```bash
python main.py --distributed --users 10 --save-output
```
Then check the `results/` folder for the new `_outputs.csv` file.
