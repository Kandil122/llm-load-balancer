# Dynamic Worker Recovery & Scheduling Updates

This document summarizes the recent architectural improvements made to the load balancing and fault tolerance logic.

### 1. Parallel Heartbeats (Fast Detection)
**Change:** The `heartbeat_loop` in `master/http_scheduler.py` was refactored from a sequential `for` loop to a parallel `asyncio.gather` execution.
*   **How it works:** Instead of checking each worker one by one (which can be slow if a worker times out), the Master now pings all configured workers simultaneously.
*   **Impact:** Detection of a recovered worker is now up to **4x faster**, occurring almost immediately after the container starts.

### 2. Concurrency Throttling (Semaphore)
**Change:** A `Semaphore` was added to the load generation logic in `main.py` for both Distributed and Hybrid modes.
*   **How it works:** It limits the number of active requests to `(num_workers * 2)`. Instead of sending all 100 requests in the first millisecond, the system now processes them in "batches."
*   **Why?** In a real-world system, requests arrive over time. By spreading out the requests, we allow the Scheduler to "see" workers that join mid-run.
*   **Result:** If a worker is started 30 seconds into a test, the pending requests in the queue will automatically be assigned to the new worker, demonstrating **True Dynamic Load Balancing**.

### 3. Recovery Logging
**Change:** Added a visual notification in the console when a worker returns to an `alive` state.
*   **Visual:** `✨ [Scheduler] http://localhost:800x has RECOVERED and is now online.`
*   **Purpose:** Provides immediate feedback during a demonstration that the self-healing logic has successfully reintegrated a failed node.

---

### How to Revert These Changes
If you prefer the original "instant-burst" behavior without mid-run recovery:
1.  In `main.py`, remove the `async with sem:` lines and the `sem` definition.
2.  In `master/http_scheduler.py`, revert the `heartbeat_loop` to a standard `for` loop.
