"""
CDW Agentic Pipeline — Case Tracker
====================================
JSON-backed persistent state tracker for batch pipeline runs.

Writes state to disk after every case update, so the pipeline can resume
after crashes (OOM, NFS timeout, GPU error) without re-processing completed cases.

State file format (JSON):
{
    "run_id": "20260320_143012",
    "started_at": "2026-03-20T14:30:12",
    "cases": {
        "PT123/CT_ABD": {"status": "completed", "updated_at": "...", "tools_run": [...], "error": ""},
        "PT456/MRI_ABD": {"status": "failed", "updated_at": "...", "tools_run": [], "error": "OOM"},
        "PT789/CT_CHEST": {"status": "pending", "updated_at": "...", "tools_run": [], "error": ""},
    }
}
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Valid state transitions
_VALID_STATES = {"pending", "running", "completed", "failed", "skipped"}


class CaseTracker:
    """
    Persistent case state tracker backed by a JSON file on disk.

    Thread-safe: all state mutations are protected by a lock.
    Writes to disk after every status change for crash resilience.

    Args:
        state_file: Path to the JSON state file. Created if it doesn't exist.
        run_id: Identifier for this pipeline run (default: timestamp).
    """

    def __init__(self, state_file: str, run_id: str = ""):
        self.state_file = state_file
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._lock = threading.Lock()
        self._state: dict = {}
        self._load_or_init()

    # ── Initialization ─────────────────────────────────────────────────────

    def _load_or_init(self) -> None:
        """Load existing state file or create a new one."""
        if os.path.isfile(self.state_file):
            with open(self.state_file) as f:
                self._state = json.load(f)
            logger.info(
                "Loaded existing state: %d cases (%d completed, %d failed, %d pending)",
                len(self._state.get("cases", {})),
                self.count("completed"),
                self.count("failed"),
                self.count("pending"),
            )
        else:
            self._state = {
                "run_id": self.run_id,
                "started_at": datetime.now().isoformat(),
                "cases": {},
            }
            os.makedirs(os.path.dirname(self.state_file) or ".", exist_ok=True)
            self._flush()
            logger.info("Created new state file: %s", self.state_file)

    def register_cases(self, case_ids: List[str]) -> int:
        """
        Register a list of case IDs. Cases already in state are not overwritten.

        Returns:
            Number of new cases added (excludes already-registered ones).
        """
        added = 0
        with self._lock:
            for cid in case_ids:
                if cid not in self._state["cases"]:
                    self._state["cases"][cid] = {
                        "status": "pending",
                        "updated_at": datetime.now().isoformat(),
                        "tools_run": [],
                        "error": "",
                    }
                    added += 1
            self._flush()
        logger.info("Registered %d new cases (%d already tracked)", added, len(case_ids) - added)
        return added

    # ── Status updates ─────────────────────────────────────────────────────

    def set_status(
        self,
        case_id: str,
        status: str,
        error: str = "",
        tools_run: Optional[List[str]] = None,
    ) -> None:
        """Update status for a single case. Writes to disk immediately."""
        if status not in _VALID_STATES:
            raise ValueError(f"Invalid status: {status!r}. Must be one of {_VALID_STATES}")
        with self._lock:
            if case_id not in self._state["cases"]:
                self._state["cases"][case_id] = {
                    "status": status,
                    "updated_at": datetime.now().isoformat(),
                    "tools_run": tools_run or [],
                    "error": error,
                }
            else:
                entry = self._state["cases"][case_id]
                entry["status"] = status
                entry["updated_at"] = datetime.now().isoformat()
                if error:
                    entry["error"] = error
                if tools_run is not None:
                    entry["tools_run"] = tools_run
            self._flush()

    def add_tool_run(self, case_id: str, tool_name: str) -> None:
        """Append a tool name to a case's tools_run list."""
        with self._lock:
            if case_id in self._state["cases"]:
                tools = self._state["cases"][case_id].get("tools_run", [])
                if tool_name not in tools:
                    tools.append(tool_name)
                    self._state["cases"][case_id]["tools_run"] = tools
                    self._flush()

    # ── Queries ────────────────────────────────────────────────────────────

    def get_status(self, case_id: str) -> str:
        """Get the current status of a case. Returns 'unknown' if not tracked."""
        return self._state.get("cases", {}).get(case_id, {}).get("status", "unknown")

    def get_pending(self) -> List[str]:
        """Return list of case IDs with status 'pending'."""
        return [cid for cid, v in self._state.get("cases", {}).items() if v["status"] == "pending"]

    def get_failed(self) -> List[str]:
        """Return list of case IDs with status 'failed'."""
        return [cid for cid, v in self._state.get("cases", {}).items() if v["status"] == "failed"]

    def get_completed(self) -> List[str]:
        """Return list of case IDs with status 'completed'."""
        return [cid for cid, v in self._state.get("cases", {}).items() if v["status"] == "completed"]

    def count(self, status: str) -> int:
        """Count cases with a given status."""
        return sum(1 for v in self._state.get("cases", {}).values() if v["status"] == status)

    @property
    def total(self) -> int:
        return len(self._state.get("cases", {}))

    def summary(self) -> Dict[str, int]:
        """Return a dict of status → count."""
        counts: Dict[str, int] = {}
        for v in self._state.get("cases", {}).values():
            s = v["status"]
            counts[s] = counts.get(s, 0) + 1
        return counts

    # ── Persistence ────────────────────────────────────────────────────────

    def _flush(self) -> None:
        """Write current state to disk. Must be called with lock held."""
        tmp = self.state_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._state, f, indent=2)
        os.replace(tmp, self.state_file)  # atomic on POSIX
