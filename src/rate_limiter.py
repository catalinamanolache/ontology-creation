"""
Rate Limiter for Google AI Studio Free Tier (Gemini 2.5 Flash-Lite).

Tracks:
  - RPM (Requests Per Minute) via a 60-second sliding window
  - RPD (Requests Per Day) via a persistent counter

Safety margins:
  - RPM: 25 of 30 max
  - RPD: 1400 of 1500 max
"""

import time
import json
import os
from collections import deque
from datetime import datetime


class RateLimiter:
    def __init__(
        self,
        max_rpm: int = 25,
        max_rpd: int = 1400,
        min_delay: float = 3.0,
        state_file: str = "data/rate_limit_state.json",
    ):
        self.max_rpm = max_rpm
        self.max_rpd = max_rpd
        self.min_delay = min_delay
        self.state_file = state_file

        # Sliding window for RPM tracking (timestamps of recent requests)
        self.request_timestamps: deque = deque()

        # Daily counter
        self.daily_count = 0
        self.daily_date = datetime.now().strftime("%Y-%m-%d")

        # Load persisted state
        self._load_state()

    def _load_state(self):
        """Load daily request count from disk (survives restarts)."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)
                saved_date = state.get("date", "")
                if saved_date == self.daily_date:
                    self.daily_count = state.get("count", 0)
                    print(f"  [RATE LIMITER] Loaded state: {self.daily_count} requests today")
                else:
                    print(f"  [RATE LIMITER] New day detected. Resetting daily counter.")
                    self.daily_count = 0
            except (json.JSONDecodeError, KeyError):
                self.daily_count = 0

    def _save_state(self):
        """Persist daily request count to disk."""
        os.makedirs(os.path.dirname(self.state_file) or ".", exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump({"date": self.daily_date, "count": self.daily_count}, f)

    def _clean_old_timestamps(self):
        """Remove timestamps older than 60 seconds from the sliding window."""
        now = time.time()
        while self.request_timestamps and (now - self.request_timestamps[0]) > 60:
            self.request_timestamps.popleft()

    def can_proceed(self) -> bool:
        """Check if we can make another request without hitting limits."""
        self._clean_old_timestamps()
        rpm_ok = len(self.request_timestamps) < self.max_rpm
        rpd_ok = self.daily_count < self.max_rpd
        return rpm_ok and rpd_ok

    def wait_if_needed(self):
        """
        Block until it's safe to make a request.
        Handles both RPM and RPD limits with appropriate waits.
        """
        # Check daily limit first
        if self.daily_count >= self.max_rpd:
            print(f"\n  [RATE LIMITER] ⛔ DAILY LIMIT REACHED ({self.daily_count}/{self.max_rpd})!")
            print(f"  [RATE LIMITER] Pipeline must stop. Resume tomorrow.")
            raise RuntimeError(
                f"Daily request limit reached ({self.daily_count}/{self.max_rpd}). "
                f"Resume tomorrow or increase the limit (paid tier)."
            )

        # Check RPM
        self._clean_old_timestamps()
        if len(self.request_timestamps) >= self.max_rpm:
            oldest = self.request_timestamps[0]
            wait_time = 60.0 - (time.time() - oldest) + 1.0  # +1s safety
            if wait_time > 0:
                print(f"  [RATE LIMITER] RPM limit reached. Waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
            self._clean_old_timestamps()

        # Enforce minimum delay between requests
        if self.request_timestamps:
            last_request = self.request_timestamps[-1]
            elapsed = time.time() - last_request
            if elapsed < self.min_delay:
                sleep_time = self.min_delay - elapsed
                time.sleep(sleep_time)

    def record_request(self):
        """Record that a request was made."""
        now = time.time()
        self.request_timestamps.append(now)
        self.daily_count += 1
        self._save_state()

    def get_status(self) -> dict:
        """Return current rate limit status."""
        self._clean_old_timestamps()
        return {
            "rpm_used": len(self.request_timestamps),
            "rpm_max": self.max_rpm,
            "rpd_used": self.daily_count,
            "rpd_max": self.max_rpd,
            "rpd_remaining": self.max_rpd - self.daily_count,
        }

    def print_status(self):
        """Print a human-readable status line."""
        s = self.get_status()
        print(
            f"  [RATE] RPM: {s['rpm_used']}/{s['rpm_max']} | "
            f"RPD: {s['rpd_used']}/{s['rpd_max']} | "
            f"Remaining today: {s['rpd_remaining']}"
        )
