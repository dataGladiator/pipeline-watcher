"""
Clock and timezone utilities for pipeline-watcher.

This module centralizes all time-related helpers used by the watcher,
ensuring consistent and timezone-aware timestamps across reports,
artifacts, and UI banners.  All functions return timezone-aware
:class:`datetime.datetime` objects (never naive datetimes).

Functions
---------
now_utc()
    Return the current UTC time with timezone info attached.
now_eastern()
    Return the current time in U.S. Eastern timezone.
now_central()
    Return the current time in U.S. Central timezone.
now_mountain()
    Return the current time in U.S. Mountain timezone.
now_pacific()
    Return the current time in U.S. Pacific timezone.
"""

from __future__ import annotations
from datetime import datetime, timezone as tz
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Core UTC clock
# ---------------------------------------------------------------------------

def now_utc() -> datetime:
    """
    Return the current UTC time.

    Returns
    -------
    datetime.datetime
        Timezone-aware datetime in UTC.

    Examples
    --------
    >>> from pipeline_watcher.clocks import now_utc
    >>> t = now_utc()
    >>> t.tzinfo
    datetime.timezone.utc
    """
    return datetime.now(tz.utc)


# ---------------------------------------------------------------------------
# U.S. timezone helpers
# ---------------------------------------------------------------------------

def now_eastern() -> datetime:
    """
    Return the current time in U.S. Eastern timezone.

    Returns
    -------
    datetime.datetime
        Timezone-aware datetime for America/New_York.
    """
    return datetime.now(ZoneInfo("America/New_York"))


def now_central() -> datetime:
    """
    Return the current time in U.S. Central timezone.

    Returns
    -------
    datetime.datetime
        Timezone-aware datetime for America/Chicago.
    """
    return datetime.now(ZoneInfo("America/Chicago"))


def now_mountain() -> datetime:
    """
    Return the current time in U.S. Mountain timezone.

    Returns
    -------
    datetime.datetime
        Timezone-aware datetime for America/Denver.
    """
    return datetime.now(ZoneInfo("America/Denver"))


def now_pacific() -> datetime:
    """
    Return the current time in U.S. Pacific timezone.

    Returns
    -------
    datetime.datetime
        Timezone-aware datetime for America/Los_Angeles.
    """
    return datetime.now(ZoneInfo("America/Los_Angeles"))
