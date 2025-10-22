from datetime import datetime, timezone as tz

def now_utc() -> datetime:
    return datetime.now(tz.utc)

