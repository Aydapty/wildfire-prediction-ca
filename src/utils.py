from datetime import datetime

def log(*args, **kwargs):
    """Helper function to print messages with a precise timestamp."""
    timestamp = datetime.now().strftime("[%H:%M:%S]")
    print(timestamp, *args, **kwargs)