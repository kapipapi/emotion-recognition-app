import threading


class SensorSource:
    """Abstract object for a sensory modality."""

    def __init__(self):
        """Initialise object."""
        self.started = False
        self.thread = None

    def start(self):
        """Start capture source."""
        if self.started:
            print('[!] Asynchronous capturing has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(
            target=self.update,
            args=()
        )
        self.thread.start()
        return self

    def update(self):
        """Update data."""
        pass

    def read(self):
        """Read data."""
        pass

    def stop(self):
        """Stop daemon."""
        self.started = False
        self.thread.join()
