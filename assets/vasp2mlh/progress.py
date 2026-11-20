"""Render progress on stderr in terminals and redirected scheduler logs."""

import sys
from contextlib import contextmanager
from threading import Event, Thread
from time import monotonic

from tqdm import tqdm


class ProgressBar(tqdm):
    def __init__(self, *args, projected=None, **kwargs):
        self.projected = projected
        super().__init__(*args, **kwargs)

    @property
    def format_dict(self):
        values = super().format_dict
        if self.projected is not None:
            values["projected"] = self.projected.value
        return values


class LogProgress(ProgressBar):
    @staticmethod
    def status_printer(file):
        def write_line(message):
            file.write(message + "\n")
            file.flush()

        return write_line


@contextmanager
def progress_bar(label, total, unit, *, projected=None):
    interactive = sys.stderr.isatty()
    renderer = ProgressBar if interactive else LogProgress
    interval = 2 if interactive else 15
    timing = "{elapsed}<{remaining}" if unit == "job" else "{elapsed}"
    points = ", projected k-points={projected}" if projected is not None else ""
    with renderer(
        total=total,
        projected=projected,
        desc=label,
        unit=unit,
        file=sys.stderr,
        disable=False,
        dynamic_ncols=interactive,
        mininterval=0.3 if interactive else interval,
        miniters=1,
        bar_format=(
            "{desc}: {percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt} ["
            + timing
            + points
            + "{postfix}]"
        ),
    ) as bar:
        # Long native calls and workers waiting on their first result still show elapsed time.
        stop = Event()

        def heartbeat():
            while not stop.wait(interval):
                bar.refresh()

        thread = Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            yield bar
        finally:
            stop.set()
            thread.join()


class ConversionProgress:
    def __init__(self, bar):
        self.bar = bar
        self.last_point = 0.0

    def stage(self, name):
        self.bar.set_postfix_str("", refresh=False)
        self.bar.set_description_str(name, refresh=False)
        if not self.bar.update(1):
            self.bar.refresh()

    def projected(self, done, total):
        self.bar.set_postfix_str(f"k-points {done}/{total}", refresh=False)
        now = monotonic()
        if done in (1, total) or now - self.last_point >= self.bar.mininterval:
            self.bar.refresh()
            self.last_point = now
