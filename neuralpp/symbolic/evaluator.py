from contextlib import contextmanager
from time import monotonic
from typing import Dict


class EvaluationLog:
    def __init__(self):
        self.counter = 0
        self.time = 0

    def increase_counter(self):
        self.counter = self.counter + 1

    def add_time_delta(self, delta):
        self.time = self.time + delta


class Evaluator:
    def __init__(self):
        self._logs: Dict[str, EvaluationLog] = {}
        self._current_section = None
        self._current_start_time = None

    @contextmanager
    def log_section(self, section_name):
        old_section, old_time = self._current_section, self._current_start_time
        start = monotonic()
        self._current_section, self._current_start_time = section_name, start

        if section_name not in self._logs:
            self._logs[section_name] = EvaluationLog()
        self._logs[section_name].increase_counter()
        try:
            if old_section is not None:
                self._logs[old_section].add_time_delta(start - old_time)
            yield
        finally:
            end = monotonic()
            self._logs[section_name].add_time_delta(end - self._current_start_time)
            self._current_section, self._current_start_time = old_section, end

    def print_result(self, prefix):
        for section_name, log in self._logs.items():
            print(f"{prefix}{section_name}: {log.counter} times in {log.time:.3f} seconds")
