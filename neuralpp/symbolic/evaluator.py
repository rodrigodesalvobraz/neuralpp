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

    @contextmanager
    def log_section(self, section_name):
        if section_name not in self._logs:
            self._logs[section_name] = EvaluationLog()

        self._logs[section_name].increase_counter()
        start = monotonic()
        try:
            yield
        finally:
            self._logs[section_name].add_time_delta(monotonic() - start)

    def print_result(self, prefix):
        for section_name, log in self._logs.items():
            print(f"{prefix}{section_name}: {log.counter} times in {log.time} seconds")
