"""
base_workflow.py

Parent class that all Week 3 workflows inherit from.
Provides two shared features:
  1. Idempotency  — _output_exists() checks if we already ran this workflow today.
  2. Output saving — _save_output() writes the result to data/reports/.
"""

from abc import ABC, abstractmethod
from pathlib import Path


class BaseWorkflow(ABC):
    """
    All workflows extend this class.

    ABC (Abstract Base Class) means Python forces every subclass
    to implement the run() method — you cannot forget it.
    """

    output_dir = Path("data/reports")

    def __init__(self, groq_api_key: str) -> None:
        self.groq_api_key = groq_api_key
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def run(self) -> dict:
        """Each workflow must implement this. Returns a result dict."""

    def _output_exists(self, filename: str) -> bool:
        """
        Idempotency check.
        If the output file already exists, the workflow was already run.
        We skip re-running to avoid duplicate side effects.
        """
        return (self.output_dir / filename).exists()

    def _save_output(self, filename: str, content: str) -> None:
        """Write workflow output to data/reports/<filename>."""
        (self.output_dir / filename).write_text(content, encoding="utf-8")
