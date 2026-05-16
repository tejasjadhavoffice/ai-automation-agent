from abc import ABC, abstractmethod
from pathlib import Path


class BaseWorkflow(ABC):

    output_dir = Path("data/reports")

    def __init__(self, groq_api_key: str) -> None:
        self.groq_api_key = groq_api_key
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def run(self) -> dict:
        """Each workflow must implement this. Returns a result dict."""

    def _is_already_processed(self, filename: str) -> bool:

        return (self.output_dir / filename).exists()

    def _save_result_to_file(self, filename: str, content: str) -> None:
        
        (self.output_dir / filename).write_text(content, encoding="utf-8")
