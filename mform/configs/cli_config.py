"""Memory Transformer CLI Configuration."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

from mform.workflow.workflow import MemoryTransformerWorkflow


class MformCLIConfig(BaseSettings):
    """Memory Transformer CLI Configuration."""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    def cli_cmd(self) -> None:
        """Run Components of Memory Transformer."""
        workflow = MemoryTransformerWorkflow()
        workflow.run()
