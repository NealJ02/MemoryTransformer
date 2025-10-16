"""Memory Transformer CLI."""
from __future__ import annotations

from pydantic_settings import BaseSettings, CliApp, CliSubCommand, SettingsConfigDict

from mform.configs.cli_config import MformCLIConfig  # noqa: TC001


class NqaCLI(BaseSettings):
    """NQA (Notes Question and Answer) CLI.

    Commands:
        notes-qna             Run Notes Q&A
    """
    mem_transform: CliSubCommand[MformCLIConfig]
    model_config = SettingsConfigDict(
        populate_by_name=True,
        validate_default=True,
        extra="ignore",
        case_sensitive=False,
        # environment file support
        env_file=".env",
        env_file_encoding="utf-8",
        # CLI Configuration
        cli_parse_args=True,
        cli_implicit_flags=True,
        cli_kebab_case=True,
    )
    def cli_cmd(self) -> None:
        """Run Notes Q&A CLI."""
        CliApp.run_subcommand(self)

def main() -> None:
    """Run Notes Q&A CLI."""
    CliApp.run(NqaCLI)


if __name__ == "__main__":
    main()
