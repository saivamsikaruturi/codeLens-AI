"""GitHub repository loader — clones repos for indexing."""

import shutil
import subprocess
import tempfile
from pathlib import Path


class GitHubLoader:
    """Clone and manage GitHub repositories for indexing."""

    def __init__(self, storage_dir: str = "/tmp/codelens_repos"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def load_repo(self, repo_url: str, branch: str = "main") -> tuple[str, str]:
        """Clone a repo and return (local_path, repo_name)."""
        repo_name = self._extract_repo_name(repo_url)
        local_path = self.storage_dir / repo_name

        if local_path.exists():
            shutil.rmtree(local_path)

        clone_url = self._normalize_url(repo_url)
        result = subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", branch, clone_url, str(local_path)],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            result = subprocess.run(
                ["git", "clone", "--depth", "1", clone_url, str(local_path)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to clone {repo_url}: {result.stderr}")

        return str(local_path), repo_name

    def _extract_repo_name(self, url: str) -> str:
        url = url.rstrip("/")
        if url.endswith(".git"):
            url = url[:-4]
        return url.split("/")[-1]

    def _normalize_url(self, url: str) -> str:
        if url.startswith("http"):
            return url
        if url.startswith("git@"):
            return url
        return f"https://github.com/{url}.git"

    def cleanup(self, repo_name: str):
        path = self.storage_dir / repo_name
        if path.exists():
            shutil.rmtree(path)
