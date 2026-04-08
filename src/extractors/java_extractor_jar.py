"""Java metadata extractor that delegates to the Java-based cheap-rag JAR.

The JAR (cheap/cheap-rag/build/libs/cheap-rag-0.1.jar) uses JavaParser for
full-fidelity AST analysis: proper generics, Javadoc, method signatures, enums,
and public-only filtering.  The Python extractor (java_extractor.py) is retired.

The JAR outputs JSON matching the Python MetadataArtifact schema.  Field mapping:
  documentation  → description  (Javadoc / comment text)
  embedding_text → metadata["embedding_text"]  (used by to_embedding_text())
  signature      → metadata["signature"]
  qualified_name → metadata["qualified_name"]
  parent_id      → metadata["parent_id"]
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from src.extractors.base import MetadataArtifact

logger = logging.getLogger(__name__)

# Default JAR location relative to cheap-rag project root
_DEFAULT_JAR = (
    Path(__file__).parent.parent.parent
    / ".."
    / "cheap"
    / "cheap-rag"
    / "build"
    / "libs"
    / "cheap-rag-0.1.jar"
)


class JavaExtractorJar:
    """Extract Java metadata by shelling out to the Java-based extractor JAR.

    The JAR uses JavaParser for production-quality extraction and outputs a JSON
    array of MetadataArtifact objects whose field names are snake_case and match
    the Python schema.

    Args:
        jar_path: Path to cheap-rag-*.jar. Defaults to the build output location
            at ``../cheap/cheap-rag/build/libs/cheap-rag-0.1.jar`` relative to
            the cheap-rag project root.
        public_only: When True, pass ``--public-only`` to the JAR so only public
            API members are extracted.  Defaults to False.
        java_executable: Path to the ``java`` binary.  Defaults to ``"java"``.
    """

    def __init__(
        self,
        jar_path: Path | str | None = None,
        public_only: bool = False,
        java_executable: str = "java",
    ) -> None:
        self.jar_path = Path(jar_path) if jar_path else _DEFAULT_JAR.resolve()
        self.public_only = public_only
        self.java_executable = java_executable

        if not self.jar_path.exists():
            raise FileNotFoundError(
                f"Java extractor JAR not found: {self.jar_path}\n"
                "Build it with: cd ../cheap && ./gradlew :cheap-rag:jar"
            )

        logger.info(f"JavaExtractorJar initialised with JAR: {self.jar_path}")

    # ------------------------------------------------------------------
    # MetadataExtractor protocol
    # ------------------------------------------------------------------

    def extract_metadata(self, source_path: Path) -> list[MetadataArtifact]:
        """Extract metadata from a Java source file or directory.

        Args:
            source_path: Path to a ``.java`` file or a directory tree.

        Returns:
            List of MetadataArtifact objects.
        """
        if not source_path.exists():
            logger.warning(f"Source path does not exist: {source_path}")
            return []

        # Write JSON to a temp file via -o so JAR logging noise on stdout
        # doesn't contaminate the artifact data.
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        cmd = [
            self.java_executable,
            "-jar",
            str(self.jar_path),
            str(source_path.resolve()),
            "-o",
            str(tmp_path),
        ]
        if self.public_only:
            cmd.append("--public-only")

        logger.info(f"Running Java extractor on: {source_path}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            logger.error(f"Java extractor failed (exit {exc.returncode}): {exc.stderr}")
            tmp_path.unlink(missing_ok=True)
            raise

        if result.stderr:
            for line in result.stderr.splitlines():
                if line.strip():
                    logger.warning(f"Java extractor: {line}")

        try:
            raw: list[dict[str, Any]] = json.loads(tmp_path.read_text(encoding="utf-8"))
        finally:
            tmp_path.unlink(missing_ok=True)

        artifacts = [self._convert(item) for item in raw]
        logger.info(f"Java extractor produced {len(artifacts)} artifacts from {source_path}")
        return artifacts

    def language(self) -> str:
        return "java"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _convert(self, raw: dict[str, Any]) -> MetadataArtifact:
        """Convert a raw JSON dict from the JAR to a Python MetadataArtifact."""
        # Core fields
        artifact_id: str = raw["id"]
        name: str = raw["name"]
        artifact_type: str = raw["type"]  # "class", "interface", "enum", ...
        language: str = raw.get("language", "java")
        source_type: str = raw.get("source_type", "code")
        module: str = raw.get("module") or ""
        # Fall back to signature, then qualified name, then bare name so
        # description is never empty (the validator requires it).
        description: str = (
            raw.get("documentation")
            or raw.get("signature")
            or raw.get("qualified_name")
            or raw.get("name")
            or ""
        )

        # Source location
        source_file: str = raw.get("source_file") or ""
        source_line: int = raw.get("source_line") or 0

        # Tags
        tags: list[str] = raw.get("tags") or []

        # Extensible metadata — include extra Java-specific fields
        metadata: dict[str, Any] = dict(raw.get("metadata") or {})
        if raw.get("embedding_text"):
            metadata["embedding_text"] = raw["embedding_text"]
        if raw.get("signature"):
            metadata["signature"] = raw["signature"]
        if raw.get("qualified_name"):
            metadata["qualified_name"] = raw["qualified_name"]
        if raw.get("parent_id"):
            metadata["parent_id"] = raw["parent_id"]

        return MetadataArtifact(
            id=artifact_id,
            name=name,
            type=artifact_type,
            source_type=source_type,
            language=language,
            module=module,
            description=description,
            tags=tags,
            source_file=source_file,
            source_line=source_line,
            metadata=metadata,
        )
