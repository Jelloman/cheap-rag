"""TypeScript code metadata extractor.

Extracts metadata from TypeScript source files using regex patterns.
Focuses on interfaces, classes, types, enums, and functions.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from .base import MetadataArtifact


class TypeScriptExtractor:
    """Extract metadata from TypeScript source files.

    Uses regex-based parsing to extract common TypeScript constructs.
    For Phase 1, this provides basic extraction without requiring heavy AST parsing.
    """

    # Regex patterns for TypeScript constructs
    INTERFACE_PATTERN = re.compile(
        r"(?:export\s+)?interface\s+(\w+)(?:\s+extends\s+([\w\s,<>]+?))?\s*\{",
        re.MULTILINE,
    )

    CLASS_PATTERN = re.compile(
        r"(?:export\s+)?(?:abstract\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([\w\s,]+?))?\s*\{",
        re.MULTILINE,
    )

    TYPE_PATTERN = re.compile(
        r"(?:export\s+)?type\s+(\w+)\s*=\s*(.+?);",
        re.MULTILINE | re.DOTALL,
    )

    ENUM_PATTERN = re.compile(
        r"(?:export\s+)?enum\s+(\w+)\s*\{",
        re.MULTILINE,
    )

    FUNCTION_PATTERN = re.compile(
        r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*(?:<[^>]+>)?\s*\(([^)]*)\)(?:\s*:\s*([^\{]+))?\s*\{",
        re.MULTILINE,
    )

    # Pattern to extract JSDoc comments
    JSDOC_PATTERN = re.compile(
        r"/\*\*(.*?)\*/",
        re.DOTALL,
    )

    def extract_metadata(self, source_path: Path) -> list[MetadataArtifact]:
        """Extract metadata from TypeScript source files.

        Args:
            source_path: Path to TypeScript source file or directory.

        Returns:
            List of extracted metadata artifacts.
        """
        artifacts: list[MetadataArtifact] = []

        if source_path.is_file():
            if source_path.suffix in (".ts", ".tsx"):
                artifacts.extend(self._extract_from_file(source_path))
        elif source_path.is_dir():
            # Recursively find all .ts and .tsx files
            for ts_file in source_path.rglob("*.ts"):
                try:
                    artifacts.extend(self._extract_from_file(ts_file))
                except Exception as e:
                    print(f"Error extracting from {ts_file}: {e}")

            for tsx_file in source_path.rglob("*.tsx"):
                try:
                    artifacts.extend(self._extract_from_file(tsx_file))
                except Exception as e:
                    print(f"Error extracting from {tsx_file}: {e}")

        return artifacts

    def _extract_from_file(self, file_path: Path) -> list[MetadataArtifact]:
        """Extract metadata from a single TypeScript file.

        Args:
            file_path: Path to TypeScript source file.

        Returns:
            List of metadata artifacts.
        """
        artifacts: list[MetadataArtifact] = []

        try:
            # Read source code
            source_code = file_path.read_text(encoding="utf-8")

            # Derive module name
            module_name = self._derive_module_name(file_path)

            # Remove comments for cleaner parsing
            # (but keep JSDoc for extraction)
            code_for_extraction = self._strip_line_comments(source_code)

            # Extract interfaces
            artifacts.extend(
                self._extract_interfaces(code_for_extraction, module_name, file_path, source_code)
            )

            # Extract classes
            artifacts.extend(
                self._extract_classes(code_for_extraction, module_name, file_path, source_code)
            )

            # Extract type aliases
            artifacts.extend(
                self._extract_types(code_for_extraction, module_name, file_path, source_code)
            )

            # Extract enums
            artifacts.extend(
                self._extract_enums(code_for_extraction, module_name, file_path, source_code)
            )

            # Extract functions
            artifacts.extend(
                self._extract_functions(code_for_extraction, module_name, file_path, source_code)
            )

        except Exception as e:
            print(f"Failed to parse {file_path}: {e}")

        return artifacts

    def _derive_module_name(self, file_path: Path) -> str:
        """Derive module name from file path.

        Args:
            file_path: Path to TypeScript file.

        Returns:
            Module name (e.g., "package/subpackage/module").
        """
        parts = file_path.parts
        # Find the start of the package (typically 'src' or similar)
        try:
            if "src" in parts:
                start_idx = parts.index("src") + 1
            else:
                start_idx = len(parts) - 2 if len(parts) > 1 else 0

            module_parts = list(parts[start_idx:])
            # Remove file extension
            if module_parts and module_parts[-1].endswith((".ts", ".tsx")):
                module_parts[-1] = module_parts[-1].rsplit(".", 1)[0]

            # Use forward slash for TS module paths
            return "/".join(module_parts) if module_parts else "unknown"
        except (ValueError, IndexError):
            return file_path.stem

    def _strip_line_comments(self, source_code: str) -> str:
        """Remove line comments but keep JSDoc comments.

        Args:
            source_code: Original source code.

        Returns:
            Code with line comments removed.
        """
        # Remove // comments
        lines = source_code.split("\n")
        cleaned_lines: list[str] = []
        for line in lines:
            # Find // outside of strings (simple approach)
            comment_pos = line.find("//")
            if comment_pos != -1:
                # Simple check: if there's a quote before //, it might be in a string
                before_comment = line[:comment_pos]
                if before_comment.count('"') % 2 == 0 and before_comment.count("'") % 2 == 0:
                    cleaned_lines.append(line[:comment_pos])
                else:
                    cleaned_lines.append(line)
            else:
                cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

    def _find_preceding_jsdoc(self, source_code: str, match_start: int) -> str:
        """Find JSDoc comment preceding a construct.

        Args:
            source_code: Full source code.
            match_start: Start position of the construct.

        Returns:
            Extracted JSDoc text, or empty string.
        """
        # Look backwards for the nearest JSDoc comment
        preceding_text = source_code[:match_start]
        jsdoc_matches = list(self.JSDOC_PATTERN.finditer(preceding_text))

        if jsdoc_matches:
            # Get the last match (closest to our construct)
            last_match = jsdoc_matches[-1]
            # Check if it's reasonably close (within 200 chars)
            if match_start - last_match.end() < 200:
                return self._clean_jsdoc(last_match.group(1))

        return ""

    def _clean_jsdoc(self, jsdoc_text: str) -> str:
        """Clean JSDoc comment text.

        Args:
            jsdoc_text: Raw JSDoc text.

        Returns:
            Cleaned text.
        """
        lines = jsdoc_text.split("\n")
        cleaned_lines: list[str] = []

        for line in lines:
            line = line.strip()
            # Remove leading asterisks
            if line.startswith("*"):
                line = line[1:].strip()
            # Skip JSDoc tags (for now)
            if line.startswith("@"):
                continue
            if line:
                cleaned_lines.append(line)

        return " ".join(cleaned_lines)

    def _extract_interfaces(
        self, code: str, module_name: str, file_path: Path, source_code: str
    ) -> list[MetadataArtifact]:
        """Extract interface definitions.

        Args:
            code: Cleaned source code.
            module_name: Module name.
            file_path: Source file path.
            source_code: Original source code (for JSDoc extraction).

        Returns:
            List of interface artifacts.
        """
        artifacts: list[MetadataArtifact] = []

        for match in self.INTERFACE_PATTERN.finditer(code):
            interface_name = match.group(1)
            extends_clause = match.group(2)

            qualified_name = f"{module_name}.{interface_name}" if module_name else interface_name

            # Extract JSDoc
            jsdoc = self._find_preceding_jsdoc(source_code, match.start())

            # Parse extends
            relations: list[str] = []
            if extends_clause:
                # Split by comma and clean
                extended: list[str] = [ext.strip() for ext in extends_clause.split(",")]
                relations = [f"extends {ext}" for ext in extended]

            # Line number (approximate)
            line_number = source_code[: match.start()].count("\n") + 1

            # Build tags
            tags = ["typescript", "code", "interface"]
            if "cheap" in module_name.lower():
                tags.append("cheap")

            artifact = MetadataArtifact(
                id=self._generate_id(qualified_name, "interface"),
                name=interface_name,
                type="interface",
                source_type="code",
                language="typescript",
                module=module_name,
                description=jsdoc or f"Interface {interface_name}",
                relations=relations,
                tags=tags,
                source_file=str(file_path),
                source_line=line_number,
            )

            artifacts.append(artifact)

        return artifacts

    def _extract_classes(
        self, code: str, module_name: str, file_path: Path, source_code: str
    ) -> list[MetadataArtifact]:
        """Extract class definitions.

        Args:
            code: Cleaned source code.
            module_name: Module name.
            file_path: Source file path.
            source_code: Original source code (for JSDoc extraction).

        Returns:
            List of class artifacts.
        """
        artifacts: list[MetadataArtifact] = []

        for match in self.CLASS_PATTERN.finditer(code):
            class_name = match.group(1)
            extends_class = match.group(2)
            implements_clause = match.group(3)

            qualified_name = f"{module_name}.{class_name}" if module_name else class_name

            # Extract JSDoc
            jsdoc = self._find_preceding_jsdoc(source_code, match.start())

            # Parse extends/implements
            relations: list[str] = []
            if extends_class:
                relations.append(f"extends {extends_class}")
            if implements_clause:
                implemented: list[str] = [impl.strip() for impl in implements_clause.split(",")]
                relations.extend([f"implements {impl}" for impl in implemented])

            # Line number
            line_number = source_code[: match.start()].count("\n") + 1

            # Build tags
            tags = ["typescript", "code", "class"]
            if "cheap" in module_name.lower():
                tags.append("cheap")

            artifact = MetadataArtifact(
                id=self._generate_id(qualified_name, "class"),
                name=class_name,
                type="class",
                source_type="code",
                language="typescript",
                module=module_name,
                description=jsdoc or f"Class {class_name}",
                relations=relations,
                tags=tags,
                source_file=str(file_path),
                source_line=line_number,
            )

            artifacts.append(artifact)

        return artifacts

    def _extract_types(
        self, code: str, module_name: str, file_path: Path, source_code: str
    ) -> list[MetadataArtifact]:
        """Extract type alias definitions.

        Args:
            code: Cleaned source code.
            module_name: Module name.
            file_path: Source file path.
            source_code: Original source code (for JSDoc extraction).

        Returns:
            List of type artifacts.
        """
        artifacts: list[MetadataArtifact] = []

        for match in self.TYPE_PATTERN.finditer(code):
            type_name = match.group(1)
            type_definition = match.group(2).strip()

            qualified_name = f"{module_name}.{type_name}" if module_name else type_name

            # Extract JSDoc
            jsdoc = self._find_preceding_jsdoc(source_code, match.start())

            # Line number
            line_number = source_code[: match.start()].count("\n") + 1

            # Build tags
            tags = ["typescript", "code", "type"]

            # Metadata
            metadata: dict[str, Any] = {
                "type_definition": type_definition[:100],  # Truncate long definitions
            }

            artifact = MetadataArtifact(
                id=self._generate_id(qualified_name, "type"),
                name=type_name,
                type="type",
                source_type="code",
                language="typescript",
                module=module_name,
                description=jsdoc or f"Type {type_name}",
                tags=tags,
                source_file=str(file_path),
                source_line=line_number,
                metadata=metadata,
            )

            artifacts.append(artifact)

        return artifacts

    def _extract_enums(
        self, code: str, module_name: str, file_path: Path, source_code: str
    ) -> list[MetadataArtifact]:
        """Extract enum definitions.

        Args:
            code: Cleaned source code.
            module_name: Module name.
            file_path: Source file path.
            source_code: Original source code (for JSDoc extraction).

        Returns:
            List of enum artifacts.
        """
        artifacts: list[MetadataArtifact] = []

        for match in self.ENUM_PATTERN.finditer(code):
            enum_name = match.group(1)

            qualified_name = f"{module_name}.{enum_name}" if module_name else enum_name

            # Extract JSDoc
            jsdoc = self._find_preceding_jsdoc(source_code, match.start())

            # Line number
            line_number = source_code[: match.start()].count("\n") + 1

            # Build tags
            tags = ["typescript", "code", "enum"]

            artifact = MetadataArtifact(
                id=self._generate_id(qualified_name, "enum"),
                name=enum_name,
                type="enum",
                source_type="code",
                language="typescript",
                module=module_name,
                description=jsdoc or f"Enum {enum_name}",
                tags=tags,
                source_file=str(file_path),
                source_line=line_number,
            )

            artifacts.append(artifact)

        return artifacts

    def _extract_functions(
        self, code: str, module_name: str, file_path: Path, source_code: str
    ) -> list[MetadataArtifact]:
        """Extract function definitions.

        Args:
            code: Cleaned source code.
            module_name: Module name.
            file_path: Source file path.
            source_code: Original source code (for JSDoc extraction).

        Returns:
            List of function artifacts.
        """
        artifacts: list[MetadataArtifact] = []

        for match in self.FUNCTION_PATTERN.finditer(code):
            func_name = match.group(1)
            params = match.group(2)
            return_type = match.group(3)

            qualified_name = f"{module_name}.{func_name}" if module_name else func_name

            # Extract JSDoc
            jsdoc = self._find_preceding_jsdoc(source_code, match.start())

            # Line number
            line_number = source_code[: match.start()].count("\n") + 1

            # Build tags
            tags = ["typescript", "code", "function"]

            # Check if async
            preceding_text = source_code[max(0, match.start() - 50) : match.start()]
            if "async" in preceding_text:
                tags.append("async")

            # Metadata
            metadata: dict[str, Any] = {}
            if params:
                metadata["parameters"] = params.strip()
            if return_type:
                metadata["return_type"] = return_type.strip()

            artifact = MetadataArtifact(
                id=self._generate_id(qualified_name, "function"),
                name=func_name,
                type="function",
                source_type="code",
                language="typescript",
                module=module_name,
                description=jsdoc or f"Function {func_name}",
                tags=tags,
                source_file=str(file_path),
                source_line=line_number,
                metadata=metadata,
            )

            artifacts.append(artifact)

        return artifacts

    def _generate_id(self, qualified_name: str, artifact_type: str) -> str:
        """Generate unique artifact ID.

        Args:
            qualified_name: Fully qualified name.
            artifact_type: Artifact type.

        Returns:
            Unique identifier.
        """
        content = f"typescript:{artifact_type}:{qualified_name}"
        hash_digest = hashlib.sha256(content.encode()).hexdigest()[:16]
        return f"typescript_{artifact_type}_{hash_digest}"

    def language(self) -> str:
        """Return the language identifier.

        Returns:
            "typescript"
        """
        return "typescript"
