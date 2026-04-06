"""Java code metadata extractor (RETIRED).

This module is retired. Use src.extractors.java_extractor_jar.JavaExtractorJar instead,
which delegates to the Java-based extractor JAR for full-fidelity JavaParser analysis.
"""

# ruff: noqa

import hashlib
from pathlib import Path
from typing import Any

import javalang  # type: ignore[import-untyped]

from .base import MetadataArtifact


class JavaExtractor:
    """Extract metadata from Java source files.

    Phase 1 scope: Extract class-level and field-level metadata only.
    Focus on CHEAP core interfaces (Catalog, Hierarchy, Entity, Aspect, Property).
    """

    def extract_metadata(self, source_path: Path) -> list[MetadataArtifact]:
        """Extract metadata from Java source files.

        Args:
            source_path: Path to Java source file or directory.

        Returns:
            List of extracted metadata artifacts.
        """
        artifacts: list[MetadataArtifact] = []

        if source_path.is_file():
            if source_path.suffix == ".java":
                artifacts.extend(self._extract_from_file(source_path))
        elif source_path.is_dir():
            # Recursively find all .java files, skipping test/build directories
            _exclude = {"test", "build", "generated", "target"}
            for java_file in source_path.rglob("*.java"):
                if any(part in _exclude for part in java_file.parts):
                    continue
                try:
                    artifacts.extend(self._extract_from_file(java_file))
                except Exception as e:
                    # Log error but continue processing other files
                    print(f"Error extracting from {java_file}: {e}")

        return artifacts

    def _extract_from_file(self, file_path: Path) -> list[MetadataArtifact]:
        """Extract metadata from a single Java file.

        Args:
            file_path: Path to Java source file.

        Returns:
            List of metadata artifacts.
        """
        artifacts: list[MetadataArtifact] = []

        try:
            # Read source code
            source_code = file_path.read_text(encoding="utf-8")

            # Parse with javalang
            tree = javalang.parse.parse(source_code)  # type: ignore[attr-defined]

            # Get package name
            package_name: str = tree.package.name if tree.package else ""  # type: ignore[attr-defined]

            # Extract from type declarations (classes, interfaces, enums)
            for _path, node in tree.filter(javalang.tree.TypeDeclaration):  # type: ignore[attr-defined]
                artifacts.extend(
                    self._extract_type_declaration(node, package_name, file_path, source_code)  # type: ignore[reportUnknownArgumentType]  # javalang
                )

        except Exception as e:
            # Log and continue
            print(f"Failed to parse {file_path}: {e}")

        return artifacts

    def _extract_type_declaration(
        self,
        node: Any,  # javalang.tree.TypeDeclaration
        package_name: str,
        file_path: Path,
        _source_code: str,
    ) -> list[MetadataArtifact]:
        """Extract metadata from a type declaration (class/interface).

        Args:
            node: Type declaration node.
            package_name: Package name.
            file_path: Source file path.
            source_code: Full source code (for extracting javadoc).

        Returns:
            List of metadata artifacts.
        """
        artifacts: list[MetadataArtifact] = []

        # Determine type
        if isinstance(node, javalang.tree.ClassDeclaration):  # type: ignore[reportUnknownMemberType,reportAttributeAccessIssue]  # javalang
            artifact_type = "class"
        elif isinstance(node, javalang.tree.InterfaceDeclaration):  # type: ignore[reportUnknownMemberType,reportAttributeAccessIssue]  # javalang
            artifact_type = "interface"
        elif isinstance(node, javalang.tree.EnumDeclaration):  # type: ignore[reportUnknownMemberType,reportAttributeAccessIssue]  # javalang
            artifact_type = "enum"
        else:
            return artifacts  # Skip other types

        # Get type name
        type_name = node.name
        qualified_name = f"{package_name}.{type_name}" if package_name else type_name

        # Extract javadoc
        javadoc = self._extract_javadoc(node.documentation)

        # Extract extends/implements
        relations: list[str] = []
        if hasattr(node, "extends") and node.extends:
            relations.append(f"extends {node.extends.name}")
        if hasattr(node, "implements") and node.implements:
            for impl in node.implements:
                relations.append(f"implements {impl.name}")

        # Build tags
        tags = ["java", "code", artifact_type]
        if "cheap" in package_name.lower():
            tags.append("cheap")
        if any(
            keyword in type_name.lower()
            for keyword in ["catalog", "hierarchy", "entity", "aspect", "property"]
        ):
            tags.append("core")

        # Create artifact for type
        type_artifact = MetadataArtifact(
            id=self._generate_id(qualified_name, artifact_type),
            name=type_name,
            type=artifact_type,
            source_type="code",
            language="java",
            module=package_name,
            description=javadoc or f"{artifact_type.capitalize()} {type_name}",
            relations=relations,
            tags=tags,
            source_file=str(file_path),
            source_line=node.position.line if node.position else 0,
        )

        artifacts.append(type_artifact)

        # Extract fields
        artifacts.extend(self._extract_fields(node, qualified_name, file_path))

        return artifacts

    def _extract_fields(
        self,
        type_node: Any,  # javalang.tree.TypeDeclaration
        qualified_type_name: str,
        file_path: Path,
    ) -> list[MetadataArtifact]:
        """Extract field metadata from a type.

        Args:
            type_node: Type declaration node.
            qualified_type_name: Fully qualified type name.
            file_path: Source file path.

        Returns:
            List of field metadata artifacts.
        """
        artifacts: list[MetadataArtifact] = []

        # Get all field declarations
        for field_decl in type_node.fields:
            for declarator in field_decl.declarators:
                field_name = declarator.name
                field_type = (
                    field_decl.type.name
                    if hasattr(field_decl.type, "name")
                    else str(field_decl.type)
                )

                # Extract javadoc
                javadoc = self._extract_javadoc(field_decl.documentation)

                # Build constraints (modifiers)
                constraints: list[str] = []
                if "final" in field_decl.modifiers:
                    constraints.append("final")
                if "static" in field_decl.modifiers:
                    constraints.append("static")

                # Build tags
                tags = ["java", "code", "field"]
                if "public" in field_decl.modifiers:
                    tags.append("public")
                if "private" in field_decl.modifiers:
                    tags.append("private")

                # Create field artifact
                field_artifact = MetadataArtifact(
                    id=self._generate_id(f"{qualified_type_name}.{field_name}", "field"),
                    name=field_name,
                    type="field",
                    source_type="code",
                    language="java",
                    module=qualified_type_name,
                    description=javadoc or f"Field {field_name} of type {field_type}",
                    constraints=constraints,
                    tags=tags,
                    source_file=str(file_path),
                    source_line=field_decl.position.line if field_decl.position else 0,
                    metadata={
                        "field_type": field_type,
                        "modifiers": field_decl.modifiers,
                    },
                )

                artifacts.append(field_artifact)

        return artifacts

    def _extract_javadoc(self, documentation: str | None) -> str:
        """Extract and clean javadoc comment.

        Args:
            documentation: Raw javadoc string.

        Returns:
            Cleaned javadoc text.
        """
        if not documentation:
            return ""

        # Remove javadoc markers and asterisks
        lines = documentation.split("\n")
        cleaned_lines: list[str] = []

        for line in lines:
            # Remove leading/trailing whitespace
            line = line.strip()
            # Remove leading asterisks
            if line.startswith("*"):
                line = line[1:].strip()
            # Skip javadoc tags (for now)
            if line.startswith("@"):
                continue
            if line:
                cleaned_lines.append(line)

        return " ".join(cleaned_lines)

    def _generate_id(self, qualified_name: str, artifact_type: str) -> str:
        """Generate unique artifact ID.

        Args:
            qualified_name: Fully qualified name.
            artifact_type: Artifact type.

        Returns:
            Unique identifier.
        """
        content = f"java:{artifact_type}:{qualified_name}"
        hash_digest = hashlib.sha256(content.encode()).hexdigest()[:16]
        return f"java_{artifact_type}_{hash_digest}"

    def language(self) -> str:
        """Return the language identifier.

        Returns:
            "java"
        """
        return "java"
