"""Python code metadata extractor.

Extracts metadata from Python source files using the built-in ast module.
Focuses on classes, functions, methods, and module-level attributes.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any

from .base import MetadataArtifact


class PythonExtractor:
    """Extract metadata from Python source files.

    Uses Python's built-in ast module for robust parsing.
    Extracts classes, functions, methods, and their docstrings.
    """

    def extract_metadata(self, source_path: Path) -> list[MetadataArtifact]:
        """Extract metadata from Python source files.

        Args:
            source_path: Path to Python source file or directory.

        Returns:
            List of extracted metadata artifacts.
        """
        artifacts: list[MetadataArtifact] = []

        if source_path.is_file():
            if source_path.suffix == ".py":
                artifacts.extend(self._extract_from_file(source_path))
        elif source_path.is_dir():
            # Recursively find all .py files
            for py_file in source_path.rglob("*.py"):
                try:
                    artifacts.extend(self._extract_from_file(py_file))
                except Exception as e:
                    # Log error but continue processing other files
                    print(f"Error extracting from {py_file}: {e}")

        return artifacts

    def _extract_from_file(self, file_path: Path) -> list[MetadataArtifact]:
        """Extract metadata from a single Python file.

        Args:
            file_path: Path to Python source file.

        Returns:
            List of metadata artifacts.
        """
        artifacts: list[MetadataArtifact] = []

        try:
            # Read source code
            source_code = file_path.read_text(encoding="utf-8")

            # Parse with ast
            tree = ast.parse(source_code, filename=str(file_path))

            # Derive module name from file path
            module_name = self._derive_module_name(file_path)

            # Extract from module body
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef):
                    artifacts.extend(self._extract_class(node, module_name, file_path))
                elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    artifacts.append(
                        self._extract_function(node, module_name, file_path, is_method=False)
                    )

        except SyntaxError as e:
            print(f"Syntax error in {file_path}: {e}")
        except Exception as e:
            print(f"Failed to parse {file_path}: {e}")

        return artifacts

    def _derive_module_name(self, file_path: Path) -> str:
        """Derive module name from file path.

        Args:
            file_path: Path to Python file.

        Returns:
            Module name (e.g., "package.subpackage.module").
        """
        # Get relative parts and remove .py extension
        parts = file_path.parts
        # Find the start of the package (typically 'src' or similar)
        try:
            if "src" in parts:
                start_idx = parts.index("src") + 1
            else:
                # Use parent directory as package root
                start_idx = len(parts) - 2 if len(parts) > 1 else 0

            module_parts = parts[start_idx:]
            # Remove .py extension from last part
            if module_parts and module_parts[-1].endswith(".py"):
                module_parts = list(module_parts)
                module_parts[-1] = module_parts[-1][:-3]

            # Remove __init__ if it's the last part
            if module_parts and module_parts[-1] == "__init__":
                module_parts = module_parts[:-1]

            return ".".join(module_parts) if module_parts else "unknown"
        except (ValueError, IndexError):
            # Fallback to file stem
            return file_path.stem

    def _extract_class(
        self,
        node: ast.ClassDef,
        module_name: str,
        file_path: Path,
    ) -> list[MetadataArtifact]:
        """Extract metadata from a class definition.

        Args:
            node: Class definition node.
            module_name: Module name.
            file_path: Source file path.

        Returns:
            List of metadata artifacts (class + methods).
        """
        artifacts: list[MetadataArtifact] = []

        class_name = node.name
        qualified_name = f"{module_name}.{class_name}" if module_name else class_name

        # Extract docstring
        docstring = ast.get_docstring(node) or ""

        # Extract base classes
        relations: list[str] = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                relations.append(f"extends {base.id}")
            elif isinstance(base, ast.Attribute):
                # Handle qualified names like module.ClassName
                relations.append(f"extends {self._get_attribute_name(base)}")

        # Build tags
        tags = ["python", "code", "class"]

        # Check for dataclass decorator
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                if decorator.id in ("dataclass", "frozen"):
                    tags.append("dataclass")
            elif (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Name)
                and decorator.func.id == "dataclass"
            ):
                tags.append("dataclass")

        # Check for Protocol
        if any(
            base
            for base in node.bases
            if (isinstance(base, ast.Name) and base.id == "Protocol")
            or (isinstance(base, ast.Attribute) and base.attr == "Protocol")
        ):
            tags.append("protocol")

        # Create class artifact
        class_artifact = MetadataArtifact(
            id=self._generate_id(qualified_name, "class"),
            name=class_name,
            type="class",
            source_type="code",
            language="python",
            module=module_name,
            description=docstring or f"Class {class_name}",
            relations=relations,
            tags=tags,
            source_file=str(file_path),
            source_line=node.lineno,
        )

        artifacts.append(class_artifact)

        # Extract methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                method_artifact = self._extract_function(
                    item, qualified_name, file_path, is_method=True
                )
                artifacts.append(method_artifact)

        return artifacts

    def _extract_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        module_or_class: str,
        file_path: Path,
        is_method: bool = False,
    ) -> MetadataArtifact:
        """Extract metadata from a function or method.

        Args:
            node: Function definition node.
            module_or_class: Module name or qualified class name.
            file_path: Source file path.
            is_method: True if this is a method, False if module-level function.

        Returns:
            Metadata artifact for the function/method.
        """
        func_name = node.name
        qualified_name = f"{module_or_class}.{func_name}" if module_or_class else func_name

        # Extract docstring
        docstring = ast.get_docstring(node) or ""

        # Extract function signature info
        args = [arg.arg for arg in node.args.args]

        # Build constraints (decorators)
        constraints: list[str] = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                constraints.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                constraints.append(self._get_attribute_name(decorator))

        # Build tags
        artifact_type = "method" if is_method else "function"
        tags = ["python", "code", artifact_type]

        # Check for async
        if isinstance(node, ast.AsyncFunctionDef):
            tags.append("async")

        # Check for special methods
        if is_method:
            if func_name.startswith("__") and func_name.endswith("__"):
                tags.append("magic")
            elif func_name.startswith("_"):
                tags.append("private")
            elif func_name == "__init__":
                tags.append("constructor")

        # Metadata
        metadata: dict[str, Any] = {
            "arguments": args,
            "is_async": isinstance(node, ast.AsyncFunctionDef),
        }

        # Return type annotation
        if node.returns:
            metadata["return_type"] = self._get_annotation_text(node.returns)

        return MetadataArtifact(
            id=self._generate_id(qualified_name, artifact_type),
            name=func_name,
            type=artifact_type,
            source_type="code",
            language="python",
            module=module_or_class,
            description=docstring or f"{artifact_type.capitalize()} {func_name}",
            constraints=constraints,
            tags=tags,
            source_file=str(file_path),
            source_line=node.lineno,
            metadata=metadata,
        )

    def _get_attribute_name(self, node: ast.Attribute) -> str:
        """Get full attribute name from ast.Attribute node.

        Args:
            node: Attribute node.

        Returns:
            Full attribute name (e.g., "module.Class").
        """
        parts = [node.attr]
        current = node.value
        while isinstance(current, ast.Attribute):
            parts.insert(0, current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.insert(0, current.id)
        return ".".join(parts)

    def _get_annotation_text(self, node: ast.expr) -> str:
        """Get text representation of a type annotation.

        Args:
            node: Type annotation node.

        Returns:
            String representation of the type.
        """
        try:
            return ast.unparse(node)
        except Exception:
            # Fallback for complex annotations
            if isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Attribute):
                return self._get_attribute_name(node)
            else:
                return "Unknown"

    def _generate_id(self, qualified_name: str, artifact_type: str) -> str:
        """Generate unique artifact ID.

        Args:
            qualified_name: Fully qualified name.
            artifact_type: Artifact type.

        Returns:
            Unique identifier.
        """
        content = f"python:{artifact_type}:{qualified_name}"
        hash_digest = hashlib.sha256(content.encode()).hexdigest()[:16]
        return f"python_{artifact_type}_{hash_digest}"

    def language(self) -> str:
        """Return the language identifier.

        Returns:
            "python"
        """
        return "python"
