"""Tests for Python metadata extractor."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from src.extractors.python_extractor import PythonExtractor


@pytest.fixture
def temp_python_file(tmp_path: Path) -> Path:
    """Create a temporary Python file for testing.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Path to temporary Python file.
    """
    test_code = dedent('''
        """Module docstring."""

        from typing import Protocol

        class DataClass:
            """A simple data class."""

            def __init__(self, name: str, value: int):
                """Initialize the data class.

                Args:
                    name: The name.
                    value: The value.
                """
                self.name = name
                self.value = value

            def get_name(self) -> str:
                """Get the name."""
                return self.name

        class MyProtocol(Protocol):
            """A protocol definition."""

            def method(self) -> None:
                """Protocol method."""
                ...

        async def async_function(param: str) -> bool:
            """An async function.

            Args:
                param: A parameter.

            Returns:
                True always.
            """
            return True

        def regular_function(x: int, y: int) -> int:
            """Add two numbers."""
            return x + y
    ''')

    file_path = tmp_path / "test_module.py"
    file_path.write_text(test_code)
    return file_path


def test_extract_classes(temp_python_file: Path) -> None:
    """Test extracting class definitions."""
    extractor = PythonExtractor()
    artifacts = extractor.extract_metadata(temp_python_file)

    # Filter classes
    classes = [a for a in artifacts if a.type == "class"]

    assert len(classes) == 2
    assert {c.name for c in classes} == {"DataClass", "MyProtocol"}

    # Check DataClass
    data_class = next(c for c in classes if c.name == "DataClass")
    assert data_class.language == "python"
    assert data_class.source_type == "code"
    assert "A simple data class" in data_class.description
    assert "python" in data_class.tags
    assert "class" in data_class.tags

    # Check Protocol
    protocol_class = next(c for c in classes if c.name == "MyProtocol")
    assert "protocol" in protocol_class.tags


def test_extract_methods(temp_python_file: Path) -> None:
    """Test extracting method definitions."""
    extractor = PythonExtractor()
    artifacts = extractor.extract_metadata(temp_python_file)

    # Filter methods
    methods = [a for a in artifacts if a.type == "method"]

    assert len(methods) >= 3  # __init__, get_name, method
    method_names = {m.name for m in methods}
    assert "__init__" in method_names
    assert "get_name" in method_names
    assert "method" in method_names

    # Check __init__
    init_method = next(m for m in methods if m.name == "__init__")
    assert "constructor" in init_method.tags or "magic" in init_method.tags
    assert init_method.metadata["arguments"] == ["self", "name", "value"]

    # Check get_name
    get_name = next(m for m in methods if m.name == "get_name")
    assert get_name.metadata.get("return_type") == "str"


def test_extract_functions(temp_python_file: Path) -> None:
    """Test extracting function definitions."""
    extractor = PythonExtractor()
    artifacts = extractor.extract_metadata(temp_python_file)

    # Filter functions
    functions = [a for a in artifacts if a.type == "function"]

    assert len(functions) == 2
    function_names = {f.name for f in functions}
    assert function_names == {"async_function", "regular_function"}

    # Check async function
    async_func = next(f for f in functions if f.name == "async_function")
    assert "async" in async_func.tags
    assert async_func.metadata["is_async"] is True
    assert async_func.metadata.get("return_type") == "bool"

    # Check regular function
    regular_func = next(f for f in functions if f.name == "regular_function")
    assert regular_func.metadata["is_async"] is False
    assert regular_func.metadata["arguments"] == ["x", "y"]


def test_language_identifier() -> None:
    """Test language identifier."""
    extractor = PythonExtractor()
    assert extractor.language() == "python"


def test_extract_from_directory(tmp_path: Path) -> None:
    """Test extracting from a directory."""
    # Create multiple Python files
    (tmp_path / "module1.py").write_text("class ClassOne:\n    pass")
    (tmp_path / "module2.py").write_text("class ClassTwo:\n    pass")
    (tmp_path / "not_python.txt").write_text("ignore this")

    extractor = PythonExtractor()
    artifacts = extractor.extract_metadata(tmp_path)

    classes = [a for a in artifacts if a.type == "class"]
    assert len(classes) == 2
    assert {c.name for c in classes} == {"ClassOne", "ClassTwo"}


def test_syntax_error_handling(tmp_path: Path) -> None:
    """Test handling of syntax errors."""
    bad_file = tmp_path / "bad.py"
    bad_file.write_text("class Broken\n    # missing colon")

    extractor = PythonExtractor()
    # Should not raise, just log and continue
    artifacts = extractor.extract_metadata(bad_file)
    assert len(artifacts) == 0


def test_module_name_derivation(tmp_path: Path) -> None:
    """Test module name derivation from path."""
    # Create nested structure
    src_dir = tmp_path / "src" / "mypackage" / "subpackage"
    src_dir.mkdir(parents=True)
    test_file = src_dir / "mymodule.py"
    test_file.write_text("class TestClass:\n    pass")

    extractor = PythonExtractor()
    artifacts = extractor.extract_metadata(test_file)

    assert len(artifacts) == 1
    assert "mypackage" in artifacts[0].module
    assert "subpackage" in artifacts[0].module
