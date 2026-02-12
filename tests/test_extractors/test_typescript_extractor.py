"""Tests for TypeScript metadata extractor."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from src.extractors.typescript_extractor import TypeScriptExtractor


@pytest.fixture
def temp_typescript_file(tmp_path: Path) -> Path:
    """Create a temporary TypeScript file for testing.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Path to temporary TypeScript file.
    """
    test_code = dedent('''
        /**
         * A simple interface for testing.
         */
        export interface ITestInterface extends BaseInterface {
            name: string;
            value: number;
        }

        /**
         * A test class.
         */
        export class TestClass extends BaseClass implements ITestInterface {
            private name: string;
            public value: number;

            constructor(name: string, value: number) {
                super();
                this.name = name;
                this.value = value;
            }
        }

        /**
         * A type alias.
         */
        export type StringOrNumber = string | number;

        /**
         * An enum definition.
         */
        export enum Color {
            Red,
            Green,
            Blue
        }

        /**
         * A simple function.
         */
        export function greet(name: string): string {
            return `Hello, ${name}`;
        }

        /**
         * An async function.
         */
        export async function fetchData(url: string): Promise<string> {
            return "data";
        }
    ''')

    file_path = tmp_path / "test.ts"
    file_path.write_text(test_code)
    return file_path


def test_extract_interfaces(temp_typescript_file: Path) -> None:
    """Test extracting interface definitions."""
    extractor = TypeScriptExtractor()
    artifacts = extractor.extract_metadata(temp_typescript_file)

    interfaces = [a for a in artifacts if a.type == "interface"]

    assert len(interfaces) == 1
    interface = interfaces[0]
    assert interface.name == "ITestInterface"
    assert interface.language == "typescript"
    assert interface.source_type == "code"
    assert "A simple interface for testing" in interface.description
    assert "extends BaseInterface" in interface.relations
    assert "typescript" in interface.tags
    assert "interface" in interface.tags


def test_extract_classes(temp_typescript_file: Path) -> None:
    """Test extracting class definitions."""
    extractor = TypeScriptExtractor()
    artifacts = extractor.extract_metadata(temp_typescript_file)

    classes = [a for a in artifacts if a.type == "class"]

    assert len(classes) == 1
    cls = classes[0]
    assert cls.name == "TestClass"
    assert cls.language == "typescript"
    assert "A test class" in cls.description
    assert "extends BaseClass" in cls.relations
    assert "implements ITestInterface" in cls.relations
    assert "typescript" in cls.tags
    assert "class" in cls.tags


def test_extract_types(temp_typescript_file: Path) -> None:
    """Test extracting type alias definitions."""
    extractor = TypeScriptExtractor()
    artifacts = extractor.extract_metadata(temp_typescript_file)

    types = [a for a in artifacts if a.type == "type"]

    assert len(types) == 1
    type_alias = types[0]
    assert type_alias.name == "StringOrNumber"
    assert type_alias.language == "typescript"
    assert "A type alias" in type_alias.description
    assert "string | number" in type_alias.metadata.get("type_definition", "")
    assert "typescript" in type_alias.tags
    assert "type" in type_alias.tags


def test_extract_enums(temp_typescript_file: Path) -> None:
    """Test extracting enum definitions."""
    extractor = TypeScriptExtractor()
    artifacts = extractor.extract_metadata(temp_typescript_file)

    enums = [a for a in artifacts if a.type == "enum"]

    assert len(enums) == 1
    enum = enums[0]
    assert enum.name == "Color"
    assert enum.language == "typescript"
    assert "An enum definition" in enum.description
    assert "typescript" in enum.tags
    assert "enum" in enum.tags


def test_extract_functions(temp_typescript_file: Path) -> None:
    """Test extracting function definitions."""
    extractor = TypeScriptExtractor()
    artifacts = extractor.extract_metadata(temp_typescript_file)

    functions = [a for a in artifacts if a.type == "function"]

    assert len(functions) == 2
    function_names = {f.name for f in functions}
    assert function_names == {"greet", "fetchData"}

    # Check regular function
    greet = next(f for f in functions if f.name == "greet")
    assert "A simple function" in greet.description
    assert greet.metadata.get("parameters") == "name: string"
    assert greet.metadata.get("return_type") == "string"

    # Check async function
    fetch = next(f for f in functions if f.name == "fetchData")
    assert "async" in fetch.tags
    assert "An async function" in fetch.description


def test_language_identifier() -> None:
    """Test language identifier."""
    extractor = TypeScriptExtractor()
    assert extractor.language() == "typescript"


def test_extract_from_directory(tmp_path: Path) -> None:
    """Test extracting from a directory."""
    # Create multiple TypeScript files
    (tmp_path / "file1.ts").write_text("export interface InterfaceOne {}")
    (tmp_path / "file2.tsx").write_text("export interface InterfaceTwo {}")
    (tmp_path / "not_ts.txt").write_text("ignore this")

    extractor = TypeScriptExtractor()
    artifacts = extractor.extract_metadata(tmp_path)

    interfaces = [a for a in artifacts if a.type == "interface"]
    assert len(interfaces) == 2
    assert {i.name for i in interfaces} == {"InterfaceOne", "InterfaceTwo"}


def test_jsdoc_extraction(tmp_path: Path) -> None:
    """Test JSDoc comment extraction."""
    code = dedent('''
        /**
         * This is a multi-line
         * JSDoc comment.
         * @param x - ignored
         */
        export interface MyInterface {
            value: number;
        }
    ''')

    file_path = tmp_path / "test.ts"
    file_path.write_text(code)

    extractor = TypeScriptExtractor()
    artifacts = extractor.extract_metadata(file_path)

    assert len(artifacts) == 1
    assert "multi-line" in artifacts[0].description
    assert "JSDoc comment" in artifacts[0].description
    # @param should be filtered out
    assert "@param" not in artifacts[0].description


def test_module_name_derivation(tmp_path: Path) -> None:
    """Test module name derivation from path."""
    # Create nested structure
    src_dir = tmp_path / "src" / "components" / "ui"
    src_dir.mkdir(parents=True)
    test_file = src_dir / "Button.ts"
    test_file.write_text("export class Button {}")

    extractor = TypeScriptExtractor()
    artifacts = extractor.extract_metadata(test_file)

    assert len(artifacts) == 1
    assert "components" in artifacts[0].module
    assert "ui" in artifacts[0].module
    # TypeScript uses forward slashes
    assert "/" in artifacts[0].module


def test_interface_multiple_extends(tmp_path: Path) -> None:
    """Test interface with multiple extends."""
    code = "export interface Multi extends Base1, Base2, Base3 {}"

    file_path = tmp_path / "test.ts"
    file_path.write_text(code)

    extractor = TypeScriptExtractor()
    artifacts = extractor.extract_metadata(file_path)

    assert len(artifacts) == 1
    interface = artifacts[0]
    # Should capture all extended interfaces
    assert len(interface.relations) == 3
    assert "extends Base1" in interface.relations
    assert "extends Base2" in interface.relations
    assert "extends Base3" in interface.relations


def test_class_implements_multiple(tmp_path: Path) -> None:
    """Test class implementing multiple interfaces."""
    code = "export class MyClass implements IFoo, IBar, IBaz {}"

    file_path = tmp_path / "test.ts"
    file_path.write_text(code)

    extractor = TypeScriptExtractor()
    artifacts = extractor.extract_metadata(file_path)

    assert len(artifacts) == 1
    cls = artifacts[0]
    # Should have 3 implements relations
    impl_relations = [r for r in cls.relations if "implements" in r]
    assert len(impl_relations) == 3
