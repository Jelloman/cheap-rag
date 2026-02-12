"""Fix MetadataArtifact instances in test files to include source_type."""

import re
from pathlib import Path


def fix_artifact_creation(content: str) -> str:
    """Add source_type to MetadataArtifact creations that are missing it."""

    # Pattern to match MetadataArtifact( with parameters but no source_type
    # This is a simplified fix - adds source_type based on language field
    lines = content.split('\n')
    result_lines = []

    in_artifact = False
    artifact_lines = []
    indent = ""

    for line in lines:
        # Check if line starts a MetadataArtifact creation
        if 'MetadataArtifact(' in line and 'source_type' not in line:
            in_artifact = True
            artifact_lines = [line]
            # Get indentation
            indent = line[:len(line) - len(line.lstrip())]
        elif in_artifact:
            artifact_lines.append(line)
            # Check if this closes the MetadataArtifact
            if ')' in line and not line.strip().endswith(','):
                # Found end - check if source_type was added
                full_artifact = '\n'.join(artifact_lines)
                if 'source_type' not in full_artifact:
                    # Determine source_type based on context
                    if 'postgresql' in full_artifact or 'sqlite' in full_artifact or 'mariadb' in full_artifact:
                        source_type = 'database'
                    elif 'java' in full_artifact or 'typescript' in full_artifact or 'python' in full_artifact:
                        source_type = 'code'
                    else:
                        # Default to database
                        source_type = 'database'

                    # Find where to insert source_type (after 'type' field)
                    modified_lines = []
                    for aline in artifact_lines:
                        modified_lines.append(aline)
                        # Add source_type after type= line
                        if 'type=' in aline and 'source_type' not in aline:
                            # Add source_type on next line
                            modified_lines.append(f'{indent}    source_type="{source_type}",')

                    result_lines.extend(modified_lines)
                else:
                    result_lines.extend(artifact_lines)

                in_artifact = False
                artifact_lines = []
        else:
            result_lines.append(line)

    return '\n'.join(result_lines)


def main():
    """Fix all test files."""
    test_dir = Path("tests/test_generation")

    for test_file in test_dir.glob("test_*.py"):
        print(f"Fixing {test_file}...")
        content = test_file.read_text()
        fixed_content = fix_artifact_creation(content)
        test_file.write_text(fixed_content)
        print(f"  Done")


if __name__ == "__main__":
    main()
