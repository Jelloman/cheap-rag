"""Prompt templates for LLM-powered answer generation."""

from __future__ import annotations

from src.extractors.base import MetadataArtifact
from src.retrieval.semantic_search import SearchResult


# System message templates

SYSTEM_MESSAGE_BASE = """You are a helpful assistant that answers questions about software metadata, database schemas, and code definitions.

CRITICAL INSTRUCTIONS:
1. Use ONLY the information provided in the context below
2. Do NOT use your general knowledge or training data
3. If the context doesn't contain enough information to answer the question, respond with "I don't know based on the provided context"
4. ALWAYS cite your sources using the format: [ArtifactName] (ID: artifact_id)
5. Be precise and accurate in your answers

When answering:
- Focus on the most relevant artifacts
- Include specific details like data types, constraints, relationships
- Cite every fact with the artifact it came from
- If multiple artifacts are relevant, synthesize information from all of them
"""

SYSTEM_MESSAGE_QWEN = """You are Qwen, a helpful AI assistant specializing in software metadata analysis.

CRITICAL RULES:
1. Answer ONLY using the provided context
2. Do NOT hallucinate or use external knowledge
3. If uncertain, say "I don't know based on the provided context"
4. ALWAYS cite sources: [ArtifactName] (ID: artifact_id)
5. Be concise and technical

Format:
- Use bullet points for multiple items
- Include relevant technical details (types, constraints)
- Cite each fact with its source artifact
"""

SYSTEM_MESSAGE_CLAUDE = """You are Claude, an AI assistant helping developers understand their codebase and database schemas through metadata analysis.

Essential Guidelines:
1. Use ONLY the provided context - do not rely on general knowledge
2. If the context is insufficient to answer, explicitly state: "I don't know based on the provided context"
3. Cite every fact using this format: [ArtifactName] (ID: artifact_id)
4. Be thorough but concise
5. Prioritize accuracy over completeness

Answer Structure:
- Start with a direct answer to the question
- Provide supporting details with citations
- For database questions, include schema, types, and relationships
- For code questions, include structure, types, and documentation
"""


def get_system_message(provider: str = "ollama") -> str:
    """Get system message for the specified LLM provider.

    Args:
        provider: LLM provider ("ollama", "anthropic", "hybrid").

    Returns:
        System message string.
    """
    if provider == "anthropic":
        return SYSTEM_MESSAGE_CLAUDE
    elif provider == "ollama":
        return SYSTEM_MESSAGE_QWEN
    else:
        # Default/hybrid uses base message
        return SYSTEM_MESSAGE_BASE


# Context formatting


def format_artifact_context(artifact: MetadataArtifact) -> str:
    """Format a single artifact as context for the LLM.

    Args:
        artifact: Metadata artifact to format.

    Returns:
        Formatted context string.
    """
    lines = []

    # Header with artifact type and name
    lines.append(f"### {artifact.type.upper()}: {artifact.name}")
    lines.append(f"**ID:** {artifact.id}")
    lines.append(f"**Language:** {artifact.language}")
    lines.append(f"**Module/Schema:** {artifact.module}")

    # Source information (for code artifacts)
    if artifact.source_file:
        location = f"{artifact.source_file}"
        if artifact.source_line:
            location += f":{artifact.source_line}"
        lines.append(f"**Source:** {location}")

    # Description
    if artifact.description:
        lines.append(f"**Description:** {artifact.description}")

    # Type-specific details
    if artifact.type == "table":
        if artifact.relations:
            lines.append(f"**Related Tables:** {', '.join(artifact.relations)}")

    elif artifact.type == "column":
        col_type = artifact.metadata.get("column_type", "")
        nullable = artifact.metadata.get("nullable", True)
        table_name = artifact.metadata.get("table_name", "")

        lines.append(f"**Table:** {table_name}")
        lines.append(f"**Type:** {col_type}")
        lines.append(f"**Nullable:** {'Yes' if nullable else 'No'}")

        if artifact.metadata.get("primary_key"):
            lines.append("**Primary Key:** Yes")
        if artifact.metadata.get("foreign_key"):
            lines.append(f"**Foreign Key:** {artifact.metadata.get('foreign_key')}")
        if artifact.metadata.get("unique"):
            lines.append("**Unique:** Yes")
        if artifact.metadata.get("indexed"):
            lines.append("**Indexed:** Yes")

    elif artifact.type == "relationship":
        from_table = artifact.metadata.get("from_table", "")
        to_table = artifact.metadata.get("to_table", "")
        cardinality = artifact.metadata.get("cardinality", "")

        lines.append(f"**From Table:** {from_table}")
        lines.append(f"**To Table:** {to_table}")
        if cardinality:
            lines.append(f"**Cardinality:** {cardinality}")

    elif artifact.type == "index":
        table_name = artifact.metadata.get("table_name", "")
        unique = artifact.metadata.get("unique", False)
        columns = artifact.metadata.get("columns", [])

        lines.append(f"**Table:** {table_name}")
        lines.append(f"**Type:** {'UNIQUE' if unique else 'NON-UNIQUE'}")
        if columns:
            lines.append(f"**Columns:** {', '.join(str(c) for c in columns if c)}")

    # Constraints
    if artifact.constraints:
        lines.append(f"**Constraints:** {', '.join(artifact.constraints)}")

    # Tags
    if artifact.tags:
        lines.append(f"**Tags:** {', '.join(artifact.tags)}")

    # Relations (for code artifacts)
    if artifact.relations and artifact.type not in ["table", "relationship"]:
        lines.append(f"**Related Artifacts:** {', '.join(artifact.relations)}")

    # Examples
    if artifact.examples:
        lines.append("**Examples:**")
        for example in artifact.examples:
            lines.append(f"  - {example}")

    return "\n".join(lines)


def format_search_results_context(results: list[SearchResult]) -> str:
    """Format search results as context for the LLM.

    Args:
        results: List of search results.

    Returns:
        Formatted context string with all artifacts.
    """
    if not results:
        return "No relevant artifacts found in the metadata."

    lines = ["## Retrieved Metadata Artifacts", ""]

    for i, result in enumerate(results, 1):
        lines.append(f"**Result {i}** (Similarity: {result.similarity:.3f})")
        lines.append(format_artifact_context(result.artifact))
        lines.append("")  # Blank line between artifacts

    return "\n".join(lines)


def build_qa_prompt(query: str, search_results: list[SearchResult]) -> str:
    """Build complete prompt for question answering.

    Args:
        query: User's question.
        search_results: Retrieved metadata artifacts.

    Returns:
        Complete prompt with context and query.
    """
    context = format_search_results_context(search_results)

    prompt = f"""{context}

## User Question

{query}

## Instructions

Answer the question using ONLY the metadata artifacts provided above.
- Cite your sources using the format: [ArtifactName] (ID: artifact_id)
- If the context doesn't contain enough information, respond with "I don't know based on the provided context"
- Be specific and include relevant technical details
- For database questions, mention schemas, types, constraints, and relationships
- For code questions, mention modules, types, and structure

Answer:"""

    return prompt


# Citation format and validation

CITATION_FORMAT = "[{artifact_name}] (ID: {artifact_id})"
CITATION_INSTRUCTION = """
CITATION REQUIREMENTS:
- Every fact must be cited with its source artifact
- Use this format: [ArtifactName] (ID: artifact_id)
- Example: The sale_order table [sale_order] (ID: postgresql_public_table_sale_order_a1b2c3) contains order information
- If multiple sources, cite each: [table1] (ID: id1), [table2] (ID: id2)
"""


def get_citation_examples() -> str:
    """Get example citations for the prompt.

    Returns:
        String with citation examples.
    """
    return """
Example Citations:
- "The sale_order table [sale_order] (ID: postgresql_public_table_sale_order_123abc) stores sales orders"
- "The id column [id] (ID: postgresql_public_column_sale_order_id_456def) is the primary key"
- "Orders are linked to partners via the partner_id foreign key [partner_id] (ID: postgresql_public_column_sale_order_partner_id_789ghi)"
"""


def format_dont_know_response() -> str:
    """Get the standard 'don't know' response template.

    Returns:
        Template for when context is insufficient.
    """
    return "I don't know based on the provided context. The retrieved metadata artifacts do not contain sufficient information to answer this question."
