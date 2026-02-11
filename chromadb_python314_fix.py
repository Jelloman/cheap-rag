"""Workaround for ChromaDB Python 3.14 compatibility issue.

This patches the ChromaDB Settings class to work with Python 3.14.
"""

import sys
from typing import Optional

# Patch before importing chromadb
def patch_chromadb_config():
    """Patch ChromaDB config to work with Python 3.14."""
    import chromadb.config as config_module

    # Create a simple Settings class that bypasses Pydantic v1
    class SettingsPatch:
        def __init__(self, **kwargs):
            # Set default values
            self.chroma_api_impl = kwargs.get("chroma_api_impl", "chromadb.api.segment.SegmentAPI")
            self.chroma_telemetry_impl = kwargs.get("chroma_telemetry_impl", "chromadb.telemetry.posthog.Posthog")
            self.chroma_db_impl = kwargs.get("chroma_db_impl", None)
            self.chroma_product_telemetry_impl = kwargs.get("chroma_product_telemetry_impl", "chromadb.telemetry.product.posthog.ProductPosthogTelemetry")
            self.chroma_segment_manager_impl = kwargs.get("chroma_segment_manager_impl", "chromadb.segment.impl.manager.local.LocalSegmentManager")

            self.anonymized_telemetry = kwargs.get("anonymized_telemetry", True)
            self.allow_reset = kwargs.get("allow_reset", False)

            self.is_persistent = kwargs.get("is_persistent", False)
            self.persist_directory = kwargs.get("persist_directory", "./chroma")

            self.chroma_server_host = kwargs.get("chroma_server_host", None)
            self.chroma_server_http_port = kwargs.get("chroma_server_http_port", None)
            self.chroma_server_ssl_enabled = kwargs.get("chroma_server_ssl_enabled", False)
            self.chroma_server_grpc_port = kwargs.get("chroma_server_grpc_port", None)
            self.chroma_server_cors_allow_origins = kwargs.get("chroma_server_cors_allow_origins", [])

            self.chroma_client_auth_provider = kwargs.get("chroma_client_auth_provider", None)
            self.chroma_client_auth_credentials = kwargs.get("chroma_client_auth_credentials", None)
            self.chroma_server_auth_provider = kwargs.get("chroma_server_auth_provider", None)
            self.chroma_server_auth_credentials_provider = kwargs.get("chroma_server_auth_credentials_provider", None)
            self.chroma_server_auth_credentials = kwargs.get("chroma_server_auth_credentials", None)
            self.chroma_server_auth_token_transport_header = kwargs.get("chroma_server_auth_token_transport_header", None)

            # Set any additional kwargs as attributes
            for key, value in kwargs.items():
                if not hasattr(self, key):
                    setattr(self, key, value)

    # Replace the Settings class
    config_module.Settings = SettingsPatch

    return config_module

# Apply patch before any chromadb imports
if sys.version_info >= (3, 14):
    try:
        patch_chromadb_config()
        print("[INFO] Applied Python 3.14 compatibility patch for ChromaDB")
    except Exception as e:
        print(f"[WARNING] Could not apply ChromaDB patch: {e}")
