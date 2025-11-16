# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field, model_validator
from typing import Any, Optional, Type, TypeVar, Dict, Literal, List, Set
from mcp_server_opensearch.global_state import get_mode
from pydantic_core import PydanticCustomError
from enum import Enum


T = TypeVar('T', bound=BaseModel)


def validate_args_for_mode(args_dict: Dict[str, Any], args_model_class: Type[T]) -> T:
    """
    Validation middleware that handles mode-specific validation.

    Args:
        args_dict: Dictionary of arguments provided by the user
        args_model_class: The Pydantic model class to validate against

    Returns:
        Validated instance of args_model_class
    """
    # Get the current mode from global state
    mode = get_mode()

    if mode == 'single':
        # In single mode, add default values for base fields
        args_dict = args_dict.copy()  # Don't modify the original
        args_dict.setdefault('opensearch_cluster_name', '')

    try:
        return args_model_class(**args_dict)
    except Exception as e:
        # Create a consistent error message format for both modes
        import re

        error_str = str(e)

        # Extract missing field names and create a clearer error message
        missing_fields = re.findall(r'^(\w+)\n  Field required', error_str, re.MULTILINE)
        if missing_fields:
            field_list = ', '.join(f"'{field}'" for field in missing_fields)
            if len(missing_fields) == 1:
                error_msg = f'Missing required field: {field_list}'
            else:
                error_msg = f'Missing required fields: {field_list}'

            # For single mode, show user input without opensearch_cluster_name
            if mode == 'single':
                user_input = {k: v for k, v in args_dict.items() if k != 'opensearch_cluster_name'}
                error_msg += f'\n\nProvided: {user_input}'
            else:
                error_msg += f'\n\nProvided: {args_dict}'

            raise ValueError(error_msg) from e

        raise e

class baseToolArgs(BaseModel):
    """Base class for all tool arguments that contains common OpenSearch connection parameters."""

    opensearch_cluster_name: str = Field(description='The name of the OpenSearch cluster')


class ListIndicesArgs(baseToolArgs):
    index: str = Field(
        default='',
        description='The name of the index to get detailed information for. If provided, returns detailed information about this specific index instead of listing all indices.',
    )
    include_detail: bool = Field(
        default=True,
        description='Whether to include detailed information. When listing indices (no index specified), if False, returns only a pure list of index names. If True, returns full metadata. When a specific index is provided, detailed information (including mappings) will be returned.',
    )


class GetIndexMappingArgs(baseToolArgs):
    index: str = Field(description='The name of the index to get mapping information for')


class SearchIndexArgs(baseToolArgs):
    index: str = Field(description='The name of the index to search in')
    query: Any = Field(description='The search query in OpenSearch query DSL format')


class GetShardsArgs(baseToolArgs):
    index: str = Field(description='The name of the index to get shard information for')


class GetClusterStateArgs(baseToolArgs):
    """Arguments for the GetClusterStateTool."""

    metric: Optional[str] = Field(
        default=None,
        description='Limit the information returned to the specified metrics. Options include: _all, blocks, metadata, nodes, routing_table, routing_nodes, master_node, version',
    )
    index: Optional[str] = Field(
        default=None, description='Limit the information returned to the specified indices'
    )

    class Config:
        json_schema_extra = {
            'examples': [{'metric': 'nodes', 'index': 'my_index'}, {'metric': '_all'}]
        }


class GetSegmentsArgs(baseToolArgs):
    """Arguments for the GetSegmentsTool."""

    index: Optional[str] = Field(
        default=None,
        description='Limit the information returned to the specified indices. If not provided, returns segments for all indices.',
    )

    class Config:
        json_schema_extra = {
            'examples': [
                {'index': 'my_index'},
                {},  # Empty example to show all segments
            ]
        }


class CatNodesArgs(baseToolArgs):
    """Arguments for the CatNodesTool."""

    metrics: Optional[str] = Field(
        default=None,
        description='A comma-separated list of metrics to display. Available metrics include: id, name, ip, port, role, master, heap.percent, ram.percent, cpu, load_1m, load_5m, load_15m, disk.total, disk.used, disk.avail, disk.used_percent',
    )

    class Config:
        json_schema_extra = {
            'examples': [
                {'metrics': 'name,ip,heap.percent,cpu,load_1m'},
                {},  # Empty example to show all node metrics
            ]
        }


class GetIndexInfoArgs(baseToolArgs):
    """Arguments for the GetIndexInfoTool."""

    index: str = Field(
        description='The name of the index to get detailed information for. Wildcards are supported.'
    )

    class Config:
        json_schema_extra = {
            'examples': [
                {'index': 'my_index'},
                {
                    'index': 'my_index*'  # Using wildcard
                },
            ]
        }


class GetIndexStatsArgs(baseToolArgs):
    """Arguments for the GetIndexStatsTool."""

    index: str = Field(
        description='The name of the index to get statistics for. Wildcards are supported.'
    )
    metric: Optional[str] = Field(
        default=None,
        description='Limit the information returned to the specified metrics. Options include: _all, completion, docs, fielddata, flush, get, indexing, merge, query_cache, refresh, request_cache, search, segments, store, warmer, bulk',
    )

    class Config:
        json_schema_extra = {
            'examples': [{'index': 'my_index'}, {'index': 'my_index', 'metric': 'search,indexing'}]
        }


class GetQueryInsightsArgs(baseToolArgs):
    """Arguments for the GetQueryInsightsTool."""

    # No additional parameters needed for the basic implementation
    # The tool will simply call GET /_insights/top_queries without parameters

    class Config:
        json_schema_extra = {
            'examples': [
                {}  # Empty example as no additional parameters are required
            ]
        }


class GetNodesHotThreadsArgs(baseToolArgs):
    """Arguments for the GetNodesHotThreadsTool."""

    # No additional parameters needed for the basic implementation
    # The tool will simply call GET /_nodes/hot_threads without parameters

    class Config:
        json_schema_extra = {
            'examples': [
                {}  # Empty example as no additional parameters are required
            ]
        }


class GetAllocationArgs(baseToolArgs):
    """Arguments for the GetAllocationTool."""

    # No additional parameters needed for the basic implementation
    # The tool will simply call GET /_cat/allocation without parameters

    class Config:
        json_schema_extra = {
            'examples': [
                {}  # Empty example as no additional parameters are required
            ]
        }


class GetLongRunningTasksArgs(baseToolArgs):
    """Arguments for the GetLongRunningTasksTool."""

    limit: Optional[int] = Field(
        default=10, description='The maximum number of tasks to return. Default is 10.'
    )

    class Config:
        json_schema_extra = {
            'examples': [
                {},  # Default example to show top 10 long-running tasks
                {
                    'limit': 5  # Example to show top 5 long-running tasks
                },
            ]
        }


class GetNodesArgs(baseToolArgs):
    """Arguments for the GetNodesTool."""

    node_id: Optional[str] = Field(
        default=None,
        description='A comma-separated list of node IDs or names to limit the returned information. Supports node filters like _local, _master, master:true, data:false, etc. Defaults to _all.',
    )
    metric: Optional[str] = Field(
        default=None,
        description='A comma-separated list of metric groups to include in the response. Options include: settings, os, process, jvm, thread_pool, transport, http, plugins, ingest, aggregations, indices. Defaults to all metrics.',
    )

    class Config:
        json_schema_extra = {
            'examples': [
                {},  # Get all nodes with all metrics
                {'node_id': 'master:true', 'metric': 'process,transport'},
                {'node_id': '_local', 'metric': 'jvm,os'},
            ]
        }


# --- Agentic Memory ---
class EmbeddingModelType(str, Enum):
    text_embedding = "TEXT_EMBEDDING"
    sparse_encoding = "SPARSE_ENCODING"


class StrategyType(str, Enum):
    semantic = "SEMANTIC"
    user_preference = "USER_PREFERENCE"
    summary = "SUMMARY"


class MemoryType(str, Enum):
    sessions = "sessions"
    working = "working"
    long_term = "long-term"
    history = "history"


class PayloadType(str, Enum):
    conversational = "conversational"
    data = "data"


ERR_FIELD_NOT_ALLOWED = "field_not_allowed"
ERR_MISSING_WORKING_FIELD = "missing_working_field"
ERR_MISSING_LONG_TERM_FIELD = "missing_long_term_field"
ERR_MESSAGES_REQUIRED = "messages_required"
ERR_FIELD_PROHIBITED = "field_prohibited"
ERR_STRUCTURED_DATA_REQUIRED = "structured_data_required"
ERR_MISSING_CONTENT_FIELD = "missing_content_field"
ERR_EMBEDDING_DIMENSION_REQUIRED = "embedding_dimension_required"


class MessageContentItem(BaseModel):
    """
    Schema for the content part of a message.
    Used for strong typing in 'messages' fields.
    """

    text: str = Field(..., description="The text content of the message.")
    content_type: str = Field(
        ..., description="The type of the content (e.g., 'text'). ", alias="type"
    )


class MessageItem(BaseModel):
    """
    Schema for a single message in 'messages' field.
    Used for strong typing.
    """

    role: Optional[str] = Field(
        None, description="The role of the entity (e.g., 'user', 'assistant')."
    )
    content: List[MessageContentItem] = Field(
        ..., description="A list of content items for this message."
    )


class BaseAgenticMemoryContainerArgs(baseToolArgs):
    """
    Base arguments for tools operating on an existing Agentic Memory Container.
    """

    memory_container_id: str = Field(..., description="The ID of the memory container.")


class UpdateAgenticMemoryArgs(BaseAgenticMemoryContainerArgs):
    """Arguments for updating a specific agentic memory by its type and ID."""

    # --- Constants for Validation ---
    _SESSION_ONLY_FIELDS: Set[str] = {"summary", "agents", "additional_info"}
    _WORKING_ONLY_FIELDS: Set[str] = {"messages", "structured_data", "binary_data"}
    _LONG_TERM_ONLY_FIELDS: Set[str] = {"memory"}
    _UPDATABLE_WORKING_FIELDS: Set[str] = {
        "messages",
        "structured_data",
        "binary_data",
        "tags",
        "metadata",
    }
    _UPDATABLE_LONG_TERM_FIELDS: Set[str] = {"memory", "tags", "metadata"}

    # --- Required Path Fields ---
    memory_type: Literal[
        MemoryType.sessions, MemoryType.working, MemoryType.long_term
    ] = Field(
        ...,
        alias="type",
        description="The memory type. Valid values are sessions, working, and long-term. Note that history memory cannot be updated.",
    )
    id: str = Field(..., description="The ID of the memory to update.")

    # --- Session memory fields ---
    summary: Optional[str] = Field(
        default=None, description="The summary of the session."
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata for the memory (for example, status, branch, or custom fields).",
    )
    agents: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional information about the agents."
    )
    additional_info: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata to associate with the session."
    )

    # --- Working memory fields ---
    messages: Optional[List[MessageItem]] = Field(
        default=None,
        description="Updated conversation messages (for conversation type).",
    )
    structured_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Updated structured data content (for data memory payloads).",
    )
    binary_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Updated binary data content (for data memory payloads).",
    )
    tags: Optional[Dict[str, Any]] = Field(
        default=None, description="Updated tags for categorization."
    )

    # --- Long-term memory fields ---
    memory: Optional[str] = Field(
        default=None, description="The updated memory content."
    )

    @model_validator(mode="after")
    def validate_memory_type_fields(self) -> "UpdateAgenticMemoryArgs":
        set_fields = self.model_fields_set

        def _raise_not_allowed_error(field_name: str, memory_type: str):
            raise PydanticCustomError(
                ERR_FIELD_NOT_ALLOWED,
                "Field '{field_name}' should not be provided when updating {memory_type} memory",
                {"field_name": field_name, "memory_type": memory_type},
            )

        if self.memory_type == MemoryType.sessions:
            disallowed_fields = self._WORKING_ONLY_FIELDS | self._LONG_TERM_ONLY_FIELDS
            for field in disallowed_fields:
                if field in set_fields:
                    _raise_not_allowed_error(field, "session")

        elif self.memory_type == MemoryType.working:
            disallowed_fields = self._SESSION_ONLY_FIELDS | self._LONG_TERM_ONLY_FIELDS
            for field in disallowed_fields:
                if field in set_fields:
                    _raise_not_allowed_error(field, "working")

            if not any(field in set_fields for field in self._UPDATABLE_WORKING_FIELDS):
                raise PydanticCustomError(
                    ERR_MISSING_WORKING_FIELD,
                    "At least one field ({fields}) must be provided for updating working memory",
                    {"fields": ", ".join(self._UPDATABLE_WORKING_FIELDS)},
                )

        elif self.memory_type == MemoryType.long_term:
            disallowed_fields = self._SESSION_ONLY_FIELDS | self._WORKING_ONLY_FIELDS
            for field in disallowed_fields:
                if field in set_fields:
                    _raise_not_allowed_error(field, "long-term")

            if not any(
                field in set_fields for field in self._UPDATABLE_LONG_TERM_FIELDS
            ):
                raise PydanticCustomError(
                    ERR_MISSING_LONG_TERM_FIELD,
                    "At least one field ({fields}) must be provided for updating long-term memory",
                    {"fields": ", ".join(self._UPDATABLE_LONG_TERM_FIELDS)},
                )

        return self

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "memory_container_id": "HudqiJkB1SltqOcZusVU",
                    "type": "sessions",
                    "id": "N2CDipkB2Mtr6INFFcX8",
                    "additional_info": {
                        # Flexible object for storing any session-specific metadata
                        "key1": "value1",
                        # Timestamp of the last activity in the session (ISO 8601 format)
                        "last_activity": "2025-09-15T17:30:00Z",
                    },
                },
                {
                    "memory_container_id": "HudqiJkB1SltqOcZusVU",
                    "type": "working",
                    "id": "XyEuiJkBeh2gPPwzjYWM",
                    # Key-value pairs for categorizing and filtering working memories
                    "tags": {"topic": "updated_topic", "priority": "high"},
                },
                {
                    "memory_container_id": "HudqiJkB1SltqOcZusVU",
                    "type": "long-term",
                    "id": "DcxjTpkBvwXRq366C1Zz",
                    # Actual memory content for long-term storage
                    "memory": "User's name is Bob Smith",
                    # Tags help in organizing and retrieving long-term memories
                    "tags": {"topic": "personal info", "updated": "true"},
                },
                {
                    "memory_container_id": "HudqiJkB1SltqOcZusVU",
                    "type": "working",
                    "id": "another_working_memory_id",
                    # Array of conversation messages (typically used for conversational memory)
                    "messages": [
                        {
                            # Role of the message sender (e.g., 'user', 'assistant')
                            "role": "user",
                            "content": [
                                # Content supports multiple types and structures
                                {"text": "Updated user message", "type": "text"}
                            ],
                        }
                    ],
                    # Custom key-value pairs for storing operational state or other context
                    "metadata": {"status": "updated"},
                },
            ]
        }


class IndexSettingsArgs(BaseModel):
    """Index settings for agentic memory storage indexes."""

    session_index: Optional[Dict[str, Any]] = Field(
        default=None, description="Custom index settings for session memory index."
    )
    short_term_memory_index: Optional[Dict[str, Any]] = Field(
        default=None, description="Custom index settings for short-term memory index."
    )
    long_term_memory_index: Optional[Dict[str, Any]] = Field(
        default=None, description="Custom index settings for long-term memory index."
    )
    long_term_memory_history_index: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Custom index settings for long-term memory history index.",
    )


class StrategyConfigurationArgs(BaseModel):
    """Strategy-specific agentic memory configuration."""

    llm_result_path: Optional[str] = Field(
        default=None,
        description='A JSONPath expression for extracting LLM results from responses. Default is the Amazon Bedrock Converse API response path ("$.output.message.content[0].text").',
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="A custom system prompt used to override the default strategy prompt.",
    )
    llm_id: Optional[str] = Field(
        default=None,
        description="The LLM model ID for this strategy. Overrides the global LLM setting.",
    )


class StrategyArgs(BaseModel):
    """Agentic memory processing strategy."""

    strategy_type: StrategyType = Field(
        ...,
        description="The strategy type. Valid values are SEMANTIC, USER_PREFERENCE, and SUMMARY.",
    )
    namespace: List[str] = Field(
        ...,
        description='An array of namespace dimensions for organizing memories (for example, ["user_id"] or ["agent_id", "session_id"]).',
    )
    configuration: Optional[StrategyConfigurationArgs] = Field(
        default=None, description="Strategy-specific configuration."
    )
    enabled: Optional[bool] = Field(
        default=True,
        description="Whether to enable the strategy in the memory container. Default is true.",
    )


class ParametersArgs(BaseModel):
    """Global parameters for the agentic memory container."""

    llm_result_path: Optional[str] = Field(
        default=None,
        description='A global JSONPath expression for extracting LLM results from responses. Default is the Amazon Bedrock Converse API response path ("$.output.message.content[0].text").',
    )


class AgenticMemoryConfigurationArgs(BaseModel):
    """Agentic memory container configuration."""

    embedding_model_type: Optional[EmbeddingModelType] = Field(
        default=None,
        description="The embedding model type. Supported types are TEXT_EMBEDDING and SPARSE_ENCODING.",
    )
    embedding_model_id: Optional[str] = Field(
        default=None, description="The embedding model ID."
    )
    embedding_dimension: Optional[int] = Field(
        default=None,
        description="The dimension of the embedding model. Required if embedding_model_type is TEXT_EMBEDDING.",
    )
    llm_id: Optional[str] = Field(
        default=None, description="The LLM model ID for processing and inference."
    )
    index_prefix: Optional[str] = Field(
        default=None,
        description="A custom prefix for memory indexes. If not specified, a default prefix is used: default when use_system_index is true, or an 8-character random UUID when use_system_index is false.",
    )
    use_system_index: Optional[bool] = Field(
        default=True, description="Whether to use system indexes. Default is true."
    )
    disable_history: Optional[bool] = Field(
        default=False,
        description="If disabled, no history will be persisted. Default is false, so history will be persisted by default.",
    )
    disable_session: Optional[bool] = Field(
        default=True,
        description="If disabled, no session will be persisted. Default is true, so the session will not be persisted by default.",
    )
    max_infer_size: Optional[int] = Field(
        default=None,
        description="Controls the top k number of similar existing memories retrieved during memory consolidation to make ADD/UPDATE/DELETE decisions.",
    )
    index_settings: Optional[IndexSettingsArgs] = Field(
        default=None,
        description="Custom OpenSearch index settings for the memory storage indexes that will be created for this container. Each memory type (sessions, working, long_term, and history) uses its own index.",
    )
    strategies: Optional[List[StrategyArgs]] = Field(
        default=None, description="An array of memory processing strategies."
    )
    parameters: Optional[ParametersArgs] = Field(
        default=None, description="Global parameters for the memory container."
    )

    @model_validator(mode="after")
    def validate_embedding_configuration(self) -> "AgenticMemoryConfigurationArgs":
        """Validate embedding model configuration."""
        if (
            self.embedding_model_type == "TEXT_EMBEDDING"
            and self.embedding_dimension is None
        ):
            raise PydanticCustomError(
                ERR_EMBEDDING_DIMENSION_REQUIRED,
                "embedding_dimension is required when embedding_model_type is TEXT_EMBEDDING",
            )
        return self


class CreateAgenticMemoryContainerArgs(baseToolArgs):
    """Arguments for creating a agentic memory container to store agentic memories."""

    name: str = Field(..., description="The name of the memory container.")
    description: Optional[str] = Field(
        default=None, description="The description of the memory container."
    )
    configuration: AgenticMemoryConfigurationArgs = Field(
        ..., description="The memory container configuration."
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "name": "agentic memory test",
                    "description": "Store conversations with semantic search and summarization",
                    "configuration": {
                        # Model type for vector embeddings
                        "embedding_model_type": "TEXT_EMBEDDING",
                        # ID of a model uploaded to ML Commons
                        "embedding_model_id": "your-embedding-model-id",
                        # Dimension of the embedding vectors the model produces
                        "embedding_dimension": 1024,
                        # ID of the LLM used for summarization and information extraction
                        "llm_id": "your-llm-model-id",
                        # Defines how memories are organized and retrieved
                        "strategies": [{"type": "SEMANTIC", "namespace": ["user_id"]}],
                    },
                },
                {
                    "name": "advanced memory container",
                    "description": "Store conversations with semantic search and summarization",
                    "configuration": {
                        "embedding_model_type": "TEXT_EMBEDDING",
                        "embedding_model_id": "your-embedding-model-id",
                        "embedding_dimension": 1024,
                        "llm_id": "your-llm-model-id",
                        # Custom prefix for all underlying OpenSearch indices
                        "index_prefix": "my_custom_prefix",
                        # If False, creates indices in standard OpenSearch
                        "use_system_index": False,
                        "strategies": [
                            {
                                "type": "SEMANTIC",
                                # Isolates memories by agent_id for multi-tenant scenarios
                                "namespace": ["agent_id"],
                                "configuration": {
                                    # JSONPath to extract specific text from LLM response
                                    "llm_result_path": "$.output.message.content[0].text",
                                    # Custom prompt for this strategy
                                    "system_prompt": "Extract semantic information from user conversations",
                                    # Optional: Override the default LLM for this strategy
                                    "llm_id": "strategy-llm-id",
                                },
                            },
                            {
                                # Infers and stores user preferences
                                "type": "USER_PREFERENCE",
                                "namespace": ["agent_id"],
                                "configuration": {
                                    "llm_result_path": "$.output.message.content[0].text"
                                },
                            },
                            {
                                # Creates and updates conversation summaries
                                "type": "SUMMARY",
                                "namespace": ["agent_id"],
                                "configuration": {
                                    "llm_result_path": "$.output.message.content[0].text"
                                },
                            },
                        ],
                        "parameters": {
                            # Default JSONPath applied across strategies
                            "llm_result_path": "$.output.message.content[0].text"
                        },
                        # Custom settings for the underlying OpenSearch indices
                        "index_settings": {
                            "session_index": {
                                "index": {
                                    # Fixed at index creation
                                    "number_of_shards": "2",
                                    # Can be changed dynamically
                                    "number_of_replicas": "2",
                                }
                            },
                            "short_term_memory_index": {
                                "index": {
                                    "number_of_shards": "2",
                                    "number_of_replicas": "2",
                                }
                            },
                            "long_term_memory_index": {
                                "index": {
                                    "number_of_shards": "2",
                                    "number_of_replicas": "2",
                                }
                            },
                            # Tracks evolution of long-term memories
                            "long_term_memory_history_index": {
                                "index": {
                                    "number_of_shards": "2",
                                    "number_of_replicas": "2",
                                }
                            },
                        },
                    },
                },
            ]
        }


class CreateAgenticMemorySessionArgs(BaseAgenticMemoryContainerArgs):
    """Arguments for creating a new session in a agentic memory container."""

    session_id: Optional[str] = Field(
        default=None,
        description="A custom session ID. If provided, this ID is used for the session. If not provided, a random ID is generated.",
    )
    summary: Optional[str] = Field(
        default=None, description="A session summary or description."
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata for the session provided as key-value pairs.",
    )
    namespace: Optional[Dict[str, str]] = Field(
        default=None, description="Namespace information for organizing the session."
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "memory_container_id": "SdjmmpgBOh0h20Y9kWuN",
                    # Optional: Client-provided ID. If omitted, OpenSearch auto-generates one, must be unique within the memory container
                    "session_id": "abc123",
                    # Optional key-value pairs for session context
                    "metadata": {"key1": "value1"},
                },
                {
                    "memory_container_id": "SdjmmpgBOh0h20Y9kWuN",
                    # Human-readable description of the session
                    "summary": "This is a test session",
                    "metadata": {"key1": "value1"},
                    # Isolates session to specific user - matches strategy namespace from container
                    "namespace": {"user_id": "bob"},
                },
                {
                    "memory_container_id": "SdjmmpgBOh0h20Y9kWuN",
                    "summary": "Session for user onboarding",
                    # Multi-dimensional namespacing supported
                    "namespace": {
                        "user_id": "alice",
                        "agent_id": "onboarding_bot",
                    },
                    # Used for filtering and organization
                    "metadata": {
                        "priority": "high",
                        "category": "onboarding",
                    },
                },
            ]
        }


class GetAgenticMemoryArgs(BaseAgenticMemoryContainerArgs):
    """Arguments for retrieving a specific agentic memory by its type and ID."""

    memory_type: MemoryType = Field(
        ...,
        alias="type",
        description="The memory type. Valid values are sessions, working, long-term, and history.",
    )
    id: str = Field(..., description="The ID of the memory to retrieve.")

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "memory_container_id": "HudqiJkB1SltqOcZusVU",
                    # Active conversation data, agent state, and temporary context used during ongoing interactions
                    "type": "working",
                    "id": "XyEuiJkBeh2gPPwzjYWM",
                },
                {
                    "memory_container_id": "HudqiJkB1SltqOcZusVU",
                    # Processed knowledge and facts extracted from conversations over time via LLM inference
                    "type": "long-term",
                    "id": "DcxjTpkBvwXRq366C1Zz",
                },
                {
                    "memory_container_id": "HudqiJkB1SltqOcZusVU",
                    # Manages conversation sessions and their metadata (start time, participants, state)
                    "type": "sessions",
                    "id": "CcxjTpkBvwXRq366A1aE",
                },
                {
                    "memory_container_id": "HudqiJkB1SltqOcZusVU",
                    # Audit trail of all memory operations (add/update/delete) across the container
                    "type": "history",
                    # Specific history record ID tracking memory evolution
                    "id": "eMxnTpkBvwXRq366hmAU",
                },
            ]
        }


class AddAgenticMemoriesArgs(BaseAgenticMemoryContainerArgs):
    """Arguments for adding memories to the agentic memory container."""

    # --- Payload Fields ---
    messages: Optional[List[MessageItem]] = Field(
        default=None, description="A list of messages for a conversational payload..."
    )
    structured_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured data content for data memory. Required when payload_type is data.",
    )
    binary_data: Optional[str] = Field(
        default=None,
        description="Binary data content encoded as a Base64 string for binary payloads.",
    )
    payload_type: PayloadType = Field(
        ..., description="The type of payload. Valid values are conversational or data."
    )

    # --- Optional Fields ---
    namespace: Optional[Dict[str, str]] = Field(
        default=None, description="The namespace context for organizing memories..."
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata for the memory..."
    )
    tags: Optional[Dict[str, Any]] = Field(
        default=None, description="Tags for categorizing and organizing memories."
    )
    infer: Optional[bool] = Field(
        default=False,
        description="Whether to use a large language model (LLM) to extract key information...",
    )

    @model_validator(mode="after")
    def validate_payload_requirements(self) -> "AddAgenticMemoriesArgs":
        """Validate that the correct fields are provided based on payload_type."""

        # Getting fields that were actually set
        set_fields = self.model_fields_set

        if self.payload_type == PayloadType.conversational:
            if "messages" not in set_fields:
                raise PydanticCustomError(
                    ERR_MESSAGES_REQUIRED,
                    "'messages' field is required when payload_type is 'conversational'",
                )
            if "structured_data" in set_fields:
                raise PydanticCustomError(
                    ERR_FIELD_PROHIBITED,
                    "'structured_data' should not be provided when payload_type is 'conversational'",
                    {"field_name": "structured_data"},
                )

        elif self.payload_type == PayloadType.data:
            if "structured_data" not in set_fields:
                raise PydanticCustomError(
                    ERR_STRUCTURED_DATA_REQUIRED,
                    "'structured_data' field is required when payload_type is 'data'",
                )
            if "messages" in set_fields:
                raise PydanticCustomError(
                    ERR_FIELD_PROHIBITED,
                    "'messages' should not be provided when payload_type is 'data'",
                    {"field_name": "messages"},
                )

        # Validate that at least one content field is provided
        content_fields = {"messages", "structured_data", "binary_data"}
        if not any(field in set_fields for field in content_fields):
            raise PydanticCustomError(
                ERR_MISSING_CONTENT_FIELD,
                "At least one content field (messages, structured_data, or binary_data) must be provided",
            )

        return self

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "memory_container_id": "SdjmmpgBOh0h20Y9kWuN",
                    # Conversational exchange between user and assistant
                    "messages": [
                        {
                            # Standard chat roles: 'user', 'assistant'
                            "role": "user",
                            "content": [
                                {
                                    "text": "I'm Bob, I really like swimming.",
                                    "type": "text",
                                }
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "text": "Cool, nice. Hope you enjoy your life.",
                                    "type": "text",
                                }
                            ],
                        },
                    ],
                    # Must match namespace from container strategies
                    "namespace": {"user_id": "bob"},
                    "metadata": {
                        # Custom workflow state tracking
                        "status": "checkpoint",
                        # Supports branching conversations for exploration
                        "branch": {
                            # Branch identifier
                            "branch_name": "high",
                            # Parent conversation point
                            "root_event_id": "228nadfs879mtgk",
                        },
                    },
                    # Enables filtering and categorization
                    "tags": {"topic": "personal info"},
                    # Enables AI processing (summarization, semantic extraction, etc.)
                    "infer": True,
                    # Determines how AI strategies are applied
                    "payload_type": "conversational",
                },
                {
                    "memory_container_id": "SdjmmpgBOh0h20Y9kWuN",
                    # Alternative to messages - for non-conversational data
                    "structured_data": {
                        "time_range": {"start": "2025-09-11", "end": "2025-09-15"}
                    },
                    "namespace": {"agent_id": "testAgent1"},
                    # Flexible schema
                    "metadata": {"status": "checkpoint", "anyobject": "abc"},
                    "tags": {"topic": "agent_state"},
                    # Skips AI processing - stores raw data only
                    "infer": False,
                    # Bypasses conversational AI pipelines
                    "payload_type": "data",
                },
            ]
        }


class SearchAgenticMemoryArgs(BaseAgenticMemoryContainerArgs):
    """Arguments for searching memories of a specific type within a agentic memory container."""

    memory_type: MemoryType = Field(
        ...,
        alias="type",
        description="The memory type. Valid values are sessions, working, long-term, and history.",
    )
    query: Dict[str, Any] = Field(
        ..., description="The search query using OpenSearch query DSL."
    )
    sort: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Sort specification for the search results."
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    # The unique identifier for the memory container to search within
                    "memory_container_id": "HudqiJkB1SltqOcZusVU",
                    # Specifies the type of memory to search (e.g., sessions, long-term, working, history)
                    "type": "sessions",
                    # OpenSearch Query DSL: matches all documents in the specified memory type
                    "query": {"match_all": {}},
                    # Sorts results by creation time, newest first
                    "sort": [{"created_time": {"order": "desc"}}],
                },
                {
                    "memory_container_id": "HudqiJkB1SltqOcZusVU",
                    "type": "long-term",
                    "query": {
                        # Term query finds exact matches in the 'namespace.user_id' field for user isolation
                        "bool": {"must": [{"term": {"namespace.user_id": "bob"}}]}
                    },
                    "sort": [{"created_time": {"order": "desc"}}],
                },
                {
                    "memory_container_id": "HudqiJkB1SltqOcZusVU",
                    # 'history' type stores past interactions; typically searched with match_all to review chronologically
                    "type": "history",
                    "query": {"match_all": {}},
                    "sort": [{"created_time": {"order": "desc"}}],
                },
                {
                    "memory_container_id": "HudqiJkB1SltqOcZusVU",
                    "type": "working",
                    "query": {
                        "bool": {
                            # Finds memories for a specific user
                            "must": [{"term": {"namespace.user_id": "bob"}}],
                            "must_not": [
                                # Excludes memories that have a 'parent_memory_id' tag
                                {"exists": {"field": "tags.parent_memory_id"}}
                            ],
                        }
                    },
                    "sort": [{"created_time": {"order": "desc"}}],
                },
                {
                    "memory_container_id": "HudqiJkB1SltqOcZusVU",
                    "type": "working",
                    # Finds memories associated with a specific session
                    "query": {"term": {"namespace.session_id": "123"}},
                    "sort": [{"created_time": {"order": "desc"}}],
                },
            ]
        }


class DeleteAgenticMemoryByIDArgs(BaseAgenticMemoryContainerArgs):
    """Arguments for deleting a specific agentic memory by its type and ID."""

    memory_type: MemoryType = Field(
        ...,
        alias="type",
        description="The type of memory to delete. Valid values are sessions, working, long-term, and history.",
    )
    id: str = Field(..., description="The ID of the specific memory to delete.")

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    # The unique identifier for the memory container from which the memory will be deleted
                    "memory_container_id": "HudqiJkB1SltqOcZusVU",
                    # Specifies the type of memory to delete. Valid values are 'sessions', 'working', 'long-term', and 'history'
                    "type": "working",
                    # The unique identifier of the specific 'working' memory to be deleted
                    "id": "XyEuiJkBeh2gPPwzjYWM",
                },
                {
                    "memory_container_id": "HudqiJkB1SltqOcZusVU",
                    # Use to delete a long-term memory, which typically stores factual information
                    "type": "long-term",
                    "id": "DcxjTpkBvwXRq366C1Zz",
                },
                {
                    "memory_container_id": "HudqiJkB1SltqOcZusVU",
                    # Use to delete a session memory, which tracks conversation sessions
                    "type": "sessions",
                    "id": "CcxjTpkBvwXRq366A1aE",
                },
                {
                    "memory_container_id": "HudqiJkB1SltqOcZusVU",
                    # Use to delete a history memory, which maintains an audit trail of memory operations
                    "type": "history",
                    "id": "eMxnTpkBvwXRq366hmAU",
                },
            ]
        }


class DeleteAgenticMemoryByQueryArgs(BaseAgenticMemoryContainerArgs):
    """Arguments for deleting agentic memories by query."""

    memory_type: MemoryType = Field(
        ...,
        alias="type",
        description="The type of memory to delete. Valid values are sessions, working, long-term, and history.",
    )
    query: Dict[str, Any] = Field(
        ...,
        description="The query to match the memories you want to delete. This should be a valid OpenSearch query DSL object.",
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    # The unique identifier for the memory container from which memories will be deleted
                    "memory_container_id": "HudqiJkB1SltqOcZusVU",
                    # The type of memory to delete. Valid values are 'sessions', 'working', 'long-term', and 'history'
                    "type": "working",
                    # Uses OpenSearch Query DSL to match all 'working' memories where the 'owner_id' field is "admin"
                    "query": {"match": {"owner_id": "admin"}},
                },
                {
                    "memory_container_id": "HudqiJkB1SltqOcZusVU",
                    "type": "long-term",
                    # Deletes 'long-term' memories created before 2025-09-01; useful for data retention policies
                    "query": {"range": {"created_time": {"lt": "2025-09-01"}}},
                },
                {
                    "memory_container_id": "HudqiJkB1SltqOcZusVU",
                    "type": "sessions",
                    # Deletes 'sessions' memories for a specific user; 'term' query finds exact matches in the 'namespace.user_id' field
                    "query": {"term": {"namespace.user_id": "inactive_user"}},
                },
            ]
        }


# ! --- Agentic Memory ---
