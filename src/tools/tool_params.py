# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field, model_validator
from typing import Any, Optional, Type, TypeVar, Dict, Literal
from mcp_server_opensearch.global_state import get_mode
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


# TODO move enums somewhere
class EmbeddingModelType(str, Enum):
    text_embedding = 'TEXT_EMBEDDING'
    sparse_encoding = 'SPARSE_ENCODING'


class StrategyType(str, Enum):
    semantic = 'SEMANTIC'
    user_preference = 'USER_PREFERENCE'
    summary = 'SUMMARY'


class MemoryType(str, Enum):
    sessions = 'sessions'
    working = 'working'
    long_term = 'long-term'
    history = 'history'


class PayloadType(str, Enum):
    conversational = 'conversational'
    data = 'data'


class IndexSettingsArgs(baseToolArgs):
    """Index settings for memory storage indexes."""
    
    session_index: Optional[Dict[str, Any]] = Field(
        default=None,
        description='Custom index settings for session memory index.'
    )
    short_term_memory_index: Optional[Dict[str, Any]] = Field(
        default=None,
        description='Custom index settings for short-term memory index.'
    )
    long_term_memory_index: Optional[Dict[str, Any]] = Field(
        default=None,
        description='Custom index settings for long-term memory index.'
    )
    long_term_memory_history_index: Optional[Dict[str, Any]] = Field(
        default=None,
        description='Custom index settings for long-term memory history index.'
    )


class StrategyConfigurationArgs(baseToolArgs):
    """Strategy-specific configuration."""
    
    llm_result_path: Optional[str] = Field(
        default=None,
        description='A JSONPath expression for extracting LLM results from responses. Default is the Amazon Bedrock Converse API response path ("$.output.message.content[0].text").'
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description='A custom system prompt used to override the default strategy prompt.'
    )
    llm_id: Optional[str] = Field(
        default=None,
        description='The LLM model ID for this strategy. Overrides the global LLM setting.'
    )


class StrategyArgs(baseToolArgs):
    """Memory processing strategy."""
    
    strategy_type: StrategyType = Field(
        ...,
        description='The strategy type. Valid values are SEMANTIC, USER_PREFERENCE, and SUMMARY.'
    )
    namespace: List[str] = Field(
        ...,
        description='An array of namespace dimensions for organizing memories (for example, ["user_id"] or ["agent_id", "session_id"]).'
    )
    configuration: Optional[StrategyConfigurationArgs] = Field(
        default=None,
        description='Strategy-specific configuration.'
    )
    enabled: Optional[bool] = Field(
        default=True,
        description='Whether to enable the strategy in the memory container. Default is true.'
    )


class ParametersArgs(baseToolArgs):
    """Global parameters for the memory container."""
    
    llm_result_path: Optional[str] = Field(
        default=None,
        description='A global JSONPath expression for extracting LLM results from responses. Default is the Amazon Bedrock Converse API response path ("$.output.message.content[0].text").'
    )


class ConfigurationArgs(baseToolArgs):
    """Memory container configuration."""
    
    embedding_model_type: Optional[EmbeddingModelType] = Field(
        default=None,
        description='The embedding model type. Supported types are TEXT_EMBEDDING and SPARSE_ENCODING.'
    )
    embedding_model_id: Optional[str] = Field(
        default=None,
        description='The embedding model ID.'
    )
    embedding_dimension: Optional[int] = Field(
        default=None,
        description='The dimension of the embedding model. Required if embedding_model_type is TEXT_EMBEDDING.'
    )
    llm_id: Optional[str] = Field(
        default=None,
        description='The LLM model ID for processing and inference.'
    )
    index_prefix: Optional[str] = Field(
        default=None,
        description='A custom prefix for memory indexes. If not specified, a default prefix is used: default when use_system_index is true, or an 8-character random UUID when use_system_index is false.'
    )
    use_system_index: Optional[bool] = Field(
        default=True,
        description='Whether to use system indexes. Default is true.'
    )
    disable_history: Optional[bool] = Field(
        default=False,
        description='If disabled, no history will be persisted. Default is false, so history will be persisted by default.'
    )
    disable_session: Optional[bool] = Field(
        default=True,
        description='If disabled, no session will be persisted. Default is true, so the session will not be persisted by default.'
    )
    max_infer_size: Optional[int] = Field(
        default=None,
        description='Controls the top k number of similar existing memories retrieved during memory consolidation to make ADD/UPDATE/DELETE decisions.'
    )
    index_settings: Optional[IndexSettingsArgs] = Field(
        default=None,
        description='Custom OpenSearch index settings for the memory storage indexes that will be created for this container. Each memory type (sessions, working, long_term, and history) uses its own index.'
    )
    strategies: Optional[List[StrategyArgs]] = Field(
        default=None,
        description='An array of memory processing strategies.'
    )
    parameters: Optional[ParametersArgs] = Field(
        default=None,
        description='Global parameters for the memory container.'
    )
    
    @model_validator(mode='after')
    def validate_embedding_configuration(self) -> 'ConfigurationArgs':
        """Validate embedding model configuration."""
        if self.embedding_model_type == 'TEXT_EMBEDDING' and self.embedding_dimension is None:
            raise ValueError('embedding_dimension is required when embedding_model_type is TEXT_EMBEDDING')
        return self


class CreateMemoryContainerArgs(baseToolArgs):
    """Arguments for creating a memory container to store agentic memories."""
    
    name: str = Field(
        ...,
        description='The name of the memory container.'
    )
    description: Optional[str] = Field(
        default=None,
        description='The description of the memory container.'
    )
    configuration: ConfigurationArgs = Field(
        ...,
        description='The memory container configuration.'
    )
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "name": "agentic memory test",
                    "description": "Store conversations with semantic search and summarization",
                    "configuration": {
                        "embedding_model_type": "TEXT_EMBEDDING",
                        "embedding_model_id": "your-embedding-model-id",
                        "embedding_dimension": 1024,
                        "llm_id": "your-llm-model-id",
                        "strategies": [
                            {
                                "type": "SEMANTIC",
                                "namespace": ["user_id"]
                            }
                        ]
                    }
                },
                {
                    "name": "advanced memory container",
                    "description": "Store conversations with semantic search and summarization",
                    "configuration": {
                        "embedding_model_type": "TEXT_EMBEDDING",
                        "embedding_model_id": "your-embedding-model-id",
                        "embedding_dimension": 1024,
                        "llm_id": "your-llm-model-id",
                        "index_prefix": "my_custom_prefix",
                        "use_system_index": False,
                        "strategies": [
                            {
                                "type": "SEMANTIC",
                                "namespace": ["agent_id"],
                                "configuration": {
                                    "llm_result_path": "$.output.message.content[0].text",
                                    "system_prompt": "Extract semantic information from user conversations",
                                    "llm_id": "strategy-llm-id"
                                }
                            },
                            {
                                "type": "USER_PREFERENCE",
                                "namespace": ["agent_id"],
                                "configuration": {
                                    "llm_result_path": "$.output.message.content[0].text"
                                }
                            },
                            {
                                "type": "SUMMARY",
                                "namespace": ["agent_id"],
                                "configuration": {
                                    "llm_result_path": "$.output.message.content[0].text"
                                }
                            }
                        ],
                        "parameters": {
                            "llm_result_path": "$.output.message.content[0].text"
                        },
                        "index_settings": {
                            "session_index": {
                                "index": {
                                    "number_of_shards": "2",
                                    "number_of_replicas": "2"
                                }
                            },
                            "short_term_memory_index": {
                                "index": {
                                    "number_of_shards": "2",
                                    "number_of_replicas": "2"
                                }
                            },
                            "long_term_memory_index": {
                                "index": {
                                    "number_of_shards": "2",
                                    "number_of_replicas": "2"
                                }
                            },
                            "long_term_memory_history_index": {
                                "index": {
                                    "number_of_shards": "2",
                                    "number_of_replicas": "2"
                                }
                            }
                        }
                    }
                }
            ]
        }


class GetMemoryArgs(baseToolArgs):
    """Arguments for retrieving a specific memory by its type and ID."""
    
    memory_container_id: str = Field(
        ...,
        description='The ID of the memory container from which to retrieve the memory.'
    )
    memory_type: MemoryType = Field(
        ...,
        alias='type',
        description='The memory type. Valid values are sessions, working, long-term, and history.'
    )
    id: str = Field(
        ...,
        description='The ID of the memory to retrieve.'
    )
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "memory_container_id": "HudqiJkB1SltqOcZusVU",
                    "type": "working",
                    "id": "XyEuiJkBeh2gPPwzjYWM"
                },
                {
                    "memory_container_id": "HudqiJkB1SltqOcZusVU",
                    "type": "long-term", 
                    "id": "DcxjTpkBvwXRq366C1Zz"
                },
                {
                    "memory_container_id": "HudqiJkB1SltqOcZusVU",
                    "type": "sessions",
                    "id": "CcxjTpkBvwXRq366A1aE"
                },
                {
                    "memory_container_id": "HudqiJkB1SltqOcZusVU",
                    "type": "history",
                    "id": "eMxnTpkBvwXRq366hmAU"
                }
            ]
        }

class CreateSessionArgs(baseToolArgs):
    """Arguments for creating a new session in a memory container."""
    
    memory_container_id: str = Field(
        ...,
        description='The ID of the memory container where the session will be created.'
    )
    session_id: Optional[str] = Field(
        default=None,
        description='A custom session ID. If provided, this ID is used for the session. If not provided, a random ID is generated.'
    )
    summary: Optional[str] = Field(
        default=None,
        description='A session summary or description.'
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description='Additional metadata for the session provided as key-value pairs.'
    )
    namespace: Optional[Dict[str, str]] = Field(
        default=None,
        description='Namespace information for organizing the session.'
    )
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "memory_container_id": "SdjmmpgBOh0h20Y9kWuN",
                    "session_id": "abc123",
                    "metadata": {
                        "key1": "value1"
                    }
                },
                {
                    "memory_container_id": "SdjmmpgBOh0h20Y9kWuN",
                    "summary": "This is a test session",
                    "metadata": {
                        "key1": "value1"
                    },
                    "namespace": {
                        "user_id": "bob"
                    }
                },
                {
                    "memory_container_id": "SdjmmpgBOh0h20Y9kWuN",
                    # Session ID will be auto-generated
                    "summary": "Session for user onboarding",
                    "namespace": {
                        "user_id": "alice",
                        "agent_id": "onboarding_bot"
                    },
                    "metadata": {
                        "priority": "high",
                        "category": "onboarding"
                    }
                }
            ]
        }


class AddMemoriesArgs(baseToolArgs):
    """Arguments for adding memories to a memory container."""
    
    memory_container_id: str = Field(
        ...,
        description='The ID of the memory container to add the memory to.'
    )
    messages: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description='A list of messages for a conversational payload. Each message must include a content field specified as an array of objects. Each object must contain the type (for example, text) and the corresponding content. Each message may include a role (commonly, user or assistant) when infer is set to true. Required when payload_type is conversational.'
    )
    structured_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description='Structured data content for data memory. Required when payload_type is data.'
    )
    binary_data: Optional[str] = Field(
        default=None,
        description='Binary data content encoded as a Base64 string for binary payloads.'
    )
    payload_type: PayloadType = Field(
        ...,
        description='The type of payload. Valid values are conversational or data.'
    )
    namespace: Optional[Dict[str, str]] = Field(
        default=None,
        description='The namespace context for organizing memories (for example, user_id, session_id, or agent_id). If session_id is not specified in the namespace field and disable_session: false (default is true), a new session with a new session ID is created.'
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description='Additional metadata for the memory (for example, status, branch, or custom fields).'
    )
    tags: Optional[Dict[str, Any]] = Field(
        default=None,
        description='Tags for categorizing and organizing memories.'
    )
    infer: Optional[bool] = Field(
        default=False,
        description='Whether to use a large language model (LLM) to extract key information from messages. Default is false. When true, the LLM extracts key information from the original text and stores it as a memory.'
    )
    
    @model_validator(mode='after')
    def validate_payload_requirements(self) -> 'AddMemoriesArgs':
        """Validate that the correct fields are provided based on payload_type."""
        errors = []
        
        if self.payload_type == 'conversational':
            if not self.messages:
                errors.append("'messages' field is required when payload_type is 'conversational'")
            if self.structured_data:
                errors.append("'structured_data' should not be provided when payload_type is 'conversational'")
                
        elif self.payload_type == 'data':
            if not self.structured_data:
                errors.append("'structured_data' field is required when payload_type is 'data'")
            if self.messages:
                errors.append("'messages' should not be provided when payload_type is 'data'")
        
        # Validate that at least one content field is provided
        content_fields = [self.messages, self.structured_data, self.binary_data]
        if not any(field for field in content_fields):
            errors.append("At least one content field (messages, structured_data, or binary_data) must be provided")
        
        if errors:
            raise ValueError("; ".join(errors))
            
        return self
    
    @model_validator(mode='after')
    def validate_messages_structure(self) -> 'AddMemoriesArgs':
        """Validate the structure of messages if provided."""
        if self.messages:
            for i, message in enumerate(self.messages):
                if not isinstance(message, dict):
                    raise ValueError(f"Message at index {i} must be a dictionary")
                
                if 'content' not in message:
                    raise ValueError(f"Message at index {i} must have a 'content' field")
                
                content = message['content']
                if not isinstance(content, list):
                    raise ValueError(f"Content in message at index {i} must be a list")
                
                for j, content_item in enumerate(content):
                    if not isinstance(content_item, dict):
                        raise ValueError(f"Content item at index {j} in message {i} must be a dictionary")
                    if 'type' not in content_item:
                        raise ValueError(f"Content item at index {j} in message {i} must have a 'type' field")
                    if 'text' not in content_item:
                        raise ValueError(f"Content item at index {j} in message {i} must have a 'text' field")
        
        return self

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "memory_container_id": "SdjmmpgBOh0h20Y9kWuN",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "text": "I'm Bob, I really like swimming.",
                                    "type": "text"
                                }
                            ]
                        },
                        {
                            "role": "assistant", 
                            "content": [
                                {
                                    "text": "Cool, nice. Hope you enjoy your life.",
                                    "type": "text"
                                }
                            ]
                        }
                    ],
                    "namespace": {
                        "user_id": "bob"
                    },
                    "metadata": {
                        "status": "checkpoint",
                        "branch": {
                            "branch_name": "high",
                            "root_event_id": "228nadfs879mtgk"
                        }
                    },
                    "tags": {
                        "topic": "personal info"
                    },
                    "infer": True,
                    "payload_type": "conversational"
                },
                {
                    "memory_container_id": "SdjmmpgBOh0h20Y9kWuN",
                    "structured_data": {
                        "time_range": {
                            "start": "2025-09-11",
                            "end": "2025-09-15"
                        }
                    },
                    "namespace": {
                        "agent_id": "testAgent1"
                    },
                    "metadata": {
                        "status": "checkpoint",
                        "anyobject": "abc"
                    },
                    "tags": {
                        "topic": "agent_state"
                    },
                    "infer": False,
                    "payload_type": "data"
                }
            ]
        }


class SearchMemoryArgs(baseToolArgs):
    """Arguments for searching memories of a specific type within a memory container."""
    
    memory_container_id: str = Field(
        ...,
        description='The ID of the memory container.'
    )
    memory_type: MemoryType = Field(
        ...,
        alias='type',
        description='The memory type. Valid values are sessions, working, long-term, and history.'
    )
    query: Dict[str, Any] = Field(
        ...,
        description='The search query using OpenSearch query DSL.'
    )
    sort: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description='Sort specification for the search results.'
    )

    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "memory_container_id": "HudqiJkB1SltqOcZusVU",
                    "type": "sessions",
                    "query": {
                        "match_all": {}
                    },
                    "sort": [
                        {
                            "created_time": {
                                "order": "desc"
                            }
                        }
                    ]
                },
                {
                    "memory_container_id": "HudqiJkB1SltqOcZusVU",
                    "type": "long-term",
                    "query": {
                        "bool": {
                            "must": [
                                {
                                    "term": {
                                        "namespace.user_id": "bob"
                                    }
                                }
                            ]
                        }
                    },
                    "sort": [
                        {
                            "created_time": {
                                "order": "desc"
                            }
                        }
                    ]
                },
                {
                    "memory_container_id": "HudqiJkB1SltqOcZusVU",
                    "type": "history",
                    "query": {
                        "match_all": {}
                    },
                    "sort": [
                        {
                            "created_time": {
                                "order": "desc"
                            }
                        }
                    ]
                },
                {
                    "memory_container_id": "HudqiJkB1SltqOcZusVU",
                    "type": "working",
                    "query": {
                        "bool": {
                            "must": [
                                {
                                    "term": {
                                        "namespace.user_id": "bob"
                                    }
                                }
                            ],
                            "must_not": [
                                {
                                    "exists": {
                                        "field": "tags.parent_memory_id"
                                    }
                                }
                            ]
                        }
                    },
                    "sort": [
                        {
                            "created_time": {
                                "order": "desc"
                            }
                        }
                    ]
                },
                {
                    "memory_container_id": "HudqiJkB1SltqOcZusVU",
                    "type": "working",
                    "query": {
                        "term": {
                            "namespace.session_id": "123"
                        }
                    },
                    "sort": [
                        {
                            "created_time": {
                                "order": "desc"
                            }
                        }
                    ]
                }
            ]
        }

class DeleteMemoryArgs(baseToolArgs):
    """Arguments for deleting a specific memory by its type and ID."""
    
    memory_container_id: str = Field(
        ...,
        description='The ID of the memory container from which to delete the memory.'
    )
    memory_type: MemoryType = Field(
        ...,
        alias='type',
        description='The type of memory to delete. Valid values are sessions, working, long-term, and history.'
    )
    id: str = Field(
        ...,
        description='The ID of the specific memory to delete.'
    )
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "memory_container_id": "HudqiJkB1SltqOcZusVU",
                    "type": "working",
                    "id": "XyEuiJkBeh2gPPwzjYWM"
                },
                {
                    "memory_container_id": "HudqiJkB1SltqOcZusVU",
                    "type": "long-term",
                    "id": "DcxjTpkBvwXRq366C1Zz"
                },
                {
                    "memory_container_id": "HudqiJkB1SltqOcZusVU",
                    "type": "sessions",
                    "id": "CcxjTpkBvwXRq366A1aE"
                },
                {
                    "memory_container_id": "HudqiJkB1SltqOcZusVU",
                    "type": "history",
                    "id": "eMxnTpkBvwXRq366hmAU"
                }
            ]
        }


class DeleteMemoryByQueryArgs(baseToolArgs):
    """Arguments for deleting memories by query."""
    
    memory_container_id: str = Field(
        ...,
        description='The ID of the memory container from which to delete the memory.'
    )
    memory_type: MemoryType = Field(
        ...,
        alias='type',
        description='The type of memory to delete. Valid values are sessions, working, long-term, and history.'
    )
    query: Dict[str, Any] = Field(
        ...,
        description='The query to match the memories you want to delete. This should be a valid OpenSearch query DSL object.'
    )
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "memory_container_id": "HudqiJkB1SltqOcZusVU",
                    "type": "working",
                    "query": {
                        "match": {
                            "owner_id": "admin"
                        }
                    }
                },
                {
                    "memory_container_id": "HudqiJkB1SltqOcZusVU",
                    "type": "long-term",
                    "query": {
                        "range": {
                            "created_time": {
                                "lt": "2025-09-01"
                            }
                        }
                    }
                },
                {
                    "memory_container_id": "HudqiJkB1SltqOcZusVU",
                    "type": "sessions",
                    "query": {
                        "term": {
                            "namespace.user_id": "inactive_user"
                        }
                    }
                }
            ]
        }


class UpdateMemoryArgs(baseToolArgs):
    """Arguments for updating a specific memory by its type and ID."""
    
    memory_container_id: str = Field(
        ...,
        description='The ID of the memory container.'
    )
    memory_type: Literal[MemoryType.sessions, MemoryType.working, MemoryType.long_term] = Field(
        ...,
        alias='type',
        description='The memory type. Valid values are sessions, working, and long-term. Note that history memory cannot be updated.'
    )
    id: str = Field(
        ...,
        description='The ID of the memory to update.'
    )
    # Session memory fields
    summary: Optional[str] = Field(
        default=None,
        description='The summary of the session.'
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description='Additional metadata for the memory (for example, status, branch, or custom fields).'
    )
    agents: Optional[Dict[str, Any]] = Field(
        default=None,
        description='Additional information about the agents.'
    )
    additional_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description='Additional metadata to associate with the session.'
    )
    # Working memory fields
    messages: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description='Updated conversation messages (for conversation type).'
    )
    structured_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description='Updated structured data content (for data memory payloads).'
    )
    binary_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description='Updated binary data content (for data memory payloads).'
    )
    tags: Optional[Dict[str, Any]] = Field(
        default=None,
        description='Updated tags for categorization.'
    )
    # Long-term memory fields
    memory: Optional[str] = Field(
        default=None,
        description='The updated memory content.'
    )
    
    @model_validator(mode='after')
    def validate_memory_type_fields(self) -> 'UpdateMemoryArgs':
        """Validate that only appropriate fields are used for each memory type."""
        errors = []
        
        # Fields that should only be used for session memory
        session_only_fields = ['summary', 'agents', 'additional_info']
        # Fields that should only be used for working memory  
        working_only_fields = ['messages', 'structured_data', 'binary_data']
        # Fields that should only be used for long-term memory
        long_term_only_fields = ['memory']
        
        if self.memory_type == 'sessions':
            for field in working_only_fields + long_term_only_fields:
                if getattr(self, field) is not None:
                    errors.append(f"Field '{field}' should not be provided when updating session memory")
                    
        elif self.memory_type == 'working':
            for field in session_only_fields + long_term_only_fields:
                if getattr(self, field) is not None:
                    errors.append(f"Field '{field}' should not be provided when updating working memory")
            
            # For working memory, at least one updatable field should be provided
            working_fields = ['messages', 'structured_data', 'binary_data', 'tags', 'metadata']
            if not any(getattr(self, field) for field in working_fields):
                errors.append("At least one field (messages, structured_data, binary_data, tags, or metadata) must be provided for updating working memory")
                
        elif self.memory_type == 'long-term':
            for field in session_only_fields + working_only_fields:
                if getattr(self, field) is not None:
                    errors.append(f"Field '{field}' should not be provided when updating long-term memory")
            
            # For long-term memory, at least one updatable field should be provided
            if self.memory is None and self.tags is None and self.metadata is None:
                errors.append("At least one field (memory, tags, or metadata) must be provided for updating long-term memory")
        
        if errors:
            raise ValueError("; ".join(errors))
            
        return self
    
    @model_validator(mode='after') 
    def validate_working_memory_content(self) -> 'UpdateMemoryArgs':
        """Validate working memory content structure."""
        if self.memory_type == 'working' and self.messages:
            for i, message in enumerate(self.messages):
                if not isinstance(message, dict):
                    raise ValueError(f"Message at index {i} must be a dictionary")
                # Basic structure validation - can be expanded as needed
                if 'content' not in message:
                    raise ValueError(f"Message at index {i} should have a 'content' field for working memory updates")
        
        return self

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "memory_container_id": "HudqiJkB1SltqOcZusVU",
                    "type": "sessions",
                    "id": "N2CDipkB2Mtr6INFFcX8",
                    "additional_info": {
                        "key1": "value1",
                        "last_activity": "2025-09-15T17:30:00Z"
                    }
                },
                {
                    "memory_container_id": "HudqiJkB1SltqOcZusVU",
                    "type": "working", 
                    "id": "XyEuiJkBeh2gPPwzjYWM",
                    "tags": {
                        "topic": "updated_topic",
                        "priority": "high"
                    }
                },
                {
                    "memory_container_id": "HudqiJkB1SltqOcZusVU",
                    "type": "long-term",
                    "id": "DcxjTpkBvwXRq366C1Zz",
                    "memory": "User's name is Bob Smith",
                    "tags": {
                        "topic": "personal info",
                        "updated": "true"
                    }
                },
                {
                    "memory_container_id": "HudqiJkB1SltqOcZusVU",
                    "type": "working",
                    "id": "another_working_memory_id",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "text": "Updated user message",
                                    "type": "text"
                                }
                            ]
                        }
                    ],
                    "metadata": {
                        "status": "updated"
                    }
                }
            ]
        }