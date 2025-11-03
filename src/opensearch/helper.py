# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from semver import Version
from tools.tool_params import *

# Configure logging
logger = logging.getLogger(__name__)


# List all the helper functions, these functions perform a single rest call to opensearch
# these functions will be used in tools folder to eventually write more complex tools
async def list_indices(args: ListIndicesArgs) -> json:
    from .client import get_opensearch_client

    async with get_opensearch_client(args) as client:
        response = await client.cat.indices(format='json')
        return response


async def get_index(args: ListIndicesArgs) -> json:
    """Get detailed information about a specific index.

    Args:
        args: ListIndicesArgs containing the index name

    Returns:
        json: Detailed index information including settings and mappings
    """
    from .client import get_opensearch_client

    async with get_opensearch_client(args) as client:
        response = await client.indices.get(index=args.index)
        return response


async def get_index_mapping(args: GetIndexMappingArgs) -> json:
    from .client import get_opensearch_client

    async with get_opensearch_client(args) as client:
        response = await client.indices.get_mapping(index=args.index)
        return response


async def search_index(args: SearchIndexArgs) -> json:
    from .client import get_opensearch_client

    async with get_opensearch_client(args) as client:
        response = await client.search(index=args.index, body=args.query)
        return response


async def get_shards(args: GetShardsArgs) -> json:
    from .client import get_opensearch_client

    async with get_opensearch_client(args) as client:
        response = await client.cat.shards(index=args.index, format='json')
        return response


async def get_segments(args: GetSegmentsArgs) -> json:
    """Get information about Lucene segments in indices.

    Args:
        args: GetSegmentsArgs containing optional index filter

    Returns:
        json: Segment information for the specified indices or all indices
    """
    from .client import get_opensearch_client

    async with get_opensearch_client(args) as client:
        # If index is provided, filter by that index
        index_param = args.index if args.index else None

        response = await client.cat.segments(index=index_param, format='json')
        return response


async def get_cluster_state(args: GetClusterStateArgs) -> json:
    """Get the current state of the cluster.

    Args:
        args: GetClusterStateArgs containing optional metric and index filters

    Returns:
        json: Cluster state information based on the requested metrics and indices
    """
    from .client import get_opensearch_client

    async with get_opensearch_client(args) as client:
        # Build parameters dictionary with non-None values
        params = {}
        if args.metric:
            params['metric'] = args.metric
        if args.index:
            params['index'] = args.index

        response = await client.cluster.state(**params)
        return response


async def get_nodes(args: CatNodesArgs) -> json:
    """Get information about nodes in the cluster.

    Args:
        args: GetNodesArgs containing optional metrics filter

    Returns:
        json: Node information for the cluster
    """
    from .client import get_opensearch_client

    async with get_opensearch_client(args) as client:
        # If metrics is provided, use it as a parameter
        metrics_param = args.metrics if args.metrics else None

        response = await client.cat.nodes(format='json', h=metrics_param)
        return response


async def get_index_info(args: GetIndexInfoArgs) -> json:
    """Get detailed information about an index including mappings, settings, and aliases.

    Args:
        args: GetIndexInfoArgs containing the index name

    Returns:
        json: Detailed index information
    """
    from .client import get_opensearch_client

    async with get_opensearch_client(args) as client:
        response = await client.indices.get(index=args.index)
        return response


async def get_index_stats(args: GetIndexStatsArgs) -> json:
    """Get statistics about an index.

    Args:
        args: GetIndexStatsArgs containing the index name and optional metric filter

    Returns:
        json: Index statistics
    """
    from .client import get_opensearch_client

    async with get_opensearch_client(args) as client:
        # Build parameters dictionary with non-None values
        params = {}
        if args.metric:
            params['metric'] = args.metric

        response = await client.indices.stats(index=args.index, **params)
        return response


async def get_query_insights(args: GetQueryInsightsArgs) -> json:
    """Get insights about top queries in the cluster.

    Args:
        args: GetQueryInsightsArgs containing connection parameters

    Returns:
        json: Query insights from the /_insights/top_queries endpoint
    """
    from .client import get_opensearch_client

    async with get_opensearch_client(args) as client:
        # Use the transport.perform_request method to make a direct REST API call
        # since the Python client might not have a dedicated method for this endpoint
        response = await client.transport.perform_request(
            method='GET', url='/_insights/top_queries'
        )

        return response


async def get_nodes_hot_threads(args: GetNodesHotThreadsArgs) -> str:
    """Get information about hot threads in the cluster nodes.

    Args:
        args: GetNodesHotThreadsArgs containing connection parameters

    Returns:
        str: Hot threads information from the /_nodes/hot_threads endpoint
    """
    from .client import get_opensearch_client

    async with get_opensearch_client(args) as client:
        # Use the transport.perform_request method to make a direct REST API call
        # The hot_threads API returns text, not JSON
        response = await client.transport.perform_request(method='GET', url='/_nodes/hot_threads')

        return response


async def get_allocation(args: GetAllocationArgs) -> json:
    """Get information about shard allocation across nodes in the cluster.

    Args:
        args: GetAllocationArgs containing connection parameters

    Returns:
        json: Allocation information from the /_cat/allocation endpoint
    """
    from .client import get_opensearch_client

    async with get_opensearch_client(args) as client:
        # Use the cat.allocation method with JSON format
        response = await client.cat.allocation(format='json')

        return response


async def get_long_running_tasks(args: GetLongRunningTasksArgs) -> json:
    """Get information about long-running tasks in the cluster, sorted by running time.

    Args:
        args: GetLongRunningTasksArgs containing limit parameter

    Returns:
        json: Task information from the /_cat/tasks endpoint, sorted by running time
    """
    from .client import get_opensearch_client

    async with get_opensearch_client(args) as client:
        # Use the transport.perform_request method to make a direct REST API call
        # since we need to sort by running_time which might not be directly supported by the client
        response = await client.transport.perform_request(
            method='GET',
            url='/_cat/tasks',
            params={
                's': 'running_time:desc',  # Sort by running time in descending order
                'format': 'json',
            },
        )

        # Limit the number of tasks returned if specified
        if args.limit and isinstance(response, list):
            return response[: args.limit]

        return response


async def get_nodes_info(args: GetNodesArgs) -> json:
    """Get detailed information about nodes in the cluster.

    Args:
        args: GetNodesArgs containing optional node_id, metric filters, and other parameters

    Returns:
        json: Detailed node information from the /_nodes endpoint
    """
    from .client import get_opensearch_client

    async with get_opensearch_client(args) as client:
        # Build the URL path based on provided parameters
        url_parts = ['/_nodes']

        # Add node_id if provided
        if args.node_id:
            url_parts.append(args.node_id)

        # Add metric if provided
        if args.metric:
            url_parts.append(args.metric)

        url = '/'.join(url_parts)

        # Use the transport.perform_request method to make a direct REST API call
        response = await client.transport.perform_request(method='GET', url=url)

        return response


async def get_opensearch_version(args: baseToolArgs) -> Version:
    """Get the version of OpenSearch cluster.

    Returns:
        Version: The version of OpenSearch cluster (SemVer style)
    """
    from .client import get_opensearch_client

    try:
        async with get_opensearch_client(args) as client:
            response = await client.info()
            return Version.parse(response['version']['number'])
    except Exception as e:
        logger.error(f'Error getting OpenSearch version: {e}')
        return None


def create_agentic_memory_container(args: CreateAgenticMemoryContainerArgs) -> json:
    """Create a new memory container for storing agentic memories.
    
    Args:
        args: CreateAgenticMemoryContainerArgs containing name, optional description, and configuration
        
    Returns:
        json: Response from the create memory container endpoint, likely including the created container details
    """
    from .client import initialize_client
    
    client = initialize_client(args)
    
    url = '/_plugins/_ml/memory_containers/_create'
    

    body = args.model_dump(exclude_none=True)
    
    response = client.transport.perform_request(
        method='POST',
        url=url,
        body=body
    )
    
    return response


def get_agentic_memory(args: GetAgenticMemoryArgs) -> json:
    """Retrieve a specific agentic memory by its type and ID from the memory container.
    
    Args:
        args: GetAgenticMemoryArgs containing memory_container_id, memory_type, and id
        
    Returns:
        json: The retrieved memory information from the /_memory endpoint
    """
    from .client import initialize_client
    
    client = initialize_client(args)
    
    url_parts = [
        '/_plugins/_ml/memory_containers',
        args.memory_container_id,
        'memories',
        args.memory_type,
        args.id
    ]
    url = '/'.join(url_parts)
    
    response = client.transport.perform_request(
        method='GET',
        url=url
    )
    
    return response


def create_agentic_memory_session(args: CreateAgenticMemorySessionArgs) -> json:
    """Create a new agentic memory session in the specified memory container.
    
    Args:
        args: CreateSessionArgs containing memory_container_id and optional session_id, summary, metadata, namespace
        
    Returns:
        json: Response from the session creation endpoint
    """
    from .client import initialize_client
    
    client = initialize_client(args)
    
    url_parts = [
        '/_plugins/_ml/memory_containers',
        args.memory_container_id,
        'memories/sessions',
    ]
    url = '/'.join(url_parts)

    body = args.model_dump(exclude={'memory_container_id'}, exclude_none=True)
    
    response = client.transport.perform_request(
        method='POST',
        url=url,
        body=body if body else None
    )
    
    return response


def add_agentic_memories(args: AddAgenticMemoriesArgs) -> json:
    """Add agentic memories to the specified memory container based on the payload type.
    
    Args:
        args: AddAgenticMemoriesArgs containing memory_container_id, payload_type, and content fields like messages or structured_data, plus optional namespace, metadata, tags, infer
        
    Returns:
        json: Response from the add memories endpoint
    """
    from .client import initialize_client
    
    client = initialize_client(args)
    
    url_parts = [
        '/_plugins/_ml/memory_containers',
        args.memory_container_id,
        'memories',
    ]
    url = '/'.join(url_parts)
    
    body = args.model_dump(exclude={'memory_container_id'}, exclude_none=True)
    
    response = client.transport.perform_request(
        method='POST',
        url=url,
        body=body
    )
    
    return response

def update_agentic_memory(args: UpdateAgenticMemoryArgs) -> json:
    """Update a specific agentic memory by its type and ID in the memory container.
    
    Args:
        args: UpdateAgenticMemoryArgs containing memory_container_id, memory_type, id, and optional update fields based on type
        
    Returns:
        json: Response from the update memory endpoint
    """
    from .client import initialize_client
    
    client = initialize_client(args)
    
    url_parts = [
        '/_plugins/_ml/memory_containers',
        args.memory_container_id,
        'memories',
        args.memory_type,
        args.id
    ]
    url = '/'.join(url_parts)
    
    body = args.model_dump(exclude={'memory_container_id', 'memory_type', 'id'}, exclude_none=True)
    
    response = client.transport.perform_request(
        method='PUT',
        url=url,
        body=body if body else None
    )
    
    return response


def delete_agentic_memory_by_id(args: DeleteAgenticMemoryByIDArgs) -> json:
    """Delete a specific agentic memory by its type and ID from the memory container.
    
    Args:
        args: DeleteAgenticMemoryByIDArgs containing memory_container_id, memory_type, and id
        
    Returns:
        json: Response from the delete memory endpoint
    """
    from .client import initialize_client
    
    client = initialize_client(args)
    
    url_parts = [
        '/_plugins/_ml/memory_containers',
        args.memory_container_id,
        'memories',
        args.memory_type,
        args.id
    ]
    url = '/'.join(url_parts)
    
    response = client.transport.perform_request(
        method='DELETE',
        url=url
    )
    
    return response


def delete_agentic_memory_by_query(args: DeleteAgenticMemoryByQueryArgs) -> json:
    """Delete agentic memories matching the provided query from the specified memory type in the container.
    
    Args:
        args: DeleteAgenticMemoryByQueryArgs containing memory_container_id, memory_type, and query
        
    Returns:
        json: Response from the delete memory by query endpoint
    """
    from .client import initialize_client
    
    client = initialize_client(args)
    
    url_parts = [
        '/_plugins/_ml/memory_containers',
        args.memory_container_id,
        'memories',
        args.memory_type,
        '_delete_by_query'
    ]
    url = '/'.join(url_parts)
    
    body = args.model_dump(exclude={'memory_container_id', 'memory_type'}, exclude_none=True)
    
    response = client.transport.perform_request(
        method='POST',
        url=url,
        body=body
    )
    
    return response


def search_agentic_memory(args: SearchAgenticMemoryArgs) -> json:
    """Search for agentic memories of a specific type within the memory container using OpenSearch query DSL.
    
    Args:
        args: SearchAgenticMemoryArgs containing memory_container_id, memory_type, query, and optional sort
        
    Returns:
        json: Search memories results
    """
    from .client import initialize_client
    
    client = initialize_client(args)
    
    url_parts = [
        '/_plugins/_ml/memory_containers',
        args.memory_container_id,
        'memories',
        args.memory_type,
        '_search'
    ]
    url = '/'.join(url_parts)
    
    body = args.model_dump(exclude={'memory_container_id', 'memory_type'}, exclude_none=True)
    
    response = client.transport.perform_request(
        method='POST',
        url=url,
        body=body
    )
    
    return response