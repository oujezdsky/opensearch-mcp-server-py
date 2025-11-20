import json
from .params import (
    AddAgenticMemoriesArgs,
    CreateAgenticMemoryContainerArgs,
    CreateAgenticMemorySessionArgs,
    DeleteAgenticMemoryByIDArgs,
    DeleteAgenticMemoryByQueryArgs,
    GetAgenticMemoryArgs,
    SearchAgenticMemoryArgs,
    UpdateAgenticMemoryArgs,
)
from opensearch.helper import (
    add_agentic_memories,
    create_agentic_memory_container,
    create_agentic_memory_session,
    delete_agentic_memory_by_id,
    delete_agentic_memory_by_query,
    get_agentic_memory,
    search_agentic_memory,
    update_agentic_memory,
)
from tools.exceptions import HelperOperationError


async def create_agentic_memory_container_tool(
    args: CreateAgenticMemoryContainerArgs,
) -> list[dict]:
    """Tool to create a new agentic memory container.

    Args:
        args: CreateAgenticMemoryContainerArgs containing the name, description, and configuration for the new container.

    Returns:
        list[dict]: A confirmation message with the new container ID in MCP format.
    """
    try:
        from tools.tools import check_tool_compatibility

        await check_tool_compatibility('CreateAgenticMemoryContainerTool', args)
        result = await create_agentic_memory_container(args)

        container_id = result.get('memory_container_id')
        if container_id:
            message = f'Successfully created memory container. ID: {container_id}. Response: {json.dumps(result)}'
        else:
            message = (
                f'Memory container created, but no ID was returned. Response: {json.dumps(result)}'
            )

        return [{'type': 'text', 'text': message}]

    except Exception as e:
        error_to_report = e
        if isinstance(e, HelperOperationError):
            error_to_report = e.original
        return [
            {
                'type': 'text',
                'text': f'Error creating memory container: {str(error_to_report)}',
            }
        ]


async def create_agentic_memory_session_tool(
    args: CreateAgenticMemorySessionArgs,
) -> list[dict]:
    """Tool to create a new session in an agentic memory container.

    Args:
        args: CreateAgenticMemorySessionArgs containing the memory_container_id and optional session details like session_id, summary, metadata, or namespace.

    Returns:
        list[dict]: A confirmation message with the new session ID in MCP format.
    """
    try:
        from tools.tools import check_tool_compatibility

        await check_tool_compatibility('CreateAgenticMemorySessionTool', args)
        result = await create_agentic_memory_session(args)

        session_id = result.get('session_id')
        message = (
            f'Successfully created session. ID: {session_id}. Response: {json.dumps(result)}'
            if session_id
            else f'Session created, but no ID was returned. Response: {json.dumps(result)}'
        )
        return [{'type': 'text', 'text': message}]
    except Exception as e:
        error_to_report = e
        if isinstance(e, HelperOperationError):
            error_to_report = e.original
        return [{'type': 'text', 'text': f'Error creating session: {str(error_to_report)}'}]


async def add_agentic_memories_tool(args: AddAgenticMemoriesArgs) -> list[dict]:
    """Tool to add memories to an agentic memory container.

    Args:
        args: AddAgenticMemoriesArgs containing the memory_container_id, payload_type, and content (either messages or structured_data).

    Returns:
        list[dict]: A confirmation message, often including the new working_memory_id or session_id, in MCP format.
    """
    try:
        from tools.tools import check_tool_compatibility

        await check_tool_compatibility('AddAgenticMemoriesTool', args)
        result = await add_agentic_memories(args)

        session_id = result.get('session_id')
        memory_id = result.get('working_memory_id')

        message = 'Successfully added memory.'
        if memory_id:
            message += f' Working Memory ID: {memory_id}.'
        if session_id:
            message += f' Session ID: {session_id}.'
        message += f' Response: {json.dumps(result)}'

        return [{'type': 'text', 'text': message}]
    except Exception as e:
        error_to_report = e
        if isinstance(e, HelperOperationError):
            error_to_report = e.original
        return [{'type': 'text', 'text': f'Error adding memory: {str(error_to_report)}'}]


async def get_agentic_memory_tool(args: GetAgenticMemoryArgs) -> list[dict]:
    """Tool to retrieve a specific agentic memory by its type and ID.

    Args:
        args: GetAgenticMemoryArgs containing the memory_container_id, memory_type, and the specific memory id.

    Returns:
        list[dict]: The retrieved memory object as a JSON string within a confirmation message, in MCP format.
    """
    try:
        from tools.tools import check_tool_compatibility

        await check_tool_compatibility('GetAgenticMemoryTool', args)
        result = await get_agentic_memory(args)

        message = f'Successfully retrieved memory {args.id} ({args.memory_type.value}): {json.dumps(result)}'
        return [{'type': 'text', 'text': message}]
    except Exception as e:
        error_to_report = e
        if isinstance(e, HelperOperationError):
            error_to_report = e.original
        return [{'type': 'text', 'text': f'Error getting memory: {str(error_to_report)}'}]


async def update_agentic_memory_tool(args: UpdateAgenticMemoryArgs) -> list[dict]:
    """Tool to update a specific agentic memory (session, working, or long-term) by its ID.

    Args:
        args: UpdateAgenticMemoryArgs containing the memory_container_id, memory_type, id, and the fields to be updated.

    Returns:
        list[dict]: A confirmation message of the update operation in MCP format.
    """
    try:
        from tools.tools import check_tool_compatibility

        await check_tool_compatibility('UpdateAgenticMemoryTool', args)
        result = await update_agentic_memory(args)

        memory_id = result.get('_id', args.id)
        message = f'Successfully updated memory {memory_id} ({args.memory_type.value}). Response: {json.dumps(result)}'
        return [{'type': 'text', 'text': message}]
    except Exception as e:
        error_to_report = e
        if isinstance(e, HelperOperationError):
            error_to_report = e.original
        return [{'type': 'text', 'text': f'Error updating memory: {str(error_to_report)}'}]


async def delete_agentic_memory_by_id_tool(
    args: DeleteAgenticMemoryByIDArgs,
) -> list[dict]:
    """Tool to delete a specific agentic memory by its type and ID.

    Args:
        args: DeleteAgenticMemoryByIDArgs containing the memory_container_id, memory_type, and the id of the memory to delete.

    Returns:
        list[dict]: A confirmation message of the deletion in MCP format.
    """
    try:
        from tools.tools import check_tool_compatibility

        await check_tool_compatibility('DeleteAgenticMemoryByIDTool', args)
        result = await delete_agentic_memory_by_id(args)

        memory_id = result.get('_id', args.id)
        message = f'Successfully deleted memory {memory_id} ({args.memory_type.value}). Response: {json.dumps(result)}'
        return [{'type': 'text', 'text': message}]
    except Exception as e:
        error_to_report = e
        if isinstance(e, HelperOperationError):
            error_to_report = e.original
        return [{'type': 'text', 'text': f'Error deleting memory: {str(error_to_report)}'}]


async def delete_agentic_memory_by_query_tool(
    args: DeleteAgenticMemoryByQueryArgs,
) -> list[dict]:
    """Tool to delete agentic memories matching an OpenSearch query DSL.

    Args:
        args: DeleteAgenticMemoryByQueryArgs containing the memory_container_id, memory_type, and the query.

    Returns:
        list[dict]: A summary of the delete-by-query operation, including counts, in MCP format.
    """
    try:
        from tools.tools import check_tool_compatibility

        await check_tool_compatibility('DeleteAgenticMemoryByQueryTool', args)
        result = await delete_agentic_memory_by_query(args)

        deleted_count = result.get('deleted', 0)
        failures = result.get('failures', [])

        if failures:
            message = (
                f'Delete by query for {args.memory_type.value} completed with {len(failures)} failures. '
                f'Deleted: {deleted_count}. Response: {json.dumps(result)}'
            )
        else:
            message = (
                f'Successfully deleted memories by query for {args.memory_type.value}. '
                f'Deleted: {deleted_count}. Response: {json.dumps(result)}'
            )

        return [{'type': 'text', 'text': message}]
    except Exception as e:
        error_to_report = e
        if isinstance(e, HelperOperationError):
            error_to_report = e.original
        return [
            {
                'type': 'text',
                'text': f'Error deleting memories by query: {str(error_to_report)}',
            }
        ]


async def search_agentic_memory_tool(args: SearchAgenticMemoryArgs) -> list[dict]:
    """Tool to search for agentic memories using an OpenSearch query DSL.

    Args:
        args: SearchAgenticMemoryArgs containing the memory_container_id, memory_type, query, and optional sort parameters.

    Returns:
        list[dict]: The search results from OpenSearch as a JSON string within a summary message, in MCP format.
    """
    try:
        from tools.tools import check_tool_compatibility

        await check_tool_compatibility('SearchAgenticMemoryTool', args)
        result = await search_agentic_memory(args)

        hits = result.get('hits', {}).get('hits', [])
        count = len(hits)
        total = result.get('hits', {}).get('total', {}).get('value', count)

        if total == 0:
            message = f'Search results for {args.memory_type.value}: No memories found. Response: {json.dumps(result)}'
        else:
            message = (
                f'Search results for {args.memory_type.value}: Found {total} memories, returning {count}. '
                f'Response: {json.dumps(result)}'
            )

        return [{'type': 'text', 'text': message}]
    except Exception as e:
        error_to_report = e
        if isinstance(e, HelperOperationError):
            error_to_report = e.original
        return [{'type': 'text', 'text': f'Error searching memory: {str(error_to_report)}'}]


__all__ = [
    'create_agentic_memory_container_tool',
    'create_agentic_memory_session_tool',
    'add_agentic_memories_tool',
    'get_agentic_memory_tool',
    'update_agentic_memory_tool',
    'delete_agentic_memory_by_id_tool',
    'delete_agentic_memory_by_query_tool',
    'search_agentic_memory_tool',
]
