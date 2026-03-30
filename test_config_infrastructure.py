"""
Config Infrastructure Integration Test
=======================================
Tests the configuration pipeline end-to-end:
  - display_name customization
  - description customization
  - arg description customization (string-only format)
  - memory_container_id auto-populate for agentic memory tools
  - config priority (memory config > env var, file config > CLI)
  - validation error handling

This script imports the module's own code and does NOT require a running server
or OpenSearch cluster.  It exercises the same code paths that the server uses
at startup.

Usage::

    PYTHONPATH=src python test_config_infrastructure.py
"""

import os
import sys
import tempfile
import textwrap

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
INFO = "\033[94mINFO\033[0m"
SECT = "\033[93m----\033[0m"

results: list[tuple[str, bool]] = []


def check(name: str, condition: bool, detail: str = ""):
    """Record and print a single check."""
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}")
    if detail:
        print(f"         {detail}")
    results.append((name, condition))
    return condition


# ---------------------------------------------------------------------------
# Build a minimal TOOL_REGISTRY snapshot for testing (avoids OpenAPI fetch)
# ---------------------------------------------------------------------------
from tools.tool_params import baseToolArgs, ListIndicesArgs, SearchIndexArgs, GetShardsArgs


async def _noop(args):
    return []


def make_tool(name, args_model, description="Test tool", http_methods="GET"):
    return {
        "display_name": name,
        "description": description,
        "input_schema": args_model.model_json_schema(),
        "function": _noop,
        "args_model": args_model,
        "min_version": "1.0.0",
        "http_methods": http_methods,
    }


def fresh_registry():
    """Return a clean registry with a few real tools for testing."""
    return {
        "ListIndexTool": make_tool("ListIndexTool", ListIndicesArgs, "Lists indices"),
        "SearchIndexTool": make_tool("SearchIndexTool", SearchIndexArgs, "Searches index"),
        "GetShardsTool": make_tool("GetShardsTool", GetShardsArgs, "Gets shards"),
    }


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
def write_temp_yaml(content: str) -> str:
    """Write YAML content to a temp file and return path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False)
    f.write(textwrap.dedent(content))
    f.close()
    return f.name


# ===========================================================================
# TEST SUITE 1: Display name, description customization
# ===========================================================================
def test_display_name_and_description():
    print(f"\n[{SECT}] 1. Display name & description customization")

    from tools.config import apply_custom_tool_config

    cfg = write_temp_yaml("""\
        version: "1.0"
        tools:
          ListIndexTool:
            display_name: "MyIndexLister"
            description: "Custom description for listing"
          SearchIndexTool:
            display_name: "SuperSearch"
    """)

    registry = fresh_registry()
    result = apply_custom_tool_config(registry, cfg, {})

    check(
        "ListIndexTool display_name changed",
        result["ListIndexTool"]["display_name"] == "MyIndexLister",
        f"got: {result['ListIndexTool']['display_name']}",
    )
    check(
        "ListIndexTool description changed",
        result["ListIndexTool"]["description"] == "Custom description for listing",
        f"got: {result['ListIndexTool']['description']}",
    )
    check(
        "SearchIndexTool display_name changed",
        result["SearchIndexTool"]["display_name"] == "SuperSearch",
        f"got: {result['SearchIndexTool']['display_name']}",
    )
    check(
        "GetShardsTool unchanged",
        result["GetShardsTool"]["display_name"] == "GetShardsTool",
        f"got: {result['GetShardsTool']['display_name']}",
    )

    os.unlink(cfg)


# ===========================================================================
# TEST SUITE 2: Arg description customization (string-only format)
# ===========================================================================
def test_arg_description():
    print(f"\n[{SECT}] 2. Arg description customization")

    from tools.config import apply_custom_tool_config

    cfg = write_temp_yaml("""\
        version: "1.0"
        tools:
          ListIndexTool:
            args:
              index: "Overridden index description"
          SearchIndexTool:
            args:
              query: "Custom query description"
    """)

    registry = fresh_registry()
    result = apply_custom_tool_config(registry, cfg, {})

    schema_list = result["ListIndexTool"]["input_schema"]
    check(
        "ListIndexTool.index description overridden in schema",
        schema_list["properties"]["index"]["description"] == "Overridden index description",
        f"got: {schema_list['properties']['index']['description'][:60]}",
    )

    # Also verify the Pydantic model_fields description was updated
    args_model = result["ListIndexTool"]["args_model"]
    check(
        "ListIndexTool.index description overridden in model_fields",
        args_model.model_fields["index"].description == "Overridden index description",
        f"got: {args_model.model_fields['index'].description[:60]}",
    )

    schema_search = result["SearchIndexTool"]["input_schema"]
    check(
        "SearchIndexTool.query description overridden",
        schema_search["properties"]["query"]["description"] == "Custom query description",
        f"got: {schema_search['properties']['query']['description'][:60]}",
    )

    os.unlink(cfg)


# ===========================================================================
# TEST SUITE 3: Args config rejects non-string values (dict format)
# ===========================================================================
def test_args_rejects_non_string():
    print(f"\n[{SECT}] 3. Args config rejects non-string values")

    from tools.config import apply_custom_tool_config

    cfg = write_temp_yaml("""\
        version: "1.0"
        tools:
          GetShardsTool:
            args:
              index:
                default: "my-default-index"
    """)

    old_env = os.environ.pop("OPENSEARCH_MEMORY_CONTAINER_ID", None)

    try:
        registry = fresh_registry()
        try:
            apply_custom_tool_config(registry, cfg, {})
            check("rejects dict-format arg config", False, "no exception raised")
        except ValueError as e:
            check(
                "rejects dict-format arg config",
                "must be a string" in str(e),
                f"error: {e}",
            )
    finally:
        if old_env is not None:
            os.environ["OPENSEARCH_MEMORY_CONTAINER_ID"] = old_env
        os.unlink(cfg)


# ===========================================================================
# TEST SUITE 4: Memory container ID auto-populate via config file
# ===========================================================================
def test_memory_container_id():
    print(f"\n[{SECT}] 4. Memory container ID auto-populate (config file)")

    from tools.config import apply_custom_tool_config
    from tools.agentic_memory.params import CreateAgenticMemorySessionArgs

    registry = fresh_registry()
    registry["CreateAgenticMemorySessionTool"] = make_tool(
        "CreateAgenticMemorySessionTool",
        CreateAgenticMemorySessionArgs,
        "Creates a session",
        "POST",
    )

    cfg = write_temp_yaml("""\
        version: "1.0"
        agentic_memory:
          memory_container_id: "cfg-container-123"
    """)

    result = apply_custom_tool_config(registry, cfg, {})
    schema = result["CreateAgenticMemorySessionTool"]["input_schema"]

    check(
        "memory_container_id has default in schema",
        schema["properties"]["memory_container_id"].get("default") == "cfg-container-123",
        f"got: {schema['properties']['memory_container_id'].get('default')}",
    )
    check(
        "memory_container_id removed from required",
        "memory_container_id" not in schema.get("required", []),
        f"required: {schema.get('required', [])}",
    )

    # Verify runtime injection
    from tools.tool_params import validate_args_for_mode

    parsed = validate_args_for_mode(
        {
            "opensearch_cluster_name": "test-cluster",
            "session_id": "sess-1",
            "summary": "test",
        },
        CreateAgenticMemorySessionArgs,
        schema,
    )
    check(
        "memory_container_id auto-populated at runtime",
        parsed.memory_container_id == "cfg-container-123",
        f"got: {parsed.memory_container_id}",
    )

    os.unlink(cfg)


# ===========================================================================
# TEST SUITE 5: Memory container ID from env var
# ===========================================================================
def test_memory_container_id_env():
    print(f"\n[{SECT}] 5. Memory container ID from env var")

    from tools.config import apply_custom_tool_config
    from tools.agentic_memory.params import CreateAgenticMemorySessionArgs

    registry = fresh_registry()
    registry["CreateAgenticMemorySessionTool"] = make_tool(
        "CreateAgenticMemorySessionTool",
        CreateAgenticMemorySessionArgs,
        "Creates a session",
        "POST",
    )

    old_env = os.environ.get("OPENSEARCH_MEMORY_CONTAINER_ID")
    os.environ["OPENSEARCH_MEMORY_CONTAINER_ID"] = "env-container-456"

    try:
        result = apply_custom_tool_config(registry, "", {})
        schema = result["CreateAgenticMemorySessionTool"]["input_schema"]

        check(
            "env var: memory_container_id has default in schema",
            schema["properties"]["memory_container_id"].get("default") == "env-container-456",
            f"got: {schema['properties']['memory_container_id'].get('default')}",
        )

        from tools.tool_params import validate_args_for_mode

        parsed = validate_args_for_mode(
            {
                "opensearch_cluster_name": "test-cluster",
                "session_id": "sess-1",
                "summary": "test",
            },
            CreateAgenticMemorySessionArgs,
            schema,
        )
        check(
            "env var: memory_container_id auto-populated at runtime",
            parsed.memory_container_id == "env-container-456",
            f"got: {parsed.memory_container_id}",
        )
    finally:
        if old_env is None:
            os.environ.pop("OPENSEARCH_MEMORY_CONTAINER_ID", None)
        else:
            os.environ["OPENSEARCH_MEMORY_CONTAINER_ID"] = old_env


# ===========================================================================
# TEST SUITE 6: Config file takes priority over env var
# ===========================================================================
def test_config_priority_over_env():
    print(f"\n[{SECT}] 6. Config file priority over env var")

    from tools.config import apply_custom_tool_config
    from tools.agentic_memory.params import CreateAgenticMemorySessionArgs

    registry = fresh_registry()
    registry["CreateAgenticMemorySessionTool"] = make_tool(
        "CreateAgenticMemorySessionTool",
        CreateAgenticMemorySessionArgs,
        "Creates a session",
        "POST",
    )

    cfg = write_temp_yaml("""\
        version: "1.0"
        agentic_memory:
          memory_container_id: "from-config-file"
    """)

    old_env = os.environ.get("OPENSEARCH_MEMORY_CONTAINER_ID")
    os.environ["OPENSEARCH_MEMORY_CONTAINER_ID"] = "from-env-var"

    try:
        result = apply_custom_tool_config(registry, cfg, {})
        schema = result["CreateAgenticMemorySessionTool"]["input_schema"]

        check(
            "config file wins over env var",
            schema["properties"]["memory_container_id"].get("default") == "from-config-file",
            f"got: {schema['properties']['memory_container_id'].get('default')}",
        )
    finally:
        if old_env is None:
            os.environ.pop("OPENSEARCH_MEMORY_CONTAINER_ID", None)
        else:
            os.environ["OPENSEARCH_MEMORY_CONTAINER_ID"] = old_env
        os.unlink(cfg)


# ===========================================================================
# TEST SUITE 7: Optional fields with None default are NOT injected
# ===========================================================================
def test_optional_none_not_injected():
    print(f"\n[{SECT}] 7. Optional fields (None default) NOT injected")

    from tools.tool_params import validate_args_for_mode
    from tools.agentic_memory.params import AddAgenticMemoriesArgs

    # Build schema and manually set memory_container_id default so validation passes
    schema = AddAgenticMemoriesArgs.model_json_schema()
    schema["properties"]["memory_container_id"]["default"] = "test-container"
    if "memory_container_id" in schema.get("required", []):
        schema["required"].remove("memory_container_id")

    parsed = validate_args_for_mode(
        {
            "opensearch_cluster_name": "test-cluster",
            "payload_type": "conversational",
            "namespace": {"user_id": "u1"},
            "infer": False,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            ],
        },
        AddAgenticMemoriesArgs,
        schema,
    )

    check(
        "structured_data not in model_fields_set (not injected)",
        "structured_data" not in parsed.model_fields_set,
        f"model_fields_set: {parsed.model_fields_set}",
    )
    check(
        "AddAgenticMemories validates successfully",
        parsed.payload_type.value == "conversational",
        f"payload_type: {parsed.payload_type}",
    )


# ===========================================================================
# TEST SUITE 8: CLI overrides (no config file)
# ===========================================================================
def test_cli_overrides():
    print(f"\n[{SECT}] 8. CLI overrides (no config file)")

    from tools.config import apply_custom_tool_config

    registry = fresh_registry()

    cli = {
        "tool.ListIndexTool.display_name": "CLI_Lister",
        "tool.ListIndexTool.description": "CLI description",
        "tool.SearchIndexTool.args.query.description": "CLI query help",
    }

    old_env = os.environ.pop("OPENSEARCH_MEMORY_CONTAINER_ID", None)

    try:
        result = apply_custom_tool_config(registry, "", cli)

        check(
            "CLI: display_name applied",
            result["ListIndexTool"]["display_name"] == "CLI_Lister",
            f"got: {result['ListIndexTool']['display_name']}",
        )
        check(
            "CLI: description applied",
            result["ListIndexTool"]["description"] == "CLI description",
            f"got: {result['ListIndexTool']['description']}",
        )
        schema = result["SearchIndexTool"]["input_schema"]
        check(
            "CLI: arg description applied",
            schema["properties"]["query"]["description"] == "CLI query help",
            f"got: {schema['properties']['query']['description'][:60]}",
        )
    finally:
        if old_env is not None:
            os.environ["OPENSEARCH_MEMORY_CONTAINER_ID"] = old_env


# ===========================================================================
# TEST SUITE 9: Config file ignores CLI when both provided
# ===========================================================================
def test_config_file_ignores_cli():
    print(f"\n[{SECT}] 9. Config file ignores CLI overrides")

    from tools.config import apply_custom_tool_config

    cfg = write_temp_yaml("""\
        version: "1.0"
        tools:
          ListIndexTool:
            display_name: "FromFile"
    """)

    cli = {
        "tool.ListIndexTool.display_name": "FromCLI",
    }

    old_env = os.environ.pop("OPENSEARCH_MEMORY_CONTAINER_ID", None)

    try:
        registry = fresh_registry()
        result = apply_custom_tool_config(registry, cfg, cli)

        check(
            "config file wins: display_name from file, not CLI",
            result["ListIndexTool"]["display_name"] == "FromFile",
            f"got: {result['ListIndexTool']['display_name']}",
        )
    finally:
        if old_env is not None:
            os.environ["OPENSEARCH_MEMORY_CONTAINER_ID"] = old_env
        os.unlink(cfg)


# ===========================================================================
# TEST SUITE 10: Original registry is not mutated (deep copy)
# ===========================================================================
def test_original_not_mutated():
    print(f"\n[{SECT}] 10. Original registry deep-copied (no mutation)")

    from tools.config import apply_custom_tool_config

    cfg = write_temp_yaml("""\
        version: "1.0"
        tools:
          ListIndexTool:
            display_name: "Mutated"
            description: "Mutated desc"
    """)

    registry = fresh_registry()
    original_name = registry["ListIndexTool"]["display_name"]

    old_env = os.environ.pop("OPENSEARCH_MEMORY_CONTAINER_ID", None)

    try:
        apply_custom_tool_config(registry, cfg, {})

        check(
            "input registry display_name unchanged",
            registry["ListIndexTool"]["display_name"] == original_name,
            f"got: {registry['ListIndexTool']['display_name']}",
        )
    finally:
        if old_env is not None:
            os.environ["OPENSEARCH_MEMORY_CONTAINER_ID"] = old_env
        os.unlink(cfg)


# ===========================================================================
# TEST SUITE 11: Validation errors for invalid config
# ===========================================================================
def test_validation_errors():
    print(f"\n[{SECT}] 11. Validation rejects invalid configurations")

    from tools.config import apply_custom_tool_config

    # Invalid tool name
    cfg1 = write_temp_yaml("""\
        version: "1.0"
        tools:
          NonExistentTool:
            display_name: "Foo"
    """)

    registry = fresh_registry()
    old_env = os.environ.pop("OPENSEARCH_MEMORY_CONTAINER_ID", None)

    try:
        try:
            apply_custom_tool_config(registry, cfg1, {})
            check("rejects unknown tool name", False, "no exception raised")
        except ValueError as e:
            check(
                "rejects unknown tool name",
                "not a valid tool name" in str(e),
                f"error: {e}",
            )

        # Invalid arg name
        cfg2 = write_temp_yaml("""\
            version: "1.0"
            tools:
              ListIndexTool:
                args:
                  nonexistent_arg: "desc"
        """)

        registry2 = fresh_registry()
        try:
            apply_custom_tool_config(registry2, cfg2, {})
            check("rejects unknown arg name", False, "no exception raised")
        except ValueError as e:
            check(
                "rejects unknown arg name",
                "does not exist" in str(e),
                f"error: {e}",
            )
        os.unlink(cfg2)

        # Invalid display name pattern
        cfg3 = write_temp_yaml("""\
            version: "1.0"
            tools:
              ListIndexTool:
                display_name: "Has Spaces!"
        """)

        registry3 = fresh_registry()
        try:
            apply_custom_tool_config(registry3, cfg3, {})
            check("rejects invalid display_name pattern", False, "no exception raised")
        except ValueError as e:
            check(
                "rejects invalid display_name pattern",
                "does not follow the required pattern" in str(e),
                f"error: {e}",
            )
        os.unlink(cfg3)

    finally:
        if old_env is not None:
            os.environ["OPENSEARCH_MEMORY_CONTAINER_ID"] = old_env
        os.unlink(cfg1)


# ===========================================================================
# TEST SUITE 12: Memory default does not affect non-memory tools
# ===========================================================================
def test_memory_default_only_memory_tools():
    print(f"\n[{SECT}] 12. Memory default only affects memory tools")

    from tools.config import apply_custom_tool_config

    old_env = os.environ.get("OPENSEARCH_MEMORY_CONTAINER_ID")
    os.environ["OPENSEARCH_MEMORY_CONTAINER_ID"] = "some-container"

    try:
        registry = fresh_registry()
        result = apply_custom_tool_config(registry, "", {})

        # Regular tools should not have memory_container_id in schema at all
        schema = result["ListIndexTool"]["input_schema"]
        check(
            "ListIndexTool has no memory_container_id property",
            "memory_container_id" not in schema.get("properties", {}),
            f"properties: {list(schema.get('properties', {}).keys())}",
        )
    finally:
        if old_env is None:
            os.environ.pop("OPENSEARCH_MEMORY_CONTAINER_ID", None)
        else:
            os.environ["OPENSEARCH_MEMORY_CONTAINER_ID"] = old_env


# ===========================================================================
# Summary
# ===========================================================================
def print_summary():
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"\n{'=' * 60}")
    print(f"  Results: {passed}/{total} passed")
    print(f"{'=' * 60}")
    for name, ok in results:
        status = PASS if ok else FAIL
        print(f"  [{status}] {name}")
    print()
    if passed < total:
        sys.exit(1)


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    print(f"[{INFO}] Config Infrastructure Integration Test")
    print(f"[{INFO}] Testing configuration pipeline without running server\n")

    test_display_name_and_description()
    test_arg_description()
    test_args_rejects_non_string()
    test_memory_container_id()
    test_memory_container_id_env()
    test_config_priority_over_env()
    test_optional_none_not_injected()
    test_cli_overrides()
    test_config_file_ignores_cli()
    test_original_not_mutated()
    test_validation_errors()
    test_memory_default_only_memory_tools()

    print_summary()
