"""
Agentic Memory Integration Test
================================
End-to-end test that exercises all 7 agentic memory tools via the MCP SSE
transport.  The script relies on the server's own config/env-var resolution
for ``memory_container_id`` — it is **never** passed in tool calls, which
validates that the auto-populate mechanism works correctly.

Prerequisites
-------------
1. A running OpenSearch >=3.3.0 cluster with a memory container already
   created (container creation is a one-time admin operation).
2. The MCP server started in streaming mode with the container ID configured::

       # Option A – config file
       agentic_memory:
         memory_container_id: "<your-container-id>"

       # Option B – environment variable
       export OPENSEARCH_MEMORY_CONTAINER_ID="<your-container-id>"

3. Environment variables for this script (all optional, sensible defaults
   are provided):

   - ``MCP_SSE``      – SSE endpoint         (default: http://localhost:9900/sse)
   - ``OS_CLUSTER``   – cluster name for multi-mode (default: local-cluster)
   - ``SESSION_ID``   – custom session ID     (default: auto-generated)

Usage::

    python agentic_test_routine.py
"""

import asyncio
import json
import os
import sys
import time
import uuid

from mcp import ClientSession
from mcp.client.sse import sse_client

# Use the module's own config resolution to read & display the container ID
# so we can verify the server will see the same value.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from tools.config import get_memory_container_id_from_config, should_enable_agentic_memory_tools

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SERVER_SSE = os.getenv("MCP_SSE", "http://localhost:9900/sse")
CLUSTER = os.getenv("OS_CLUSTER", "local-cluster")
SESSION_ID = os.getenv("SESSION_ID", f"test-{uuid.uuid4().hex[:8]}")
CONFIG_PATH = os.getenv("MCP_CONFIG", "")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
INFO = "\033[94mINFO\033[0m"


def txt(resp) -> str:
    """Extract text from a tool response."""
    return resp.content[0].text if resp.content else ""


def extract_json(s: str) -> dict:
    """Parse the JSON payload from a tool text response."""
    marker = "Response:"
    idx = s.find(marker)
    if idx == -1:
        return {}
    return json.loads(s[idx + len(marker):].strip())


def is_error(response_text: str) -> bool:
    """Check if the tool response indicates an error."""
    lower = response_text.lower()
    if "missing required field" in lower:
        return True
    return "error" in lower and "successfully" not in lower


def step(name: str, response_text: str) -> bool:
    """Print a test step result and return True on success."""
    ok = not is_error(response_text)
    status = PASS if ok else FAIL
    # Truncate long responses for readability
    display = response_text[:200] + ("..." if len(response_text) > 200 else "")
    print(f"  [{status}] {name}")
    print(f"         {display}\n")
    return ok


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
def preflight():
    """Validate that memory_container_id is resolvable before hitting the server."""
    print(f"[{INFO}] Pre-flight checks")
    print(f"  SSE endpoint : {SERVER_SSE}")
    print(f"  Cluster      : {CLUSTER}")
    print(f"  Session ID   : {SESSION_ID}")
    print(f"  Config path  : {CONFIG_PATH or '(none – using env var)'}")

    if not should_enable_agentic_memory_tools(CONFIG_PATH):
        print(
            f"\n[{FAIL}] memory_container_id is not configured.\n"
            "  Set it via config file (agentic_memory.memory_container_id) or\n"
            "  environment variable OPENSEARCH_MEMORY_CONTAINER_ID.\n"
        )
        sys.exit(1)

    cid = get_memory_container_id_from_config(CONFIG_PATH)
    print(f"  Container ID : {cid}")
    print()
    return cid


# ---------------------------------------------------------------------------
# Main test routine
# ---------------------------------------------------------------------------
async def run_tests():
    preflight()

    results: list[tuple[str, bool]] = []

    async with sse_client(SERVER_SSE) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 0. Cluster health (sanity check)
            r = await session.call_tool(
                "ClusterHealthTool",
                {"opensearch_cluster_name": CLUSTER},
            )
            results.append(("ClusterHealthTool", step("ClusterHealthTool", txt(r))))

            # ------------------------------------------------------------------
            # 1. CreateAgenticMemorySessionTool
            #    NOTE: memory_container_id is NOT passed – auto-populated by server
            # ------------------------------------------------------------------
            r = await session.call_tool(
                "CreateAgenticMemorySessionTool",
                {
                    "opensearch_cluster_name": CLUSTER,
                    "session_id": SESSION_ID,
                    "summary": "Integration test session",
                    "namespace": {"user_id": "test-user"},
                    "metadata": {"source": "agentic_test_routine"},
                },
            )
            results.append(("CreateSession", step("CreateAgenticMemorySessionTool", txt(r))))

            # ------------------------------------------------------------------
            # 2. AddAgenticMemoriesTool
            # ------------------------------------------------------------------
            r = await session.call_tool(
                "AddAgenticMemoriesTool",
                {
                    "opensearch_cluster_name": CLUSTER,
                    "payload_type": "conversational",
                    "namespace": {"user_id": "test-user", "session_id": SESSION_ID},
                    "metadata": {"topic": "preferences"},
                    "infer": False,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Remember: my favorite color is blue."}
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "Got it — your favorite color is blue."}
                            ],
                        },
                    ],
                },
            )
            results.append(("AddMemories", step("AddAgenticMemoriesTool", txt(r))))

            # Give OpenSearch a moment to index
            time.sleep(1)

            # ------------------------------------------------------------------
            # 3. SearchAgenticMemoryTool
            # ------------------------------------------------------------------
            r = await session.call_tool(
                "SearchAgenticMemoryTool",
                {
                    "opensearch_cluster_name": CLUSTER,
                    "type": "working",
                    "query": {"match_all": {}},
                },
            )
            search_txt = txt(r)
            results.append(("SearchMemory", step("SearchAgenticMemoryTool (working)", search_txt)))

            # Extract a document ID for subsequent operations
            resp = extract_json(search_txt)
            hits = resp.get("hits", {}).get("hits", [])
            if not hits:
                print(f"  [{FAIL}] No hits found — cannot continue GET/UPDATE/DELETE tests.\n")
                print_summary(results)
                return

            hit = sorted(
                hits,
                key=lambda h: h.get("_source", {}).get("created_time", 0),
                reverse=True,
            )[0]
            doc_id = hit["_id"]
            print(f"  [{INFO}] Using document _id: {doc_id}\n")

            # ------------------------------------------------------------------
            # 4. GetAgenticMemoryTool
            # ------------------------------------------------------------------
            r = await session.call_tool(
                "GetAgenticMemoryTool",
                {
                    "opensearch_cluster_name": CLUSTER,
                    "type": "working",
                    "id": doc_id,
                },
            )
            results.append(("GetMemory", step("GetAgenticMemoryTool (working)", txt(r))))

            # ------------------------------------------------------------------
            # 5. UpdateAgenticMemoryTool
            # ------------------------------------------------------------------
            r = await session.call_tool(
                "UpdateAgenticMemoryTool",
                {
                    "opensearch_cluster_name": CLUSTER,
                    "type": "working",
                    "id": doc_id,
                    "metadata": {"updated_by": "agentic_test_routine"},
                },
            )
            results.append(("UpdateMemory", step("UpdateAgenticMemoryTool (working)", txt(r))))

            # ------------------------------------------------------------------
            # 6. DeleteAgenticMemoryByIDTool
            # ------------------------------------------------------------------
            r = await session.call_tool(
                "DeleteAgenticMemoryByIDTool",
                {
                    "opensearch_cluster_name": CLUSTER,
                    "type": "working",
                    "id": doc_id,
                },
            )
            results.append(("DeleteByID", step("DeleteAgenticMemoryByIDTool (working)", txt(r))))

            # Verify deletion
            r = await session.call_tool(
                "GetAgenticMemoryTool",
                {
                    "opensearch_cluster_name": CLUSTER,
                    "type": "working",
                    "id": doc_id,
                },
            )
            get_after = txt(r)
            deleted_ok = "error" in get_after.lower() or "not found" in get_after.lower()
            status = PASS if deleted_ok else FAIL
            print(f"  [{status}] Verify deletion (expect not-found)")
            display = get_after[:200]
            print(f"         {display}\n")
            results.append(("VerifyDeletion", deleted_ok))

            # ------------------------------------------------------------------
            # 7. DeleteAgenticMemoryByQueryTool
            #    Add another memory, then delete it by query
            # ------------------------------------------------------------------
            r = await session.call_tool(
                "AddAgenticMemoriesTool",
                {
                    "opensearch_cluster_name": CLUSTER,
                    "payload_type": "conversational",
                    "namespace": {"user_id": "test-user", "session_id": SESSION_ID},
                    "metadata": {"topic": "cleanup-target"},
                    "infer": False,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "This memory should be deleted by query."}
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "Understood, marking for deletion."}
                            ],
                        },
                    ],
                },
            )
            step("AddAgenticMemoriesTool (for DeleteByQuery)", txt(r))

            time.sleep(1)

            r = await session.call_tool(
                "DeleteAgenticMemoryByQueryTool",
                {
                    "opensearch_cluster_name": CLUSTER,
                    "type": "working",
                    "query": {
                        "match": {"metadata.topic": "cleanup-target"},
                    },
                },
            )
            results.append(("DeleteByQuery", step("DeleteAgenticMemoryByQueryTool", txt(r))))

    print_summary(results)


def print_summary(results: list[tuple[str, bool]]):
    """Print a final summary table."""
    passed = sum(1 for _, ok in results if ok)
    total = len(results)

    print("=" * 60)
    print(f"  Results: {passed}/{total} passed")
    print("=" * 60)
    for name, ok in results:
        status = PASS if ok else FAIL
        print(f"  [{status}] {name}")
    print()

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_tests())
