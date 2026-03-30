# 1) create memory container (after compose up opensearch)

curl -k -u admin:'HesloVesl0,fGs1' \
-X POST "https://localhost:9200/_plugins/_ml/memory_containers/_create" \
-H 'Content-Type: application/json' \
-d '{
  "name": "local-dev-memory",
  "description": "local dev container (security on)",
  "configuration": {
    "disable_session": false
  }
}' | jq .


# 2) utilize memory_container_id from 1)

# 3) Run MCP server (after compose up app)

source /app/.venv/bin/activate
python -m mcp_server_opensearch --mode multi --transport stream --port 9900 --config ./config.yml --debug

