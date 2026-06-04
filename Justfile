# Chroma memory helpers (export FLYWHEEL_MEMORY_DIR or use default ./.memory)
memory_dir := env_var_or_default("FLYWHEEL_MEMORY_DIR", "./.memory")

# Index all api_docs_dump into collection api_docs (read-only after this)
index-api-docs:
    FLYWHEEL_MEMORY_DIR="{{memory_dir}}" python -m memory.cli index-api-docs

# Rebuild api_docs from scratch
index-api-docs-force:
    FLYWHEEL_MEMORY_DIR="{{memory_dir}}" python -m memory.cli index-api-docs --force

# Query api_docs by task description (top API names + descriptions)
search-api-docs query top_k="20":
    FLYWHEEL_MEMORY_DIR="{{memory_dir}}" python -m memory.cli search-api-docs "{{query}}" --top-k {{top_k}}

# Same, with full doc previews from api_docs_dump
search-api-docs-verbose query top_k="10":
    FLYWHEEL_MEMORY_DIR="{{memory_dir}}" python -m memory.cli search-api-docs "{{query}}" --top-k {{top_k}} --docs

# Search successful code examples by task description
search-code-examples query top_k="5":
    FLYWHEEL_MEMORY_DIR="{{memory_dir}}" python -m memory.cli search-code-examples "{{query}}" --top-k {{top_k}}
