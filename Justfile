python := ".venv/bin/python"

# Search the wins store by task description
# Usage: just search "top 4 R&B songs on spotify"
#        just search "pay joseph on venmo" apps=venmo,phone
#        just search "spotify" top_k=5 no_code=true
search query apps="" top_k="3" no_code="false":
    {{ python }} tools/retrieve_wins.py "{{ query }}" \
        {{ if apps != "" { "--apps " + apps } else { "" } }} \
        --top-k {{ top_k }} \
        {{ if no_code == "true" { "--no-code" } else { "" } }}

# List all wins in the store
search-list:
    {{ python }} tools/retrieve_wins.py --list

# Search and emit raw JSON
search-json query apps="" top_k="5":
    {{ python }} tools/retrieve_wins.py "{{ query }}" \
        {{ if apps != "" { "--apps " + apps } else { "" } }} \
        --top-k {{ top_k }} \
        --json
