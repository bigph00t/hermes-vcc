# hermes-vcc

**Lossless conversation archiving and VCC-enhanced compression for AI agents.**

<!-- Badges (uncomment when published / CI is wired) -->
<!-- ![PyPI](https://img.shields.io/pypi/v/hermes-vcc) -->
<!-- ![Python](https://img.shields.io/pypi/pyversions/hermes-vcc) -->
<!-- ![License](https://img.shields.io/github/license/nousresearch/hermes-vcc) -->
<!-- ![Tests](https://img.shields.io/github/actions/workflow/status/nousresearch/hermes-vcc/tests.yml?label=tests) -->

---

## Motivation

Every long-running AI agent session faces the same problem: **context windows are finite**. When the conversation grows too large, the agent must compress it — summarizing older turns to free token budget. This works, but it is fundamentally lossy. Tool outputs, exact error messages, reasoning chains, multi-step debugging sessions — all reduced to a paragraph.

**hermes-vcc** solves this by adding a lossless archival layer that runs *before* each compression cycle:

1. The full conversation is converted to a structured format and archived to disk.
2. The VCC (Virtual Context Compiler) produces human-readable and machine-searchable views of the archive.
3. A recovery tool lets the agent query these archives *after* compression, retrieving exact details on demand.

The result: aggressive compression thresholds become safe because nothing is truly lost. The agent can compress early and often, keeping its active context lean, while retaining the ability to recover any detail from any point in the session.

## Key Concepts

| Concept | Module | Purpose |
|---------|--------|---------|
| **Adapter** | `hermes_vcc.adapter` | Converts OpenAI chat-format messages to VCC-compatible Anthropic JSONL records |
| **Archive** | `hermes_vcc.archive` | Writes timestamped JSONL snapshots and compiles VCC views before each compression |
| **Recovery Tool** | `hermes_vcc.recovery` | Agent-facing tool (`vcc_recover`) for querying archives post-compression |
| **Enhanced Summary** | `hermes_vcc.enhanced_summary` | Uses VCC `.min.txt` as a structural skeleton to improve LLM-generated summaries |
| **Hooks** | `hermes_vcc.hooks` | Non-invasive monkey-patching to install archiving and recovery on a running agent |
| **Config** | `hermes_vcc.config` | Reads `compression.vcc` from Hermes `config.yaml` with safe defaults |

## Quick Start

### Install

```bash
pip install hermes-vcc
# or from source:
pip install -e /path/to/hermes-vcc
```

### Configure (for Hermes agent integration)

Add to your `~/.hermes/config.yaml`:

```yaml
compression:
  vcc:
    enabled: true
    archive_dir: ~/.hermes/vcc_archives
    enhanced_summary: true
    recovery_tool: true
    retain_archives: 10
```

### Automatic Operation

When integrated with Hermes, VCC hooks install automatically. Every time the agent compresses context:

1. The full conversation is archived as JSONL + compiled VCC views.
2. The compression summary is enriched with VCC structural context.
3. The agent gains a `vcc_recover` tool to query archived details.

No code changes to the agent are required.

## How It Works

```
                         Hermes Agent
                              |
                     [context too large]
                              |
                              v
                +--------------------------+
                |   _compress_context()    |
                |   (monkey-patched by     |
                |    hermes-vcc hooks)     |
                +--------------------------+
                              |
              +---------------+---------------+
              |                               |
              v                               v
  +---------------------+        +----------------------+
  |   ARCHIVE PHASE     |        |   COMPRESS PHASE     |
  |                     |        |   (original method)   |
  |  1. adapter.py      |        |                      |
  |     OpenAI -> JSONL  |        |  LLM summarizes the  |
  |                     |        |  conversation with    |
  |  2. archive.py      |        |  VCC .min.txt as     |
  |     Write JSONL to   |        |  structural backbone  |
  |     disk per cycle  |        |                      |
  |                     |        +----------------------+
  |  3. VCC compile     |                    |
  |     .jsonl -> .txt   |                    v
  |     .jsonl -> .min   |        +----------------------+
  |                     |        |  Compressed context   |
  |  4. manifest.json   |        |  returned to agent    |
  |     Update metadata |        +----------------------+
  +---------------------+
              |
              v
  +---------------------+
  |   ON-DISK ARCHIVE   |
  |                     |
  |   vcc_archives/     |
  |   +-- session_abc/  |
  |       +-- cycle_1.jsonl
  |       +-- cycle_1.txt
  |       +-- cycle_1.min.txt
  |       +-- cycle_2.jsonl
  |       +-- cycle_2.txt
  |       +-- cycle_2.min.txt
  |       +-- manifest.json
  +---------------------+
              ^
              |
  +---------------------+
  |   RECOVERY TOOL     |
  |   vcc_recover       |
  |                     |
  |   list   -> cycles  |
  |   overview -> .min  |
  |   search -> regex   |
  |   read   -> lines   |
  +---------------------+
```

## Recovery Tool

After compression, the agent can invoke `vcc_recover` to retrieve lost details. Here is an example interaction:

```
Agent: I need to check what error occurred in the database migration earlier,
       but that was compressed out of context.

Agent calls: vcc_recover(action="list")
  -> VCC Archives (3 cycles):
       cycle_1: 2026-03-31T10:00:00Z (45 messages, ~12,400 tokens)
       cycle_2: 2026-03-31T11:30:00Z (62 messages, ~18,200 tokens)
       cycle_3: 2026-03-31T13:15:00Z (38 messages, ~9,800 tokens)

Agent calls: vcc_recover(action="search", query="migration.*error")
  -> Search: /migration.*error/ in cycle_3 (2 matches in 284 lines)
     ---
        102: [tool_result] Running database migration...
     >> 103: Error: relation "users_v2" already exists
        104: Migration failed with exit code 1
     ---
     >> 198: Fixed migration error by adding IF NOT EXISTS clause
        199: Re-running migration...
        200: Migration completed successfully

Agent calls: vcc_recover(action="read", query="100-110")
  -> === cycle_3 lines 100-110 (of 284) ===
       100: [assistant] Let me run the migration script.
       101: [tool_use] terminal(command="python manage.py migrate")
       102: [tool_result] Running database migration...
       103: Error: relation "users_v2" already exists
       ...
```

## Configuration Reference

All settings live under `compression.vcc` in `~/.hermes/config.yaml`:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `true` | Master switch for all VCC functionality |
| `archive_dir` | path | `~/.hermes/vcc_archives` | Root directory for session archive storage |
| `enhanced_summary` | bool | `true` | Use VCC `.min.txt` to enrich LLM compression summaries |
| `recovery_tool` | bool | `true` | Register `vcc_recover` tool on the agent |
| `retain_archives` | int | `10` | Maximum archive cycles to keep per session (oldest pruned) |

## Standalone Usage

hermes-vcc can be used outside of Hermes for any application that needs to convert OpenAI-format conversations to structured archives.

```python
from hermes_vcc.adapter import convert_conversation, records_to_jsonl

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is VCC?"},
    {"role": "assistant", "content": "VCC is the Virtual Context Compiler..."},
]

# Convert to VCC JSONL records
records = convert_conversation(messages)

# Serialize to JSONL string
jsonl = records_to_jsonl(records)
print(jsonl)
```

For archival with VCC compilation:

```python
from pathlib import Path
from hermes_vcc.archive import archive_before_compression

session_dir = archive_before_compression(
    messages=messages,
    session_id="my-session",
    archive_dir=Path("./archives"),
    compression_cycle=1,
)
# Produces: ./archives/my-session/cycle_1.jsonl, .txt, .min.txt, manifest.json
```

For post-hoc recovery:

```python
from hermes_vcc.recovery import handle_vcc_recover

# List available archives
result = handle_vcc_recover(action="list", archive_dir=Path("./archives"))

# Search for specific content
result = handle_vcc_recover(
    action="search",
    query="database.*error",
    archive_dir=Path("./archives"),
)
```

## Academic Context

This project builds on the **Virtual Context Compiler (VCC)** by Lvmin Zhang et al. at Stanford University. VCC was introduced as a method to compile conversation transcripts into compressed, view-oriented representations that preserve structural information while reducing token count.

Key results from the VCC paper (evaluated on AppWorld benchmarks):

- **+1 to +4 percentage points** improvement in task completion accuracy when using VCC-compiled context versus raw transcripts.
- **~60% fewer tokens** consumed by VCC-compiled views compared to full conversation logs.
- The `.min.txt` (brief mode) view provides a structural skeleton that captures tool call patterns, decision points, and error recovery flows in a fraction of the original size.

hermes-vcc applies these findings to the practical problem of context compression in long-running agent sessions, using VCC as both an archival format and a compression enhancement.

**Reference:** Lvmin Zhang, *Virtual Context Compiler*, Stanford University / ControlNet. [GitHub: lllyasviel/VCC](https://github.com/lllyasviel/VCC)

## Architecture

```
hermes_vcc/
    __init__.py          # Package metadata and version
    adapter.py           # OpenAI -> VCC JSONL format conversion
    archive.py           # Pre-compression archival pipeline
    recovery.py          # Agent-facing recovery tool (vcc_recover)
    enhanced_summary.py  # VCC-augmented compression summaries
    hooks.py             # Non-invasive agent integration (monkey-patching)
    config.py            # Configuration loading from Hermes config.yaml
    utils.py             # Shared utilities (VCC import, token estimation, etc.)

vendor/
    VCC.py               # Vendored VCC compiler (upstream: lllyasviel/VCC)

tests/
    conftest.py          # Shared fixtures (sample messages, temp dirs)
    test_adapter.py      # Format conversion tests
    ...

docs/
    ARCHITECTURE.md      # System design and data flow
    FORMAT_MAPPING.md    # Detailed OpenAI -> VCC field mapping
    INTEGRATION.md       # Hermes integration guide

examples/
    basic_usage.py       # Standalone adapter + JSONL demo
    manual_archive.py    # On-demand archival demo
    recovery_demo.py     # Recovery tool demo
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed component diagrams and data flow.

## Contributing

1. Fork the repository.
2. Create a feature branch from `main`.
3. Install dev dependencies: `pip install -e ".[dev]"`
4. Run tests: `pytest`
5. Submit a pull request with a clear description.

### Development Setup

```bash
git clone https://github.com/nousresearch/hermes-vcc.git
cd hermes-vcc
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

### Code Standards

- Python 3.11+ required.
- Type hints on all public APIs.
- All public functions must have docstrings.
- Tests required for new functionality.
- No exceptions may propagate from archive/hook code to the agent.

## License

Apache License 2.0. See [LICENSE](LICENSE) for full text.

## Acknowledgments

- **Lvmin Zhang** (Stanford / ControlNet) for the Virtual Context Compiler (VCC) and the research demonstrating its effectiveness on agent benchmarks.
- **Nous Research** for the Hermes agent framework that this project integrates with.
