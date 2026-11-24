# B108: author-approved direct official MCP delivery

2026-11-24, before any Colab timing fit or data upload. Author requests Colab
execution through MCP. The official googlecolab/colab-mcp connection is live;
its first diagnostic reported CPU,so no GPU timing is launched until an actual
T4 is observed. User is asked to choose T4 in the connected notebook UI.

Amend the original manual-upload delivery only: the same hash-bound9.26MiB
private archive may be delivered automatically through the official notebook
MCP tools. Send binary chunks in a single owned transfer cell,acknowledge each
offset,verify whole-archive SHA256 and flat archive paths before extraction.
No additional observations,TEST,credentials,keys or .env are transferred.
Overwrite only the newly created owned transfer cell; preserve the user's
other notebook cells. Do not expose transfer payloads in chat output.

`research/colab_t4_operator.py` uses the official MCP SDK bridge's recorded
tools to check GPU,stage this archive,start the registered run108 controller,
poll aggregate progress and retrieve the aggregate results archive. The
operator rechecks actual T4 before upload and launch,records cell IDs and
remote PID,and refuses duplicate staging/launch. No local GPU work,local
training resume,new VM provisioning,paid upgrade,Google Drive mounting or
full scientific model selection is performed. The registered numerical
protocol,all30 child attempts and normal-exit qualification stay unchanged.

The existing archive was exported under source3805f7d and CPU equality checks
passed. Preserve its source/data hashes unchanged; this amendment adds only
transport/orchestration. Commit/push this addendum and operator before use.
