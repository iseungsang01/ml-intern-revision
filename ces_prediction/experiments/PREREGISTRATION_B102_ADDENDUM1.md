# B.102 addendum1: bounded review completion and notifier gates

Before execution, the first Fable review hit max_tokens and is preserved as
incomplete, not accepted as a finished review. Permit one final review attempt
with max8000 output tokens, no further automatic paid retry. Resolve visible
issues before execution: no Claude task launch from the notifier; ignore
pre-launch statuses; check every markdown metric row against the bound JSON;
recheck all published input hashes, deadline, frozen sources and validation
binding after audit and verify copied destinations. Persist restart allowance;
refuse restart on supervisor diagnostic or newly appended traceback. The
watcher gets only the originating local cvmux environment in an ignored local
config, never in API input or logs. The review request contains code only.
