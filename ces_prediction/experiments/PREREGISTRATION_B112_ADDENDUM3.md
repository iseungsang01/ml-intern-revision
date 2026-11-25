# B112 addendum3: native Files upload into the existing T4

2026-11-25, export passed, zero scientific fits/inferences. Exact factored
archive is485998833bytes. The Python files.upload widget used by B111 was
slow for a much smaller archive. Use Colab's native Files sidebar upload for
this larger file, then the prepared launch cell reads the exact existing
/content/ces_b112_bundle.zip instead of invoking another upload widget.
Do not launch until native upload completes; size/SHA/manifest/actual T4 and
duplicate-launch checks stay identical. Same B111 notebook and allocated T4.

research/prepare_b112_native.py preserves the widget archive/verification,
refreshes only manifest.sources to include this committed addendum, repacks
the identical payload files without refitting or regenerating inputs, invokes
the original notebook preparer, and replaces only its widget upload statement
with an existing-file instruction. It records the exact final cell/notebook
hash and native helper hash in the private attempt; delivery receipt names
this route. All data/model/statistics bytes remain unchanged. Commit/push
before preparing/using the route. Original evidence remains preserved.
