# B115 status

failed

{
  "error": "AssertionError('GPU occupied; no other process stopped')",
  "traceback": "Traceback (most recent call last):\n  File \"C:\\Users\\lss\\Documents\\GitHub\\ml-intern-revision\\ces_prediction\\experiments\\b115_budget\\queue115.py\", line 50, in <module>\n    try:main()\n        ~~~~^^\n  File \"C:\\Users\\lss\\Documents\\GitHub\\ml-intern-revision\\ces_prediction\\experiments\\b115_budget\\queue115.py\", line 42, in main\n    assert used<1024 and util<10,'GPU occupied; no other process stopped'\n           ^^^^^^^^^^^^^^^^^^^^^\nAssertionError: GPU occupied; no other process stopped\n",
  "no_retry": true
}

Plan: b115_plan.md. Live: data/.b115/workflow/status.json. No duplicate or frozen-source edits.
