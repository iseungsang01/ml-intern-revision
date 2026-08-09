@echo off
REM Detached launcher for the window sweep batch (run from anywhere).
REM
REM FOR_DISABLE_CONSOLE_CTRL_HANDLER stops the Intel Fortran RTL -- pulled in via MKL by
REM numpy/torch -- from installing a console control handler that ABORTS training when the
REM parent console closes. Without it a long batch dies mid-run with
REM   forrtl: error (200): program aborting due to window-CLOSE event
REM and exit code 3221225786 (STATUS_CONTROL_C_EXIT) or 1073807364 (DBG_TERMINATE_PROCESS).
REM Observed twice during the 24-run sweep; --resume absorbed the lost runs.
set FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
set KMP_HANDLE_SIGNALS=0
cd /d "%~dp0..\..\.."
py -u ces_prediction\experiments\window_sweep\run_window_sweep.py --resume >> data\.wsweep_hf_batch.log 2>&1
