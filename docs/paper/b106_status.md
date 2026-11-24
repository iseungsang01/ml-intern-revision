# B.106: train LSTM + correction, compare MSE first

State: implementation/preregistration before any new fit.
Author authorized training; Claude/API review is explicitly unavailable and
no new API calls are made. Fixed four B105 units and current report F.
Twenty GPU trajectories = four full LSTMs + sixteen file-OOF LSTMs.
Two post-correction policies (D10-frozen and mixed) use honest OOF residuals.
All eight availability rules are evaluated on common rows against GP+fast
boosting,history-only boosting,GP,hold and saved TCN outputs.

Registration must be committed with code before prepare/register.
Entry `py ces_prediction/experiments/b106_lstm/pipeline106.py prepare`, then
`register`. Status action is read-only. Caches under ignored `data/.b106/`.
No source mutation after launch. Whole8h cap is an upper bound,not an ETA;
each trajectory can stop early. No new architecture search is implied.
