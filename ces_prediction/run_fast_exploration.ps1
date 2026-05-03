$env:CES_SEED = "42"
$env:CES_EPOCHS = "3"
$env:CES_MAX_TRAIN_SAMPLES = "50000"
$env:CES_MAX_VAL_SAMPLES = "10000"
$env:CES_BATCH_SIZE = "1024"
$env:CES_TEMPORAL_SUBSETS = "1"

python .\automl_agent_loop.py --max-iterations 20
