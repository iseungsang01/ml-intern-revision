# Project Setup & Execution Commands

git clone https://github.com/iseungsang01/ml-intern-revision.git
cd ml-intern-revision

# Install Dependencies
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

# Required for AutoML Slack notifications
# Set these before running ces_prediction/automl_agent_loop.py
# SLACK_BOT_TOKEN=<your-slack-bot-token>
# SLACK_CHANNEL_ID=<target-channel-id>

# Verify
python -m pytest -q

# Train
python ces_prediction/train.py
