import os
from slack_notifier import send_iteration_update, send_insight_report, send_slack_summary, send_experiment_start

def test_slack():
    print("--- Testing Slack Notifications ---")
    
    # 0. Test Experiment Start
    print("Sending test experiment start notification...")
    send_experiment_start(300, 32)

    # 1. Test Iteration Update
    print("Sending test iteration update...")
    dummy_metrics = {
        "final_val_loss": 0.1234,
        "final_train_loss": 0.5678
    }
    send_iteration_update(999, dummy_metrics)
    
    # 2. Test Insight Report
    print("Sending test insight report...")
    dummy_log = [
        {"iteration": 1, "val_loss": 0.9, "code": "class Model: pass"},
        {"iteration": 2, "val_loss": 0.5, "code": "class Model: pass # improved"},
    ]
    send_insight_report(dummy_log, 2)
    
    # 3. Test Summary
    print("Sending test summary...")
    dummy_history = [
        {"iteration": 1, "val_loss": 0.9},
        {"iteration": 2, "val_loss": 0.5},
    ]
    send_slack_summary(dummy_history, 10)
    
    print("\n--- Test Complete ---")
    print("Check your Slack channel for messages.")

if __name__ == "__main__":
    test_slack()
