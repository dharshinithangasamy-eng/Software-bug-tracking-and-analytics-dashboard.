import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json

random.seed(42)
np.random.seed(42)

COMPONENTS = ["Auth", "Payment", "Dashboard", "API", "Database", "UI", "Notification", "Search", "Reports", "Admin"]
SEVERITIES = ["Critical", "High", "Medium", "Low"]
STATUSES = ["Open", "In Progress", "Resolved", "Closed", "Reopened"]
ASSIGNEES = ["Alice Chen", "Bob Kumar", "Carol Smith", "David Park", "Eva Lopez", "Frank Wu"]
TAGS = ["regression", "performance", "security", "ui-bug", "data-loss", "crash", "timeout", "memory-leak"]
ENVIRONMENTS = ["Production", "Staging", "Development", "QA"]

severity_weights = [0.1, 0.25, 0.40, 0.25]
status_weights = [0.30, 0.25, 0.25, 0.15, 0.05]

def generate_bugs(n=500):
    bugs = []
    start_date = datetime(2024, 1, 1)
    
    for i in range(n):
        created = start_date + timedelta(days=random.randint(0, 365), hours=random.randint(0, 23))
        severity = random.choices(SEVERITIES, weights=severity_weights)[0]
        status = random.choices(STATUSES, weights=status_weights)[0]
        
        resolve_days = {"Critical": random.randint(1, 5), "High": random.randint(3, 14),
                        "Medium": random.randint(7, 30), "Low": random.randint(14, 60)}
        
        resolved_at = None
        if status in ["Resolved", "Closed"]:
            resolved_at = created + timedelta(days=resolve_days[severity])
        
        bug = {
            "id": f"BUG-{1000 + i}",
            "title": f"Issue in {random.choice(COMPONENTS)} module - {random.choice(TAGS)}",
            "component": random.choice(COMPONENTS),
            "severity": severity,
            "status": status,
            "assignee": random.choice(ASSIGNEES),
            "environment": random.choice(ENVIRONMENTS),
            "tags": random.sample(TAGS, k=random.randint(1, 3)),
            "created_at": created.isoformat(),
            "resolved_at": resolved_at.isoformat() if resolved_at else None,
            "resolution_days": resolve_days[severity] if resolved_at else None,
            "votes": random.randint(0, 50),
            "comments": random.randint(0, 30),
            "is_regression": random.random() < 0.15,
            "sprint": f"Sprint {random.randint(1, 20)}",
        }
        bugs.append(bug)
    
    return pd.DataFrame(bugs)

if __name__ == "__main__":
    df = generate_bugs(500)
    df.to_csv("/home/claude/bug_tracker/data/bugs.csv", index=False)
    print(f"Generated {len(df)} bugs")
    print(df.head())
