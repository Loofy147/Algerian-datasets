# Runbook Template: [Procedure Name]

**Version**: 1.0
**Owner**: [Team Name, e.g., Data Engineering]
**Last Updated**: [YYYY-MM-DD]

---

## 1. Procedure Overview
*A brief, one-sentence description of what this procedure achieves.*

### **When to Use This Runbook:**
*   *Trigger Condition 1 (e.g., A data quality alert for a Gold dataset fires).*
*   *Trigger Condition 2 (e.g., A new data provider needs to be onboarded).*

### **Expected Outcome:**
*   *A clear, one-sentence description of the desired state after this runbook is successfully completed (e.g., The data quality issue is resolved and the pipeline is re-enabled).*

---

## 2. Pre-Requisites & Dependencies
*   **Required Tools**: List of tools needed (e.g., `kubectl`, `psql`, dbt CLI, AWS CLI).
*   **Required Access**: List of access permissions needed (e.g., Kubernetes cluster admin, Production database read/write).
*   **Dependencies**: Other systems or services that must be operational (e.g., Kafka cluster, S3 bucket).

---

## 3. Step-by-Step Procedure

### Step 1: Triage & Acknowledgment
1.  **Acknowledge Alert**: Acknowledge the alert in PagerDuty/Slack to notify the team.
2.  **Create Incident Ticket**: Create a new ticket in Jira, linking to the alert.
3.  **Initial Assessment**: Briefly assess the impact (e.g., "Affects 1 Gold dataset, 5 downstream dashboards").

### Step 2: Containment / Preparation
*This section is for immediate actions to prevent further damage or prepare for the main task.*
1.  **Disable Pipeline**: `airflow dags pause <dag_id>`
2.  **Notify Stakeholders**: Post a message in the `#data-status` Slack channel.
3.  **Isolate Bad Data**: Move the problematic batch from the `bronze` layer to a `quarantine` directory.

### Step 3: Diagnosis & Root Cause Analysis
1.  **Check Logs**: `kubectl logs -l app=<pod_name> -f`
2.  **Query Data**: Run validation queries against the raw data to identify the specific issue (e.g., unexpected `NULL` values, schema change).
3.  **Document Findings**: Add findings to the Jira ticket. **Root Cause**: *[Example: Upstream provider changed date format in their API response.]*

### Step 4: Resolution & Recovery
1.  **Develop Fix**: Create a hotfix branch to update the dbt model / ingestion script to handle the new date format.
2.  **Deploy Fix**: Merge and deploy the hotfix via the CI/CD pipeline.
3.  **Reprocess Data**: Manually trigger a new Airflow DAG run for the quarantined batch.
4.  **Verify Outcome**: Query the "Gold" table to confirm the data is now correct and all quality tests are passing.

### Step 5: Post-Incident / Finalization
1.  **Re-enable Pipeline**: `airflow dags unpause <dag_id>`
2.  **Update Stakeholders**: Post a "Resolved" message in the `#data-status` Slack channel.
3.  **Close Ticket**: Close the Jira ticket, ensuring the root cause and resolution are fully documented.

---

## 4. Rollback Plan
*Instructions for undoing the procedure if something goes wrong.*

1.  **Revert Code**: Revert the hotfix merge commit.
2.  **Re-deploy**: Re-deploy the previous stable version of the application.
3.  **Restore Data (if necessary)**: Use Apache Iceberg's time-travel feature to revert the table to the last known good snapshot. `CALL system.rollback_to_snapshot('<table>', <snapshot_id>);`

---

## 5. Escalation Contacts
*   **Primary On-Call**: [Name / PagerDuty Schedule]
*   **Secondary On-Call**: [Name / PagerDuty Schedule]
*   **Team Lead / Manager**: [Name]
*   **Relevant Stakeholders**: [e.g., Head of Business Intelligence]
