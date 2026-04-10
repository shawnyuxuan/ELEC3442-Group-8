prompt = f"""
Predicted sleep cluster:
- cluster_id: {cluster_id}
- cluster_label: {cluster_label}
- cluster_meaning: {cluster_meaning}
- energy_level: {energy_level}
- recommended_work_style: {recommended_work_style}

Today's schedule:
{schedule_text}

Constraints:
{constraints_text}

Task:
Revise the schedule for today based on the predicted sleep state.
Keep fixed events unchanged.
Prioritize high-value tasks.
Suggest a practical adjusted schedule.

Return JSON only.
"""
