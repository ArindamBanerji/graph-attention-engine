FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e ".[mcp]"
EXPOSE 8080
CMD ["rgi-mcp-server", "--http", "--port", "8080", "--categories", "alert_triage,incident_response,threat_detection", "--actions", "investigate,escalate,mitigate,monitor", "--factors", "severity,confidence,blast_radius,recurrence,time_sensitivity"]
