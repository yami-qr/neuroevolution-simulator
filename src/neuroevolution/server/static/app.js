const byId = (id) => document.getElementById(id);

async function bootstrap() {
  try {
    const response = await fetch("/ready");
    if (!response.ok) throw new Error("Database is not ready.");
    byId("connection-dot").className = "status-dot connected";
    byId("connection-label").textContent = "Backend connected";
    const data = await fetch("/api/v1/experiments").then((result) =>
      result.json(),
    );
    byId("experiment-count").textContent = `${data.total} experiments`;
    for (const experiment of data.items) {
      byId("experiment-select").add(new Option(experiment.name, experiment.id));
    }
    byId("experiment-select").disabled = data.total === 0;
  } catch {
    byId("connection-dot").className = "status-dot offline";
    byId("connection-label").textContent = "Backend unavailable";
    byId("connection-banner").hidden = false;
    byId("connection-message").textContent =
      "The database is not ready. Check the API connection and migrations.";
  }
}
byId("retry-connection").addEventListener("click", () => location.reload());
bootstrap();
