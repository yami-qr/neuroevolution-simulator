import {
  ApiClient,
  SelectionGate,
  advancePlayback,
  fitnessCsv,
  frameIndex,
} from "./data.js";
import { Chart } from "./charts.js";
import { Arena } from "./arena.js";

const byId = (id) => document.getElementById(id);
const text = (id, value) => {
  byId(id).textContent = value;
};
const element = (tag, className, content) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (content !== undefined) node.textContent = content;
  return node;
};
const numeric = (value, digits = 1) =>
  value == null
    ? "—"
    : new Intl.NumberFormat("en-GB", {
        maximumFractionDigits: digits,
        minimumFractionDigits: digits,
      }).format(value);
const labelStatus = (status) =>
  ({ timed_out: "Time limit", reached: "Goal reached", active: "Navigating" })[
    status
  ] ??
  status?.replace(/^./, (letter) => letter.toUpperCase()) ??
  "Ready";
const active = (run) => run && ["queued", "running"].includes(run.status);
const api = new ApiClient();
const experimentGate = new SelectionGate(),
  runGate = new SelectionGate(),
  inspectionGate = new SelectionGate();
const state = {
  defaults: null,
  experiments: [],
  experiment: null,
  runs: [],
  run: null,
  history: [],
  generation: null,
  population: [],
  replay: null,
  connected: false,
  busy: false,
  time: 0,
  playing: false,
  lastTick: null,
};
let pollTimer, animation, toastTimer;

const fitnessChart = new Chart(byId("fitness-chart"), {
  select: inspectGeneration,
});
const completionChart = new Chart(byId("completion-chart"), {
  completion: true,
  select: inspectGeneration,
});
const arena = new Arena(byId("arena"), {
  select: (index) => inspectAgent(index),
  hover: (agent, event) => {
    const tooltip = byId("arena-tooltip");
    tooltip.hidden = !agent;
    if (!agent) return;
    const bounds = byId("arena").getBoundingClientRect();
    tooltip.textContent = `Agent ${agent.index + 1} · ${numeric(agent.fitness)} · ${labelStatus(agent.final_state.status)}`;
    tooltip.style.left = `${Math.max(8, Math.min(event.clientX - bounds.left + 12, bounds.width - 190))}px`;
    tooltip.style.top = `${Math.max(8, event.clientY - bounds.top - 38)}px`;
  },
});

function remember(key, value) {
  try {
    localStorage.setItem(`lab.${key}`, value ?? "");
  } catch {
    /* Storage is optional. */
  }
}
function remembered(key) {
  try {
    return localStorage.getItem(`lab.${key}`);
  } catch {
    return null;
  }
}
function notify(message) {
  clearTimeout(toastTimer);
  text("toast", message);
  byId("toast").hidden = false;
  toastTimer = setTimeout(() => {
    byId("toast").hidden = true;
  }, 7000);
}

function connection(connected, message = "") {
  state.connected = connected;
  byId("connection-dot").className =
    `status-dot ${connected ? "connected" : "offline"}`;
  text(
    "connection-label",
    connected ? "Backend connected" : "Connection interrupted",
  );
  byId("connection-banner").hidden = connected;
  text(
    "connection-message",
    message ||
      "The API or database is unavailable. Check the server and migrations.",
  );
  text(
    "workspace-footer-state",
    connected
      ? "Saved results · connected to FastAPI"
      : "Waiting for the backend",
  );
  renderRun();
}

function chooseOptions(select, options, value, empty) {
  select.replaceChildren();
  if (!options.length) select.add(new Option(empty, ""));
  for (const [label, key] of options) select.add(new Option(label, key));
  select.value = value ?? "";
  select.disabled = options.length === 0;
}

function renderExperiments() {
  chooseOptions(
    byId("experiment-select"),
    state.experiments.map((item) => [item.name, item.id]),
    state.experiment?.id,
    "Choose an experiment",
  );
  text(
    "experiment-count",
    `${state.experiments.length} experiment${state.experiments.length === 1 ? "" : "s"}`,
  );
  const library = byId("experiment-library");
  if (!state.experiments.length) {
    library.replaceChildren(
      element(
        "div",
        "library-empty",
        "No experiments yet. Create your first one to begin.",
      ),
    );
    return;
  }
  const table = element("table", "experiment-table");
  const head = table.createTHead().insertRow();
  for (const label of [
    "EXPERIMENT",
    "POPULATION",
    "GENERATIONS",
    "SEED",
    "CREATED",
    "",
  ]) {
    const cell = element("th", "", label);
    cell.scope = "col";
    head.append(cell);
  }
  const body = table.createTBody();
  for (const item of state.experiments) {
    const row = body.insertRow();
    const name = row.insertCell();
    name.append(
      element("span", "", item.name),
      element("small", "", item.description || "Slalom obstacle course"),
    );
    for (const value of [
      item.settings.evolution.population_size,
      item.settings.generations,
      item.settings.seed,
      new Date(item.created_at).toLocaleDateString("en-GB"),
    ])
      row.insertCell().textContent = value;
    const button = element("button", "button small", "Open ↗");
    button.setAttribute("aria-label", `Open ${item.name}`);
    button.addEventListener("click", () => {
      location.hash = "simulation";
      selectExperiment(item.id);
    });
    row.insertCell().append(button);
  }
  library.replaceChildren(table);
}

function renderRuns() {
  chooseOptions(
    byId("run-select"),
    state.runs.map((item) => [
      `${labelStatus(item.status)} · ${new Date(item.created_at).toLocaleString("en-GB", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit", second: "2-digit" })} · ${item.id.slice(0, 6)}`,
      item.id,
    ]),
    state.run?.id,
    "No runs yet",
  );
}

function renderRun() {
  const run = state.run;
  const badge = byId("run-status");
  badge.className = `badge ${run?.status ?? ""}`;
  badge.textContent =
    run?.cancel_requested && active(run)
      ? "Cancelling"
      : labelStatus(run?.status);
  byId("start-run").disabled =
    !state.experiment || !state.connected || state.busy || active(run);
  byId("cancel-run").hidden = !active(run);
  byId("cancel-run").disabled =
    !state.connected || state.busy || Boolean(run?.cancel_requested);
  let message = state.experiment
    ? "Settings are saved. Start a run to evolve a population."
    : "Create an experiment to begin. Every run keeps its settings and results.";
  if (run?.status === "queued")
    message =
      "Queued · waiting for an experiment worker. Start the worker if this status persists.";
  if (run?.status === "running")
    message = `Training · ${run.completed_generations} of ${run.requested_generations} generations saved. Charts update as each generation finishes.`;
  if (run?.cancel_requested && active(run))
    message =
      "Cancellation requested · the worker will stop after the current generation.";
  if (run?.status === "completed")
    message = `Run complete · ${run.completed_generations} generations saved. Best ever ${numeric(run.champion_fitness)} in generation ${run.champion_generation}.`;
  if (run?.status === "cancelled")
    message = `Run cancelled · ${run.completed_generations} saved generations are still available.`;
  if (run?.status === "failed")
    message =
      run.error ||
      "Run failed. Saved generations are still available; inspect the worker logs.";
  text("run-message", message);
  const settings =
    run?.settings ?? state.experiment?.settings ?? state.defaults;
  if (!settings) return;
  const layers = settings.network.layer_sizes;
  text("network-inputs", layers[0]);
  text("network-hidden", layers.slice(1, -1).join(" · ") || "—");
  text("network-outputs", layers.at(-1));
  const entries = [
    ["Population", settings.evolution.population_size],
    ["Generations", settings.generations],
    ["Mutation rate", `${numeric(settings.evolution.mutation_rate * 100, 0)}%`],
    ["Mutation strength", settings.evolution.mutation_scale],
    [
      "Elites / tournament",
      `${settings.evolution.elite_count} / ${settings.evolution.tournament_size}`,
    ],
    ["Crossover", `${numeric(settings.evolution.crossover_rate * 100, 0)}%`],
    ["Random seed", settings.seed],
    [
      "Episode limit",
      `${numeric(settings.agent.max_steps * settings.agent.timestep, 0)}s`,
    ],
  ];
  byId("configuration-readouts").replaceChildren(
    ...entries.map(([key, value]) => {
      const pair = element("div");
      pair.append(element("dt", "", key), element("dd", "", value));
      return pair;
    }),
  );
}

function renderMetrics() {
  const row = state.history.find(
    (item) => item.generation === state.generation,
  );
  text(
    "metric-generation",
    row ? String(row.generation).padStart(3, "0") : "—",
  );
  text(
    "metric-generation-context",
    row
      ? `${state.history.length} / ${state.run.requested_generations} generations saved`
      : "No saved generations",
  );
  text("metric-best", numeric(row?.best_fitness));
  text("metric-mean", numeric(row?.mean_fitness));
  text(
    "metric-best-context",
    row ? `Generation ${row.generation} champion` : "Strongest controller",
  );
  text(
    "metric-completed",
    row ? `${row.completed_count} / ${row.population_size}` : "—",
  );
  text(
    "metric-completion-context",
    row
      ? `${numeric(row.completion_rate * 100)}% reached the goal`
      : "Successful attempts",
  );
  chooseOptions(
    byId("generation-select"),
    state.history.map((item) => [`${item.generation}`, item.generation]),
    state.generation,
    "—",
  );
  byId("export-csv").disabled = !state.history.length;
  byId("export-png").disabled = !state.history.length;
  fitnessChart.update(state.history, state.generation);
  completionChart.update(state.history, state.generation);
}

function renderFrame() {
  const replay = state.replay;
  const index = replay ? frameIndex(replay.frames, state.time) : -1;
  const frame = index >= 0 ? replay.frames[index] : null;
  const course =
    replay?.course ??
    state.run?.settings.course ??
    state.experiment?.settings.course ??
    state.defaults?.course;
  arena.update({
    course,
    replay,
    frame,
    frameIndex: index,
    population: state.population,
    sensors: byId("show-sensors").checked,
    trail: byId("show-trail").checked,
  });
  text(
    "arena-caption",
    course
      ? `${numeric(course.width, 0)} × ${numeric(course.height, 0)} units`
      : "Obstacle course",
  );
  byId("arena-empty").hidden = Boolean(replay);
  byId("play-replay").disabled = !replay;
  text("play-replay", state.playing ? "Ⅱ Pause replay" : "▶ Play replay");
  byId("replay-scrubber").disabled = !replay;
  byId("replay-scrubber").value = replay?.elapsed_seconds
    ? Math.round((state.time / replay.elapsed_seconds) * 1000)
    : 0;
  text(
    "replay-time",
    `${state.time.toFixed(2)}s / ${(replay?.elapsed_seconds ?? 0).toFixed(2)}s`,
  );
  text(
    "replay-caption",
    replay
      ? `${replay.sampled ? "Sampled replay" : "Recorded attempt"} · generation ${state.generation} · ${replay.agent.index == null ? "champion" : `agent ${replay.agent.index + 1}`}`
      : "Recorded attempts · playback does not change training",
  );
  text("replay-outcome", replay ? labelStatus(replay.final_state.status) : "");
  byId("download-controller").disabled = !replay;
  const badge = byId("agent-status");
  badge.className = `badge ${frame?.state.status ?? ""}`;
  badge.textContent = frame
    ? labelStatus(frame.state.status)
    : "Awaiting replay";
  text(
    "agent-fitness",
    replay ? `${numeric(replay.fitness.total)} fitness` : "—",
  );
  text(
    "agent-position",
    frame
      ? `${numeric(frame.state.position[0], 0)}, ${numeric(frame.state.position[1], 0)}`
      : "—",
  );
  text("agent-speed", frame ? `${numeric(frame.state.speed, 0)} units/s` : "—");
  text(
    "agent-heading",
    frame ? `${numeric((frame.state.heading * 180) / Math.PI, 0)}°` : "—",
  );
  text(
    "agent-checkpoints",
    frame
      ? `${frame.state.checkpoints_passed} / ${course.checkpoints.length}`
      : "—",
  );
  const rayCount =
    replay?.sensor_config.ray_angles.length ??
    state.defaults?.sensors.ray_angles.length ??
    7;
  const bars = byId("sensor-bars");
  if (bars.children.length !== rayCount)
    bars.replaceChildren(
      ...Array.from({ length: rayCount }, () => {
        const bar = element("div", "sensor-bar");
        bar.append(element("span"));
        return bar;
      }),
    );
  [...bars.children].forEach((bar, ray) => {
    const distance = frame?.inputs[ray] ?? 0;
    bar.firstChild.style.height = `${distance * 100}%`;
    bar.title = frame
      ? `${numeric((replay.sensor_config.ray_angles[ray] * 180) / Math.PI, 0)}°: ${numeric(distance * replay.sensor_config.max_distance, 0)} units`
      : "No observation yet";
    bar.setAttribute("aria-label", bar.title);
  });
  const outputs = frame?.outputs;
  text(
    "steering-output",
    outputs ? numeric(Math.max(-1, Math.min(1, outputs[0])), 2) : "—",
  );
  text(
    "throttle-output",
    outputs
      ? `${numeric((Math.max(-1, Math.min(1, outputs[1])) + 1) * 50, 0)}%`
      : "—",
  );
}

function stopPlayback() {
  state.playing = false;
  state.lastTick = null;
  cancelAnimationFrame(animation);
}
function clearInspection() {
  inspectionGate.next();
  stopPlayback();
  state.replay = null;
  state.population = [];
  state.time = 0;
  byId("arena-empty").querySelector("strong").textContent =
    "No replay selected";
  byId("arena-empty").querySelector("span:last-child").textContent =
    "Save a generation to explore its agents.";
  chooseOptions(byId("agent-select"), [], null, "Generation champion");
  renderFrame();
}

async function selectExperiment(id, preferredRun) {
  const task = experimentGate.next();
  state.experiment = state.experiments.find((item) => item.id === id) ?? null;
  state.runs = [];
  selectRun(null);
  renderExperiments();
  renderRuns();
  renderRun();
  remember("experiment", id);
  if (!state.experiment) return;
  try {
    const runs = await api.pages(
      `/api/v1/runs?experiment_id=${encodeURIComponent(id)}`,
      { signal: task.signal },
    );
    if (!task.current()) return;
    state.runs = runs;
    const selection =
      runs.find((item) => item.id === (preferredRun ?? remembered("run"))) ??
      runs[0];
    renderRuns();
    await selectRun(selection?.id ?? null);
  } catch (error) {
    if (task.current()) connection(false, error.message);
  }
}

async function selectRun(id) {
  clearTimeout(pollTimer);
  const task = runGate.next();
  state.run = state.runs.find((item) => item.id === id) ?? null;
  state.history = [];
  state.generation = null;
  byId("follow-latest").checked = true;
  clearInspection();
  renderRuns();
  renderRun();
  renderMetrics();
  if (!state.run) return;
  remember("run", id);
  await pollRun(id, task);
}

async function pollRun(id, task) {
  if (!task.current()) return;
  try {
    const [run, history] = await Promise.all([
      api.request(`/api/v1/runs/${id}`, { signal: task.signal }),
      api.pages(`/api/v1/runs/${id}/generations`, {
        signal: task.signal,
        limit: 1000,
      }),
    ]);
    if (!task.current()) return;
    const changed = history.length !== state.history.length;
    state.run = run;
    state.history = history;
    state.runs = state.runs.map((item) => (item.id === id ? run : item));
    connection(true);
    renderRuns();
    if (
      history.length &&
      (state.generation === null || (changed && byId("follow-latest").checked))
    )
      await inspectGeneration(history.at(-1).generation, true);
    else if (changed) renderMetrics();
  } catch (error) {
    if (task.current()) connection(false, error.message);
  } finally {
    if (task.current())
      pollTimer = setTimeout(
        () => pollRun(id, task),
        active(state.run) ? 1500 : 8000,
      );
  }
}

async function inspectGeneration(generation, following = false) {
  if (
    !state.run ||
    !state.history.some((item) => item.generation === generation)
  )
    return;
  if (!following) byId("follow-latest").checked = false;
  state.generation = generation;
  clearInspection();
  renderMetrics();
  const task = inspectionGate.next();
  const id = state.run.id;
  byId("arena-empty").querySelector("strong").textContent =
    "Loading saved population…";
  byId("arena-empty").querySelector("span:last-child").textContent =
    "Preparing the generation champion’s replay.";
  try {
    const population = await api.request(
      `/api/v1/runs/${id}/generations/${generation}/population`,
      { signal: task.signal },
    );
    if (!task.current()) return;
    state.population = population.agents;
    chooseOptions(
      byId("agent-select"),
      [
        ["Generation champion", "champion"],
        ...population.agents.map((agent) => [
          `Agent ${agent.index + 1} · ${numeric(agent.fitness)}${agent.index === population.champion_index ? " ★" : ""}`,
          agent.index,
        ]),
      ],
      "champion",
      "Generation champion",
    );
    await loadReplay(task, id, generation);
  } catch (error) {
    if (task.current()) inspectionError(error);
  }
}

async function inspectAgent(index) {
  if (state.generation == null) return;
  const task = inspectionGate.next();
  stopPlayback();
  state.replay = null;
  state.time = 0;
  byId("agent-select").value = index == null ? "champion" : String(index);
  renderFrame();
  try {
    await loadReplay(task, state.run.id, state.generation, index);
  } catch (error) {
    if (task.current()) inspectionError(error);
  }
}

async function loadReplay(task, id, generation, agent) {
  const suffix = agent == null ? "" : `?agent=${agent}`;
  const replay = await api.request(
    `/api/v1/runs/${id}/generations/${generation}/replay${suffix}`,
    { signal: task.signal },
  );
  if (!task.current()) return;
  state.replay = replay;
  state.time = 0;
  renderFrame();
}
function inspectionError(error) {
  byId("arena-empty").querySelector("strong").textContent =
    "Replay unavailable";
  byId("arena-empty").querySelector("span:last-child").textContent =
    error.message;
  notify(error.message);
}

async function queueRun() {
  if (!state.experiment || state.busy) return;
  state.busy = true;
  renderRun();
  const experimentId = state.experiment.id;
  try {
    const run = await api.request(`/api/v1/experiments/${experimentId}/runs`, {
      method: "POST",
      body: {},
    });
    if (state.experiment?.id === experimentId) {
      state.runs.unshift(run);
      renderRuns();
      await selectRun(run.id);
    }
    notify("Run queued. The worker will evaluate the population.");
  } catch (error) {
    notify(error.message);
  } finally {
    state.busy = false;
    renderRun();
  }
}
async function cancelRun() {
  const id = state.run?.id;
  if (!id || state.busy) return;
  state.busy = true;
  renderRun();
  try {
    const run = await api.request(`/api/v1/runs/${id}/cancel`, {
      method: "POST",
    });
    if (state.run?.id === id) {
      state.run = run;
      state.runs = state.runs.map((item) => (item.id === id ? run : item));
      renderRuns();
    }
    notify("Cancellation requested. Saved generations are retained.");
  } catch (error) {
    notify(error.message);
  } finally {
    state.busy = false;
    renderRun();
  }
}

function animate(timestamp) {
  if (!state.playing || !state.replay) return;
  const delta =
    state.lastTick == null
      ? 0
      : Math.min(0.1, (timestamp - state.lastTick) / 1000);
  state.lastTick = timestamp;
  Object.assign(
    state,
    advancePlayback(
      state.time,
      delta,
      Number(byId("playback-speed").value),
      state.replay.elapsed_seconds,
    ),
  );
  renderFrame();
  if (state.playing) animation = requestAnimationFrame(animate);
}
function download(blob, filename) {
  const url = URL.createObjectURL(blob);
  const anchor = element("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function navigate() {
  const experiments = location.hash === "#experiments";
  byId("view-simulation").hidden = experiments;
  byId("view-experiments").hidden = !experiments;
  for (const view of ["simulation", "experiments"]) {
    const selected = view === (experiments ? "experiments" : "simulation");
    byId(`nav-${view}`).classList.toggle("active", selected);
    if (selected) byId(`nav-${view}`).setAttribute("aria-current", "page");
    else byId(`nav-${view}`).removeAttribute("aria-current");
  }
  text("breadcrumb-view", experiments ? "EXPERIMENTS" : "SIMULATION");
}
function openDialog() {
  if (!state.defaults) {
    notify("Connect to the API before creating an experiment.");
    return;
  }
  byId("form-error").hidden = true;
  byId("experiment-dialog").showModal();
}
async function saveExperiment(event) {
  event.preventDefault();
  const form = byId("experiment-form"),
    value = (id) => Number(byId(id).value),
    settings = structuredClone(state.defaults);
  settings.seed = value("new-seed");
  settings.generations = value("new-generations");
  Object.assign(settings.evolution, {
    population_size: value("new-population"),
    mutation_rate: value("new-mutation-rate") / 100,
    mutation_scale: value("new-mutation-scale"),
    elite_count: value("new-elites"),
    tournament_size: value("new-tournament"),
  });
  settings.network.layer_sizes = [
    settings.sensors.ray_angles.length + 3,
    value("new-hidden"),
    2,
  ];
  settings.agent.max_steps = Math.round(
    value("new-duration") / settings.agent.timestep,
  );
  byId("form-error").hidden = true;
  if (
    settings.evolution.elite_count > settings.evolution.population_size ||
    settings.evolution.tournament_size > settings.evolution.population_size
  ) {
    text(
      "form-error",
      "Elites and tournament size cannot exceed the population. Check Advanced settings.",
    );
    byId("form-error").hidden = false;
    return;
  }
  const intent = event.submitter?.value;
  const buttons = [...form.querySelectorAll("button[type=submit]")];
  buttons.forEach((button) => {
    button.disabled = true;
  });
  let experiment;
  try {
    experiment = await api.request("/api/v1/experiments", {
      method: "POST",
      body: {
        name: byId("experiment-name").value.trim(),
        description: byId("new-description").value.trim(),
        settings,
      },
    });
    state.experiments.unshift(experiment);
    byId("experiment-dialog").close();
    location.hash = "simulation";
    await selectExperiment(experiment.id);
  } catch (error) {
    text("form-error", error.message);
    byId("form-error").hidden = false;
  } finally {
    buttons.forEach((button) => {
      button.disabled = false;
    });
  }
  if (experiment) {
    if (intent === "run") await queueRun();
    else notify("Experiment saved. Start a run when you’re ready.");
  }
}
async function bootstrap() {
  try {
    const [defaults, experiments] = await Promise.all([
      api.request("/api/v1/defaults"),
      api.pages("/api/v1/experiments"),
      api.request("/ready"),
    ]);
    state.defaults = defaults;
    state.experiments = experiments;
    connection(true);
    renderExperiments();
    renderFrame();
    const selection =
      experiments.find((item) => item.id === remembered("experiment")) ??
      experiments[0];
    await selectExperiment(selection?.id ?? null);
  } catch (error) {
    connection(false, error.message);
  }
}

byId("experiment-select").addEventListener("change", (event) =>
  selectExperiment(event.target.value),
);
byId("run-select").addEventListener("change", (event) =>
  selectRun(event.target.value),
);
byId("generation-select").addEventListener("change", (event) =>
  inspectGeneration(Number(event.target.value)),
);
byId("agent-select").addEventListener("change", (event) =>
  inspectAgent(
    event.target.value === "champion" ? undefined : Number(event.target.value),
  ),
);
byId("follow-latest").addEventListener("change", () => {
  if (byId("follow-latest").checked && state.history.length)
    inspectGeneration(state.history.at(-1).generation, true);
});
byId("start-run").addEventListener("click", queueRun);
byId("cancel-run").addEventListener("click", cancelRun);
byId("retry-connection").addEventListener("click", () => location.reload());
byId("show-sensors").addEventListener("change", renderFrame);
byId("show-trail").addEventListener("change", renderFrame);
byId("play-replay").addEventListener("click", () => {
  if (!state.replay) return;
  if (state.playing) stopPlayback();
  else {
    if (state.time >= state.replay.elapsed_seconds) state.time = 0;
    state.playing = true;
    state.lastTick = null;
    animation = requestAnimationFrame(animate);
  }
  renderFrame();
});
byId("replay-scrubber").addEventListener("input", (event) => {
  stopPlayback();
  state.time =
    (Number(event.target.value) / 1000) * (state.replay?.elapsed_seconds ?? 0);
  renderFrame();
});
byId("download-controller").addEventListener("click", () => {
  if (state.replay)
    download(
      new Blob([JSON.stringify(state.replay.controller, null, 2)], {
        type: "application/json",
      }),
      `controller-${state.run.id}-g${state.generation}.json`,
    );
});
byId("export-csv").addEventListener("click", () =>
  download(
    new Blob([fitnessCsv(state.history)], { type: "text/csv;charset=utf-8" }),
    `fitness-${state.run.id}.csv`,
  ),
);
byId("export-png").addEventListener("click", async () => {
  const id = state.run?.id;
  if (!id) return;
  byId("export-png").disabled = true;
  try {
    download(
      await api.request(`/api/v1/runs/${id}/fitness.png`, { blob: true }),
      `fitness-${id}.png`,
    );
  } catch (error) {
    notify(error.message);
  } finally {
    byId("export-png").disabled = !state.history.length;
  }
});
document
  .querySelectorAll("[data-new-experiment]")
  .forEach((button) => button.addEventListener("click", openDialog));
byId("close-dialog").addEventListener("click", () =>
  byId("experiment-dialog").close(),
);
byId("experiment-form").addEventListener("submit", saveExperiment);
window.addEventListener("hashchange", navigate);
document.addEventListener("visibilitychange", () => {
  state.lastTick = null;
});
navigate();
bootstrap();
