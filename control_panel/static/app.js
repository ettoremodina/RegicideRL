"use strict";

const token = document.querySelector('meta[name="control-token"]').content;
const state = {
  bootstrap: null,
  overview: null,
  jobs: [],
  runs: [],
  view: "overview",
  jobCategory: "All",
  selectedCommand: null,
  selectedRun: null,
  selectedConfig: null,
  configHash: null,
  browserScope: "artifacts",
  browserPath: "",
  logJobId: null,
  refreshBusy: false,
};

const pageTitles = {
  overview: "Overview",
  jobs: "Jobs",
  runs: "Runs & monitor",
  config: "Configuration",
  artifacts: "Files & artifacts",
  repository: "Repository",
};

document.addEventListener("DOMContentLoaded", initialize);

async function initialize() {
  bindStaticEvents();
  try {
    state.bootstrap = await api("/api/bootstrap");
    state.overview = state.bootstrap.overview;
    state.jobs = state.bootstrap.jobs;
    state.runs = state.bootstrap.runs;
    renderNavigationData();
    renderAll();
    await loadConfig(state.bootstrap.configurations[0].id);
    await loadBrowser("artifacts", "");
    setConnected(true);
  } catch (error) {
    setConnected(false);
    toast("Could not start the panel", error.message, "error");
  }
  window.setInterval(refreshCurrentState, 4000);
}

function bindStaticEvents() {
  document.querySelectorAll(".nav-item").forEach((button) => {
    button.addEventListener("click", () => {
      const category = button.dataset.category || null;
      setView(button.dataset.view, category);
    });
  });
  document.querySelectorAll("[data-go]").forEach((button) => {
    button.addEventListener("click", () => setView(button.dataset.go));
  });
  document.getElementById("refresh-button").addEventListener("click", refreshCurrentState);
  document.getElementById("command-search").addEventListener("input", renderCommands);
  document.getElementById("command-form").addEventListener("submit", submitCommand);
  document.getElementById("jobs-table").addEventListener("click", handleJobAction);
  document.getElementById("overview-jobs").addEventListener("click", handleJobAction);
  document.getElementById("runs-table").addEventListener("click", handleRunSelection);
  document.getElementById("overview-runs").addEventListener("click", handleRunSelection);
  document.getElementById("run-filter-button").addEventListener("click", filterRuns);
  document.getElementById("run-search").addEventListener("keydown", (event) => {
    if (event.key === "Enter") filterRuns();
  });
  document.getElementById("config-reload-button").addEventListener("click", () => {
    if (state.selectedConfig) loadConfig(state.selectedConfig);
  });
  document.getElementById("config-preview-button").addEventListener("click", previewConfig);
  document.getElementById("config-save-button").addEventListener("click", saveConfig);
  document.getElementById("log-close-button").addEventListener("click", closeJobLog);
  document.getElementById("refresh-log-button").addEventListener("click", refreshJobLog);
  document.getElementById("stop-job-button").addEventListener("click", stopSelectedJob);
  document.getElementById("shutdown-button").addEventListener("click", shutdownPanel);
}

async function api(path, options = {}) {
  const headers = { "X-Control-Token": token, ...(options.headers || {}) };
  if (options.body && !headers["Content-Type"]) headers["Content-Type"] = "application/json";
  const response = await fetch(path, { ...options, headers, cache: "no-store" });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `${response.status} ${response.statusText}`);
  return payload;
}

function setView(view, category = null) {
  state.view = view;
  if (view === "jobs") state.jobCategory = category || "All";
  document.querySelectorAll(".view").forEach((section) => section.classList.remove("active"));
  document.getElementById(`view-${view}`).classList.add("active");
  document.querySelectorAll(".nav-item").forEach((button) => {
    const matchesView = button.dataset.view === view;
    const matchesCategory = !button.dataset.category || button.dataset.category === category;
    button.classList.toggle("active", matchesView && matchesCategory);
  });
  document.getElementById("page-title").textContent = category || pageTitles[view];
  window.location.hash = category ? `${view}:${category}` : view;
  if (view === "jobs") renderCommands();
  if (view === "repository") renderRepository();
}

function renderNavigationData() {
  const categories = ["All", ...new Set(state.bootstrap.commands.map((item) => item.category))];
  const filter = document.getElementById("job-category-filter");
  filter.innerHTML = categories.map((category) => (
    `<button data-category="${escapeHtml(category)}">${escapeHtml(category)}</button>`
  )).join("");
  filter.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-category]");
    if (!button) return;
    state.jobCategory = button.dataset.category;
    renderCommands();
  });

  const scopeFilter = document.getElementById("scope-filter");
  scopeFilter.innerHTML = state.bootstrap.scopes.map((scope) => (
    `<button data-scope="${scope.id}">${escapeHtml(scope.label)}</button>`
  )).join("");
  scopeFilter.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-scope]");
    if (button) loadBrowser(button.dataset.scope, "");
  });

  renderConfigList();
}

function renderAll() {
  renderOverview();
  renderCommands();
  renderJobs();
  renderRuns();
  renderRepository();
}

async function refreshCurrentState() {
  if (state.refreshBusy || document.hidden) return;
  state.refreshBusy = true;
  try {
    const [overview, jobs, runs] = await Promise.all([
      api("/api/overview"),
      api("/api/jobs?limit=100"),
      api("/api/runs?limit=100"),
    ]);
    state.overview = overview;
    state.jobs = jobs.jobs;
    state.runs = runs.runs;
    renderOverview();
    renderJobs();
    renderRuns();
    renderRepository();
    if (state.logJobId && document.getElementById("log-dialog").open) await refreshJobLog();
    if (state.selectedRun && state.view === "runs") await loadRun(state.selectedRun, false);
    setConnected(true);
  } catch (error) {
    setConnected(false);
  } finally {
    state.refreshBusy = false;
  }
}

function renderOverview() {
  if (!state.overview) return;
  const { catalog, inventory, system, git, health, jobs } = state.overview;
  document.getElementById("overview-metrics").innerHTML = [
    metricCard("Catalog runs", catalog.total_runs, `${catalog.statuses.running || 0} logger-running`),
    metricCard("Panel jobs", jobs.active, `${state.jobs.length} retained in history`),
    metricCard("Repository files", formatNumber(inventory.source_files), `${formatBytes(inventory.source_bytes)} source/docs`),
    metricCard("Artifacts", formatBytes(inventory.artifact_bytes), `${formatNumber(inventory.artifact_files)} files`),
  ].join("");

  const quick = state.bootstrap.commands.filter((command) => command.quick_action).slice(0, 4);
  document.getElementById("quick-actions").innerHTML = quick.map((command, index) => (
    `<button class="${index === 0 ? "primary-button" : "secondary-button"}" data-command="${command.id}">${escapeHtml(command.title)}</button>`
  )).join("");
  document.getElementById("quick-actions").querySelectorAll("[data-command]").forEach((button) => {
    button.addEventListener("click", () => openCommand(button.dataset.command));
  });

  document.getElementById("overview-runs").innerHTML = runsTable(state.runs.slice(0, 7), true);
  document.getElementById("system-resources").innerHTML = [
    resourceRow("CPU", system.cpu_percent, `${system.cpu_percent.toFixed(1)}% · ${system.logical_cpus || "?"} threads`),
    resourceRow("Memory", system.memory_percent, `${formatBytes(system.memory_used)} / ${formatBytes(system.memory_total)}`),
    resourceRow("Disk", system.disk_percent, `${formatBytes(system.disk_used)} / ${formatBytes(system.disk_total)}`),
  ].join("");
  document.getElementById("health-list").innerHTML = health.map((item) => (
    `<div class="health-item" data-level="${item.level}"><i class="health-dot"></i><div><strong>${escapeHtml(item.title)}</strong><p>${escapeHtml(item.message)}</p></div></div>`
  )).join("");
  document.getElementById("overview-jobs").innerHTML = jobsTable(state.jobs.slice(0, 6), true);
  document.title = `${jobs.active ? `(${jobs.active}) ` : ""}Regicide Control Panel`;
  void git;
}

function metricCard(label, value, note) {
  return `<article class="metric-card"><span class="metric-label">${escapeHtml(label)}</span><strong>${escapeHtml(String(value))}</strong><small>${escapeHtml(note)}</small></article>`;
}

function resourceRow(label, percentage, detail) {
  const width = Math.max(0, Math.min(100, percentage || 0));
  return `<div class="resource-row"><header><span>${escapeHtml(label)}</span><span>${escapeHtml(detail)}</span></header><div class="meter"><span style="width:${width}%"></span></div></div>`;
}

function renderCommands() {
  if (!state.bootstrap) return;
  const query = document.getElementById("command-search").value.trim().toLowerCase();
  const commands = state.bootstrap.commands.filter((command) => {
    const categoryMatches = state.jobCategory === "All" || command.category === state.jobCategory;
    const text = `${command.title} ${command.description} ${command.tags.join(" ")}`.toLowerCase();
    return categoryMatches && (!query || text.includes(query));
  });
  document.querySelectorAll("#job-category-filter button").forEach((button) => {
    button.classList.toggle("active", button.dataset.category === state.jobCategory);
  });
  document.getElementById("command-grid").innerHTML = commands.length ? commands.map(commandCard).join("") : emptyState("⌕", "No workflows found", "Try another category or search term.");
  document.querySelectorAll("#command-grid [data-command]").forEach((button) => {
    button.addEventListener("click", () => openCommand(button.dataset.command));
  });
}

function commandCard(command) {
  const tags = command.tags.slice(0, 2).map((tagName) => `<span class="tag">${escapeHtml(tagName)}</span>`).join("");
  return `<article class="command-card">
    <header><span class="command-symbol">${commandSymbol(command.category)}</span><span class="risk-badge risk-${command.risk}">${escapeHtml(command.risk)}</span></header>
    <h3>${escapeHtml(command.title)}</h3><p>${escapeHtml(command.description)}</p>
    <footer><div class="command-tags">${tags}</div><button class="text-button" data-command="${command.id}">Configure →</button></footer>
  </article>`;
}

function commandSymbol(category) {
  return ({ Play: "♠", Evaluate: "◎", Train: "◆", Analyze: "⌁", Quality: "✓", Maintenance: "⚙" })[category] || "▶";
}

function openCommand(commandId) {
  const command = state.bootstrap.commands.find((item) => item.id === commandId);
  if (!command) return;
  state.selectedCommand = command;
  document.getElementById("command-category").textContent = `${command.category} · ${command.source}`;
  document.getElementById("command-title").textContent = command.title;
  document.getElementById("command-description").textContent = command.description;
  const warning = document.getElementById("command-warning");
  warning.textContent = command.confirmation;
  warning.classList.toggle("hidden", !command.confirmation);
  document.getElementById("command-fields").innerHTML = command.parameters.length
    ? command.parameters.map(commandField).join("")
    : `<div class="field full"><p class="muted">This workflow has no configurable arguments.</p></div>`;
  document.getElementById("command-dialog").showModal();
}

function commandField(field) {
  const required = field.required ? "required" : "";
  const min = field.minimum ?? "";
  const max = field.maximum ?? "";
  const defaultValue = field.default ?? "";
  const help = field.help ? `<small>${escapeHtml(field.help)}</small>` : "";
  let input;
  if (field.kind === "choice") {
    input = `<select name="${field.key}" ${required}>${field.choices.map((choice) => `<option value="${escapeHtml(choice)}" ${choice === defaultValue ? "selected" : ""}>${escapeHtml(choice)}</option>`).join("")}</select>`;
  } else if (field.kind === "multi_choice") {
    const selected = Array.isArray(defaultValue) ? defaultValue : [];
    input = `<div class="multi-choice">${field.choices.map((choice) => `<label><input type="checkbox" name="${field.key}" value="${escapeHtml(choice)}" ${selected.includes(choice) ? "checked" : ""}> ${escapeHtml(choice)}</label>`).join("")}</div>`;
  } else if (field.kind === "boolean") {
    input = `<label class="multi-choice"><input type="checkbox" name="${field.key}" ${defaultValue ? "checked" : ""}> Enabled</label>`;
  } else {
    const type = field.kind === "integer" || field.kind === "number" ? "number" : "text";
    const step = field.kind === "integer" ? "1" : field.kind === "number" ? "any" : "";
    input = `<input type="${type}" name="${field.key}" value="${escapeHtml(String(defaultValue))}" placeholder="${escapeHtml(field.placeholder || "")}" ${required} ${min !== "" ? `min="${min}"` : ""} ${max !== "" ? `max="${max}"` : ""} ${step ? `step="${step}"` : ""}>`;
  }
  return `<div class="field ${field.kind === "multi_choice" ? "full" : ""}"><label>${escapeHtml(field.label)}${field.required ? " *" : ""}</label>${input}${help}</div>`;
}

async function submitCommand(event) {
  event.preventDefault();
  if (event.submitter?.value === "cancel") {
    document.getElementById("command-dialog").close();
    return;
  }
  const command = state.selectedCommand;
  if (!command) return;
  const form = event.currentTarget;
  if (!form.reportValidity()) return;
  const parameters = collectCommandParameters(command, form);
  if (command.confirmation) {
    const accepted = await confirmAction(command.title, command.confirmation, "Start job");
    if (!accepted) return;
  }
  try {
    const job = await api("/api/jobs/start", {
      method: "POST",
      body: JSON.stringify({ command_id: command.id, parameters }),
    });
    document.getElementById("command-dialog").close();
    toast("Job started", `${job.title} · ${job.job_id}`, "success");
    state.jobs.unshift(job);
    setView("jobs");
    await refreshCurrentState();
    openJobLog(job.job_id);
  } catch (error) {
    toast("Could not start job", error.message, "error");
  }
}

function collectCommandParameters(command, form) {
  const values = {};
  command.parameters.forEach((field) => {
    if (field.kind === "multi_choice") {
      values[field.key] = [...form.querySelectorAll(`[name="${field.key}"]:checked`)].map((input) => input.value);
    } else if (field.kind === "boolean") {
      values[field.key] = form.querySelector(`[name="${field.key}"]`).checked;
    } else {
      const value = form.querySelector(`[name="${field.key}"]`).value;
      values[field.key] = value === "" ? null : value;
    }
  });
  return values;
}

function renderJobs() {
  document.getElementById("jobs-table").innerHTML = jobsTable(state.jobs, false);
  document.getElementById("job-count-label").textContent = `${state.jobs.filter((job) => job.active).length} active · ${state.jobs.length} retained`;
}

function jobsTable(jobs, compact) {
  if (!jobs.length) return emptyState("▶", "No panel jobs yet", "Launch any workflow to create durable process history.");
  return `<table><thead><tr><th>Workflow</th><th>Status</th><th>Elapsed</th>${compact ? "" : "<th>PID</th>"}<th></th></tr></thead><tbody>${jobs.map((job) => `
    <tr><td><strong>${escapeHtml(job.title)}</strong><span class="mono truncate">${escapeHtml(job.job_id)}</span></td><td>${statusBadge(job.status)}</td><td>${formatDuration(job.elapsed_seconds)}</td>${compact ? "" : `<td class="mono">${job.pid || "—"}</td>`}<td><button class="text-button" data-job-log="${job.job_id}">Output</button>${job.active ? `<button class="text-button danger-text" data-job-stop="${job.job_id}">Stop</button>` : ""}</td></tr>`).join("")}</tbody></table>`;
}

function handleJobAction(event) {
  const logButton = event.target.closest("[data-job-log]");
  if (logButton) openJobLog(logButton.dataset.jobLog);
  const stopButton = event.target.closest("[data-job-stop]");
  if (stopButton) stopJob(stopButton.dataset.jobStop);
}

async function openJobLog(jobId) {
  state.logJobId = jobId;
  document.getElementById("log-dialog").showModal();
  await refreshJobLog();
}

function closeJobLog() {
  state.logJobId = null;
  document.getElementById("log-dialog").close();
}

async function refreshJobLog() {
  if (!state.logJobId) return;
  try {
    const job = await api(`/api/job-log?id=${encodeURIComponent(state.logJobId)}`);
    document.getElementById("log-title").textContent = job.title;
    document.getElementById("log-status").innerHTML = `${statusBadge(job.status)} · ${formatDuration(job.elapsed_seconds)} · PID ${job.pid || "—"}`;
    const output = document.getElementById("job-log");
    const nearBottom = output.scrollTop + output.clientHeight >= output.scrollHeight - 35;
    output.textContent = job.log || "Waiting for output…";
    if (nearBottom) output.scrollTop = output.scrollHeight;
    document.getElementById("stop-job-button").disabled = !job.active;
  } catch (error) {
    toast("Could not load job output", error.message, "error");
  }
}

async function stopSelectedJob() {
  if (state.logJobId) await stopJob(state.logJobId);
}

async function stopJob(jobId) {
  const accepted = await confirmAction("Stop this job?", "The panel will first request a graceful interrupt. Child jobs are isolated from the panel.", "Request stop");
  if (!accepted) return;
  try {
    let result = await api("/api/jobs/stop", {
      method: "POST",
      body: JSON.stringify({ job_id: jobId, force: false }),
    });
    if (result.requires_force) {
      const force = await confirmAction("Force-stop process tree?", "The job ignored the graceful interrupt. Force-stop may leave the logger run marked running.", "Force stop");
      if (force) {
        result = await api("/api/jobs/stop", {
          method: "POST",
          body: JSON.stringify({ job_id: jobId, force: true }),
        });
      }
    }
    toast("Stop request processed", `${result.title} · ${result.status}`, "success");
    await refreshCurrentState();
  } catch (error) {
    toast("Could not stop job", error.message, "error");
  }
}

function renderRuns() {
  document.getElementById("runs-table").innerHTML = runsTable(state.runs, false);
}

function runsTable(runs, compact) {
  if (!runs.length) return emptyState("◉", "No catalog runs", "Run a workflow that uses ml_logger to populate this view.");
  return `<table><thead><tr><th>Run</th><th>Type</th><th>State</th>${compact ? "" : "<th>Started</th>"}<th>Time</th></tr></thead><tbody>${runs.map((run) => `
    <tr data-clickable="true" data-run-id="${escapeHtml(run.run_id)}"><td><strong class="truncate">${escapeHtml(run.name)}</strong><span class="mono truncate">${escapeHtml(run.run_id)}</span></td><td>${escapeHtml(run.run_type)}</td><td>${statusBadge(run.effective_state)}</td>${compact ? "" : `<td>${formatDate(run.started_at)}</td>`}<td>${formatDuration(run.duration_seconds)}</td></tr>`).join("")}</tbody></table>`;
}

function handleRunSelection(event) {
  const row = event.target.closest("[data-run-id]");
  if (!row) return;
  setView("runs");
  loadRun(row.dataset.runId);
}

async function filterRuns() {
  const params = new URLSearchParams({ limit: "200" });
  const status = document.getElementById("run-status-filter").value;
  const type = document.getElementById("run-type-filter").value.trim();
  const query = document.getElementById("run-search").value.trim();
  if (status) params.set("status", status);
  if (type) params.set("type", type);
  if (query) params.set("q", query);
  try {
    state.runs = (await api(`/api/runs?${params}`)).runs;
    renderRuns();
  } catch (error) {
    toast("Could not filter runs", error.message, "error");
  }
}

async function loadRun(runId, showLoading = true) {
  state.selectedRun = runId;
  const panel = document.getElementById("run-detail");
  if (showLoading) panel.innerHTML = emptyState("◌", "Loading run", "Reading catalog events and artifacts…");
  try {
    const run = await api(`/api/run?id=${encodeURIComponent(runId)}`);
    if (state.selectedRun !== runId) return;
    renderRunDetail(run);
  } catch (error) {
    panel.innerHTML = emptyState("!", "Could not load run", error.message);
  }
}

function renderRunDetail(run) {
  const activePane = document.querySelector("#run-detail .detail-tabs button.active")?.dataset.pane || "summary";
  const metricNames = Object.keys(run.metric_series);
  const charts = metricNames.length ? metricNames.slice(0, 12).map((name) => chartCard(name, run.metric_series[name])).join("") : emptyState("⌁", "No persisted metrics", "This workflow may expose only logs, telemetry, or TensorBoard files.");
  const files = run.files.length ? `<table><thead><tr><th>File</th><th>Size</th><th></th></tr></thead><tbody>${run.files.map((file) => `<tr><td><span class="mono truncate">${escapeHtml(file.run_relative_path)}</span></td><td>${formatBytes(file.size)}</td><td><a class="text-button" target="_blank" rel="noopener" href="${viewUrl("repo", file.path)}">Open</a></td></tr>`).join("")}</tbody></table>` : emptyState("▱", "No run directory", "The catalog path is missing or empty.");
  const games = run.games.length ? `<table><thead><tr><th>Game</th><th>State</th><th>Victory</th><th>Bosses</th><th>Turns</th><th></th></tr></thead><tbody>${run.games.map((game) => `<tr><td class="mono">${escapeHtml(game.game_id)}</td><td>${statusBadge(game.status)}</td><td>${game.victory == null ? "—" : game.victory ? "yes" : "no"}</td><td>${game.bosses_defeated ?? "—"}</td><td>${game.turns ?? "—"}</td><td><button class="text-button" data-game-replay="${escapeHtml(game.game_id)}">Replay</button></td></tr>`).join("")}</tbody></table><div id="game-replay-content"></div>` : emptyState("♠", "No recorded games", "Game recording may be disabled for this run type.");
  const telemetry = run.telemetry.at(-1)?.payload || {};
  const manifest = JSON.stringify(run.manifest, null, 2);
  document.getElementById("run-detail").innerHTML = `
    <div class="run-title-row"><div><p class="eyebrow">${escapeHtml(run.run_type)}</p><h2>${escapeHtml(run.name)}</h2><span class="mono muted">${escapeHtml(run.run_id)}</span></div>${statusBadge(run.effective_state)}</div>
    <div class="detail-tabs"><button class="active" data-pane="summary">Summary</button><button data-pane="metrics">Metrics</button><button data-pane="logs">Logs</button><button data-pane="games">Games (${run.games.length})</button><button data-pane="files">Files (${run.files.length})</button><button data-pane="manifest">Manifest</button></div>
    <div class="detail-pane active" data-pane-content="summary"><div class="detail-grid">
      ${detailKv("Logger status", run.status)}${detailKv("Effective state", run.effective_state)}${detailKv("Started", formatDate(run.started_at))}${detailKv("Duration", formatDuration(run.duration_seconds))}${detailKv("Catalog path", run.path)}${detailKv("Path exists", run.path_exists ? "yes" : "no")}${detailKv("Telemetry PID", run.telemetry_pid || "—")}${detailKv("Process evidence", run.process_alive ? "PID currently exists" : "not alive / unavailable")}
    </div>${telemetry.system ? `<h3 style="margin:20px 0 10px">Latest telemetry</h3><div class="detail-grid">${detailKv("CPU", `${telemetry.system.cpu_percent ?? "—"}%`)}${detailKv("Memory", `${telemetry.system.memory_percent ?? "—"}%`)}${detailKv("Process RSS", `${telemetry.process?.rss_mb ?? "—"} MB`)}${detailKv("GPU", telemetry.gpu?.available ? telemetry.gpu.devices?.[0]?.name || "available" : "unavailable")}</div>` : ""}</div>
    <div class="detail-pane" data-pane-content="metrics"><div class="chart-grid">${charts}</div></div>
    <div class="detail-pane" data-pane-content="logs"><pre class="log-block">${escapeHtml(run.log || "No persisted run.log output.")}</pre></div>
    <div class="detail-pane" data-pane-content="games"><div class="table-wrap">${games}</div></div>
    <div class="detail-pane" data-pane-content="files"><div class="table-wrap">${files}</div></div>
    <div class="detail-pane" data-pane-content="manifest"><pre class="json-block">${escapeHtml(manifest)}</pre></div>`;
  document.querySelectorAll("#run-detail .detail-tabs button").forEach((button) => button.addEventListener("click", () => selectRunPane(button.dataset.pane)));
  document.querySelectorAll("#run-detail [data-game-replay]").forEach((button) => button.addEventListener("click", () => loadGameReplay(button.dataset.gameReplay)));
  selectRunPane(activePane);
}

function selectRunPane(pane) {
  document.querySelectorAll("#run-detail [data-pane]").forEach((button) => button.classList.toggle("active", button.dataset.pane === pane));
  document.querySelectorAll("#run-detail [data-pane-content]").forEach((content) => content.classList.toggle("active", content.dataset.paneContent === pane));
}

async function loadGameReplay(gameId) {
  const container = document.getElementById("game-replay-content");
  if (!container) return;
  container.innerHTML = emptyState("◌", "Loading replay", gameId);
  try {
    const game = await api(`/api/game?id=${encodeURIComponent(gameId)}`);
    const rows = game.events.map((event, index) => {
      const action = event.action || {};
      const cards = Array.isArray(action.cards) ? action.cards.join(", ") : "";
      return `<tr><td>${event.sequence ?? index + 1}</td><td>${action.player ?? "—"}</td><td>${escapeHtml(action.phase || "—")}</td><td>${escapeHtml(action.kind || "event")}</td><td>${escapeHtml(cards)}</td></tr>`;
    }).join("");
    container.innerHTML = `<div class="panel-heading" style="padding-inline:0"><div><p class="eyebrow">RECORDED REPLAY</p><h3>${escapeHtml(game.game_id)}</h3></div>${statusBadge(game.status)}</div><div class="detail-grid">${detailKv("Victory", game.victory == null ? "—" : game.victory ? "yes" : "no")}${detailKv("Bosses defeated", game.bosses_defeated ?? "—")}${detailKv("Turns", game.turns ?? "—")}${detailKv("Events", `${game.events.length}${game.events_truncated ? "+" : ""}`)}</div>${rows ? `<div class="table-wrap" style="margin-top:14px"><table><thead><tr><th>#</th><th>Player</th><th>Phase</th><th>Action</th><th>Cards</th></tr></thead><tbody>${rows}</tbody></table></div>` : `<p class="muted" style="margin-top:14px">This game was recorded at summary level.</p>`}`;
  } catch (error) {
    container.innerHTML = emptyState("!", "Could not load replay", error.message);
  }
}

function detailKv(label, value) {
  return `<div class="detail-kv"><span>${escapeHtml(label)}</span><strong>${escapeHtml(String(value))}</strong></div>`;
}

function chartCard(name, points) {
  const values = points.map((point) => Number(point.value)).filter(Number.isFinite);
  const latest = values.at(-1);
  return `<article class="chart-card"><header><strong title="${escapeHtml(name)}">${escapeHtml(name)}</strong><span>${formatMetric(latest)}</span></header>${sparkline(values)}</article>`;
}

function sparkline(values) {
  if (!values.length) return "";
  const width = 220;
  const height = 64;
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  const range = maximum - minimum || 1;
  const coordinates = values.map((value, index) => {
    const x = values.length === 1 ? width / 2 : (index / (values.length - 1)) * width;
    const y = height - 5 - ((value - minimum) / range) * (height - 10);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  return `<svg class="sparkline" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none"><polyline points="${coordinates}"></polyline></svg>`;
}

function renderConfigList() {
  document.getElementById("config-list").innerHTML = state.bootstrap.configurations.map((config) => `<button class="config-item" data-config="${config.id}"><strong>${escapeHtml(config.title)}</strong><span>${escapeHtml(config.path)}</span></button>`).join("");
  document.querySelectorAll("#config-list [data-config]").forEach((button) => button.addEventListener("click", () => loadConfig(button.dataset.config)));
}

async function loadConfig(configId) {
  try {
    const config = await api(`/api/config?id=${encodeURIComponent(configId)}`);
    state.selectedConfig = configId;
    state.configHash = config.sha256;
    document.querySelectorAll("#config-list [data-config]").forEach((button) => button.classList.toggle("active", button.dataset.config === configId));
    document.getElementById("config-path").textContent = config.path;
    document.getElementById("config-title").textContent = config.title;
    document.getElementById("config-note").textContent = config.restart_note;
    const editor = document.getElementById("config-editor");
    editor.value = config.text;
    editor.disabled = !config.editable;
    document.getElementById("config-preview-button").disabled = !config.editable;
    document.getElementById("config-save-button").disabled = !config.editable;
    document.getElementById("config-validation").classList.remove("visible");
  } catch (error) {
    toast("Could not load configuration", error.message, "error");
  }
}

async function previewConfig() {
  if (!state.selectedConfig) return null;
  try {
    const result = await api("/api/config/preview", {
      method: "POST",
      body: JSON.stringify({
        config_id: state.selectedConfig,
        text: document.getElementById("config-editor").value,
        expected_sha256: state.configHash,
      }),
    });
    renderConfigValidation(result);
    return result;
  } catch (error) {
    toast("Configuration check failed", error.message, "error");
    return null;
  }
}

function renderConfigValidation(result) {
  const panel = document.getElementById("config-validation");
  panel.classList.add("visible");
  const errors = result.errors.map((item) => `<div class="validation-error">Error: ${escapeHtml(item)}</div>`).join("");
  const warnings = result.warnings.map((item) => `<div class="validation-warning">Warning: ${escapeHtml(item)}</div>`).join("");
  panel.innerHTML = `<div class="validation-summary">${result.valid ? statusBadge("completed", "valid YAML") : statusBadge("failed", "invalid")}${result.changed ? `<span class="tag">changed</span>` : `<span class="tag">unchanged</span>`}</div>${errors}${warnings}<pre>${escapeHtml(result.diff)}</pre>`;
}

async function saveConfig() {
  const preview = await previewConfig();
  if (!preview || !preview.valid || !preview.changed) return;
  const accepted = await confirmAction("Save configuration?", "The current file will be backed up under artifacts/control_panel before the atomic save. Active jobs keep their snapshots.", "Save file");
  if (!accepted) return;
  try {
    const result = await api("/api/config/save", {
      method: "POST",
      body: JSON.stringify({
        config_id: state.selectedConfig,
        text: document.getElementById("config-editor").value,
        expected_sha256: state.configHash,
      }),
    });
    state.configHash = result.sha256;
    toast("Configuration saved", result.backup ? `Backup: ${result.backup}` : "No content changed", "success");
    await refreshCurrentState();
  } catch (error) {
    toast("Could not save configuration", error.message, "error");
  }
}

async function loadBrowser(scope, path) {
  try {
    const listing = await api(`/api/browser?scope=${encodeURIComponent(scope)}&path=${encodeURIComponent(path)}`);
    state.browserScope = scope;
    state.browserPath = listing.path;
    renderBrowser(listing);
  } catch (error) {
    toast("Could not browse path", error.message, "error");
  }
}

function renderBrowser(listing) {
  document.querySelectorAll("#scope-filter [data-scope]").forEach((button) => button.classList.toggle("active", button.dataset.scope === listing.scope));
  renderBreadcrumbs(listing.path);
  const container = document.getElementById("browser-list");
  container.innerHTML = listing.entries.length ? listing.entries.map((entry) => `<button class="file-row" data-kind="${entry.kind}" data-path="${escapeHtml(entry.path)}"><span class="file-icon">${entry.kind === "directory" ? "▰" : "▱"}</span><span class="file-name">${escapeHtml(entry.name)}</span><span class="file-meta">${entry.kind === "file" ? formatBytes(entry.size) : "folder"}</span><span class="file-meta">${formatDate(entry.modified_at)}</span></button>`).join("") : emptyState("▱", "Empty directory", listing.path || listing.scope);
  container.querySelectorAll("[data-path]").forEach((button) => button.addEventListener("click", () => {
    if (button.dataset.kind === "directory") loadBrowser(listing.scope, button.dataset.path);
    else previewFile(listing.scope, button.dataset.path);
  }));
}

function renderBreadcrumbs(path) {
  const parts = path ? path.split("/") : [];
  const crumbs = [{ label: state.browserScope, path: "" }];
  parts.forEach((part, index) => crumbs.push({ label: part, path: parts.slice(0, index + 1).join("/") }));
  const container = document.getElementById("browser-breadcrumbs");
  container.innerHTML = crumbs.map((crumb, index) => `${index ? "<span class=\"muted\">/</span>" : ""}<button data-path="${escapeHtml(crumb.path)}">${escapeHtml(crumb.label)}</button>`).join("");
  container.querySelectorAll("button").forEach((button) => button.addEventListener("click", () => loadBrowser(state.browserScope, button.dataset.path)));
}

async function previewFile(scope, path) {
  const preview = document.getElementById("file-preview");
  const name = path.split("/").at(-1);
  const extension = name.includes(".") ? `.${name.split(".").at(-1).toLowerCase()}` : "";
  const openUrl = viewUrl(scope, path);
  const downloadUrl = `${openUrl}&download=1`;
  const heading = `<div class="preview-heading"><div><strong>${escapeHtml(name)}</strong><div class="mono muted">${escapeHtml(path)}</div></div><div><a class="text-button" target="_blank" rel="noopener" href="${openUrl}">Open</a><a class="text-button" href="${downloadUrl}">Download</a></div></div>`;
  if ([".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"].includes(extension)) {
    preview.innerHTML = `${heading}<div class="preview-content"><img src="${openUrl}" alt="${escapeHtml(name)}"></div>`;
    return;
  }
  if ([".pdf", ".html", ".htm"].includes(extension)) {
    preview.innerHTML = `${heading}<div class="preview-content"><iframe sandbox src="${openUrl}" title="${escapeHtml(name)}"></iframe></div>`;
    return;
  }
  try {
    const text = await api(`/api/text?scope=${encodeURIComponent(scope)}&path=${encodeURIComponent(path)}`);
    preview.innerHTML = `${heading}<div class="preview-content"><pre>${escapeHtml(text.content)}</pre></div>`;
  } catch (error) {
    preview.innerHTML = `${heading}${emptyState("▱", "Preview unavailable", error.message)}`;
  }
}

function renderRepository() {
  if (!state.overview) return;
  const { inventory, git } = state.overview;
  document.getElementById("repo-metrics").innerHTML = [
    metricCard("Source & docs", formatNumber(inventory.source_files), formatBytes(inventory.source_bytes)),
    metricCard("Entry points", inventory.entrypoints.length, `${inventory.registered_entrypoints} available as forms`),
    metricCard("Git branch", git.branch, git.commit || "no commit"),
    metricCard("Working changes", git.change_count, git.dirty ? "preserved by the panel" : "clean tree"),
  ].join("");
  document.getElementById("entrypoint-list").innerHTML = inventory.entrypoints.map((entry) => `<div class="entrypoint-row"><span class="mono">${escapeHtml(entry.source)}</span>${entry.registered ? `<span class="status-badge status-completed">registered</span>` : `<span class="tag">discovered / internal</span>`}</div>`).join("");
  document.getElementById("git-panel").innerHTML = `<div class="git-summary"><span class="tag">${escapeHtml(git.branch)}</span><span class="tag mono">${escapeHtml(git.commit)}</span>${git.dirty ? `<span class="risk-badge risk-maintenance">dirty</span>` : statusBadge("completed", "clean")}</div><div class="git-changes">${git.changes.length ? git.changes.map((change) => `<div class="git-change">${escapeHtml(change)}</div>`).join("") : `<span class="muted">No local changes.</span>`}</div>`;
  const areas = Object.entries(inventory.areas);
  const maximum = Math.max(...areas.map(([, count]) => count), 1);
  document.getElementById("area-list").innerHTML = areas.map(([name, count]) => `<div class="area-row"><span>${escapeHtml(name)}</span><div class="area-bar"><span style="width:${(count / maximum) * 100}%"></span></div><strong>${count}</strong></div>`).join("");
}

async function shutdownPanel() {
  const accepted = await confirmAction("Stop the control panel?", "The browser app will close. Running child jobs continue independently and will be reconciled next time.", "Stop panel");
  if (!accepted) return;
  try {
    await api("/api/shutdown", { method: "POST", body: "{}" });
    toast("Control panel stopped", "You may close this browser tab. Child jobs were left running.", "success");
    setConnected(false);
  } catch (error) {
    toast("Could not stop panel", error.message, "error");
  }
}

function confirmAction(title, message, label = "Confirm") {
  const dialog = document.getElementById("confirm-dialog");
  document.getElementById("confirm-title").textContent = title;
  document.getElementById("confirm-message").textContent = message;
  document.getElementById("confirm-accept").textContent = label;
  dialog.returnValue = "cancel";
  dialog.showModal();
  return new Promise((resolve) => {
    dialog.addEventListener("close", () => resolve(dialog.returnValue === "default"), { once: true });
  });
}

function statusBadge(status, label = null) {
  const normalized = String(status || "unknown").toLowerCase().replaceAll("_", "-");
  return `<span class="status-badge status-${escapeHtml(normalized)}">${escapeHtml(label || status || "unknown")}</span>`;
}

function emptyState(symbol, title, message) {
  return `<div class="empty-state"><span>${escapeHtml(symbol)}</span><h3>${escapeHtml(title)}</h3><p>${escapeHtml(message)}</p></div>`;
}

function viewUrl(scope, path) {
  return `/view?token=${encodeURIComponent(token)}&scope=${encodeURIComponent(scope)}&path=${encodeURIComponent(path)}`;
}

function formatBytes(value) {
  const bytes = Number(value || 0);
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let scaled = bytes;
  let unit = -1;
  do { scaled /= 1024; unit += 1; } while (scaled >= 1024 && unit < units.length - 1);
  return `${scaled >= 100 ? scaled.toFixed(0) : scaled.toFixed(1)} ${units[unit]}`;
}

function formatDuration(seconds) {
  const value = Math.max(0, Number(seconds || 0));
  if (value < 60) return `${Math.floor(value)}s`;
  if (value < 3600) return `${Math.floor(value / 60)}m ${Math.floor(value % 60)}s`;
  if (value < 86400) return `${Math.floor(value / 3600)}h ${Math.floor((value % 3600) / 60)}m`;
  return `${Math.floor(value / 86400)}d ${Math.floor((value % 86400) / 3600)}h`;
}

function formatDate(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat(undefined, { month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit" }).format(date);
}

function formatMetric(value) {
  if (!Number.isFinite(value)) return "—";
  if (Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < .001)) return value.toExponential(2);
  return value.toFixed(Math.abs(value) < 10 ? 4 : 2).replace(/0+$/, "").replace(/\.$/, "");
}

function formatNumber(value) {
  return new Intl.NumberFormat().format(Number(value || 0));
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[character]);
}

function toast(title, message, type = "") {
  const item = document.createElement("div");
  item.className = `toast ${type}`;
  item.innerHTML = `<strong>${escapeHtml(title)}</strong><span>${escapeHtml(message)}</span>`;
  document.getElementById("toast-stack").appendChild(item);
  window.setTimeout(() => item.remove(), 5000);
}

function setConnected(connected) {
  const pill = document.querySelector(".connection-pill");
  pill.classList.toggle("offline", !connected);
  document.getElementById("connection-label").textContent = connected ? "Connected" : "Disconnected";
}
