/**
 * Bike Parking Buddy — Dashboard JavaScript
 * Connects to the FastAPI backend at API_BASE and drives all 5 panels.
 */

const API_BASE = "http://localhost:8000";

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

async function apiFetch(path, options = {}) {
  const res = await fetch(API_BASE + path, options);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

function occClass(pct) {
  if (pct < 0.50) return "low";
  if (pct < 0.75) return "mid";
  return "high";
}

function formatPct(pct) {
  return Math.round(pct * 100) + "%";
}

function el(id) { return document.getElementById(id); }

// ─────────────────────────────────────────────────────────────────────────────
// API Health Check
// ─────────────────────────────────────────────────────────────────────────────

async function checkHealth() {
  const dot  = el("status-dot");
  const label = el("status-label");
  try {
    const data = await apiFetch("/health/");
    dot.className = "status-dot online";
    label.textContent = `API online · v${data.version}`;
  } catch {
    dot.className = "status-dot offline";
    label.textContent = "API offline — start uvicorn";
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Navigation
// ─────────────────────────────────────────────────────────────────────────────

const panelTitles = {
  overview: "Overview",
  predict:  "Prediction Lab",
  zones:    "Zone Explorer",
  hnsw:     "HNSW Search",
  slm:      "SLM Tools",
};

function showPanel(name) {
  document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));

  const panel = el(`panel-${name}`);
  const navBtn = el(`nav-${name}`);
  if (panel) panel.classList.add("active");
  if (navBtn) navBtn.classList.add("active");
  el("panel-title").textContent = panelTitles[name] || name;

  // Lazy-load panel data
  if (name === "zones")    loadZones();
  if (name === "slm")      loadToolSchemas();
  if (name === "overview") loadOverview();
}

document.querySelectorAll(".nav-item").forEach(btn => {
  btn.addEventListener("click", e => {
    e.preventDefault();
    showPanel(btn.dataset.panel);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Overview Panel
// ─────────────────────────────────────────────────────────────────────────────

let overviewMap = null;
let overviewMarkers = [];

async function loadOverview() {
  // Latency probe
  const t0 = performance.now();
  let zones = [];
  try {
    zones = await apiFetch("/zones/");
    const latency = Math.round(performance.now() - t0);
    el("kpi-latency-val").textContent = `${latency}ms`;
    el("kpi-zones-val").textContent   = zones.length;
  } catch {
    el("kpi-latency-val").textContent = "—";
  }

  renderOverviewZones(zones);
  initOverviewMap(zones);
}

function renderOverviewZones(zones) {
  const list = el("overview-zone-list");

  // Fetch detail for each zone to get occupancy (mock data returns it)
  Promise.all(
    zones.map(z => apiFetch(`/zones/${z.zone_id}`).catch(() => z))
  ).then(details => {
    list.innerHTML = details.map(z => {
      const occ = z.current_occupancy_pct ?? 0;
      const cls = occClass(occ);
      const fillH = Math.round(occ * 40);
      return `
        <div class="zone-item">
          <div class="zone-fill-bar">
            <div class="zone-fill-bar-inner" style="height:${fillH}px; margin-top:${40-fillH}px;
              background:${occ > 0.75 ? 'var(--accent-red)' : occ > 0.5 ? 'var(--accent-amber)' : 'var(--accent-green)'}"></div>
          </div>
          <div class="zone-info">
            <div class="zone-name">${z.name}</div>
            <div class="zone-meta">${z.venue_type ?? 'unknown'} · transit ${z.transit_score?.toFixed(2) ?? '—'}</div>
          </div>
          <div class="zone-pct pct-${cls}">${formatPct(occ)}</div>
        </div>`;
    }).join("");
  });
}

function initOverviewMap(zones) {
  if (!overviewMap) {
    overviewMap = L.map("overview-map", { zoomControl: true, attributionControl: true });
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "© OpenStreetMap contributors",
    }).addTo(overviewMap);
  }

  // Clear old markers
  overviewMarkers.forEach(m => m.remove());
  overviewMarkers = [];

  if (!zones.length) return;

  zones.forEach(z => {
    const marker = L.circleMarker([z.lat, z.lon], {
      radius: 10,
      fillColor: "#00d4ff",
      fillOpacity: 0.85,
      color: "#fff",
      weight: 2,
    }).addTo(overviewMap);

    marker.bindPopup(`
      <strong>${z.name}</strong><br/>
      Zone: <code>${z.zone_id}</code><br/>
      Venue: ${z.venue_type ?? "—"}<br/>
      Transit score: ${z.transit_score?.toFixed(2) ?? "—"}
    `);
    overviewMarkers.push(marker);
  });

  const bounds = zones.map(z => [z.lat, z.lon]);
  overviewMap.fitBounds(bounds, { padding: [30, 30] });
}

// ─────────────────────────────────────────────────────────────────────────────
// Prediction Lab
// ─────────────────────────────────────────────────────────────────────────────

// Set default timestamp to now
const tsInput = el("p-timestamp");
if (tsInput) {
  const now = new Date();
  now.setMinutes(now.getMinutes() - now.getTimezoneOffset());
  tsInput.value = now.toISOString().slice(0, 16);
}

el("p-lookahead")?.addEventListener("input", () => {
  el("p-lookahead-val").textContent = el("p-lookahead").value + " min";
});

el("predict-form")?.addEventListener("submit", async e => {
  e.preventDefault();
  const btn = el("btn-predict");
  btn.textContent = "Running…";
  btn.disabled = true;

  const payload = {
    zone_id:           el("p-zone-id").value,
    current_timestamp: new Date(el("p-timestamp").value).toISOString(),
    lookahead_mins:    parseInt(el("p-lookahead").value),
  };

  try {
    const result = await apiFetch("/predict/", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });
    renderPredictResult(result);
  } catch (err) {
    el("predict-result").innerHTML = `<div class="result-empty" style="color:var(--accent-red)">
      Error: ${err.message}<br/><small>Is the API running?</small></div>`;
  } finally {
    btn.textContent = "Run Prediction →";
    btn.disabled = false;
  }
});

function renderPredictResult(r) {
  const fillPct = Math.round(r.fill_probability * 100);
  const confPct = Math.round(r.confidence * 100);
  const cls = occClass(r.fill_probability);

  const altsHtml = r.alternative_zones?.length
    ? r.alternative_zones.map(a => `
        <div class="alt-zone-item">
          <span class="alt-zone-id">${a.zone_id}</span>
          <span class="alt-zone-pct pct-${occClass(a.fill_probability)}">${formatPct(a.fill_probability)} full</span>
          <span class="alt-zone-dist">${a.distance_m ? `${a.distance_m}m away` : ""}</span>
        </div>`).join("")
    : `<div class="result-empty" style="padding:16px 0">No alternatives needed ✓</div>`;

  el("predict-result").innerHTML = `
    <div class="result-fill-gauge">
      <div class="gauge-label">Fill Probability (${r.lookahead_mins} min lookahead)</div>
      <div class="gauge-value">${fillPct}%</div>
      <div class="gauge-bar-track" style="width:100%">
        <div class="gauge-bar-fill" style="width:${fillPct}%;
          background:${r.fill_probability > 0.75 ? 'var(--accent-red)' : r.fill_probability > 0.5 ? 'var(--accent-amber)' : 'var(--accent-green)'}"></div>
      </div>
    </div>
    <div class="result-meta-grid">
      <div class="result-meta-item">
        <div class="result-meta-label">Confidence</div>
        <div class="result-meta-val" style="color:var(--accent-cyan)">${confPct}%</div>
      </div>
      <div class="result-meta-item">
        <div class="result-meta-label">Model</div>
        <div class="result-meta-val" style="font-size:0.75rem; color:var(--text-muted)">${r.model_version}</div>
      </div>
    </div>
    <div class="result-meta-item" style="background:var(--bg-elevated); border-radius:8px; padding:12px; font-size:0.82rem; color:var(--text-secondary)">
      💡 ${r.reason}
    </div>
    ${r.alternative_zones?.length ? `<div><div class="label-sm" style="margin-bottom:8px">Alternative Zones</div><div class="result-alts">${altsHtml}</div></div>` : altsHtml}
  `;
}

// ─────────────────────────────────────────────────────────────────────────────
// Zone Explorer
// ─────────────────────────────────────────────────────────────────────────────

async function loadZones() {
  const tbody = el("zones-tbody");
  tbody.innerHTML = `<tr><td colspan="6" class="loading-cell">Loading…</td></tr>`;

  try {
    const zones = await apiFetch("/zones/");
    const details = await Promise.all(
      zones.map(z => apiFetch(`/zones/${z.zone_id}`).catch(() => z))
    );
    renderZonesTable(details);
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="6" class="loading-cell" style="color:var(--accent-red)">
      Failed to load zones: ${err.message}</td></tr>`;
  }
}

function renderZonesTable(zones) {
  const tbody = el("zones-tbody");
  tbody.innerHTML = zones.map(z => {
    const occ = z.current_occupancy_pct ?? null;
    const occHtml = occ !== null
      ? `<span class="occ-pill occ-${occClass(occ)}">${formatPct(occ)}</span>`
      : `<span class="occ-pill" style="background:none;color:var(--text-muted)">—</span>`;
    return `
      <tr>
        <td>${z.zone_id}</td>
        <td style="color:var(--text-primary);font-weight:500">${z.name}</td>
        <td>${z.venue_type ?? "—"}</td>
        <td style="font-family:'JetBrains Mono',monospace">${z.transit_score?.toFixed(2) ?? "—"}</td>
        <td>${occHtml}</td>
        <td style="font-size:0.78rem;font-family:'JetBrains Mono',monospace;color:var(--text-muted)">
          ${z.lat.toFixed(4)}, ${z.lon.toFixed(4)}</td>
      </tr>`;
  }).join("");
}

// Zone search filter
el("zone-search")?.addEventListener("input", function () {
  const q = this.value.toLowerCase();
  el("zones-tbody").querySelectorAll("tr").forEach(row => {
    row.style.display = row.textContent.toLowerCase().includes(q) ? "" : "none";
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// HNSW Search Panel
// ─────────────────────────────────────────────────────────────────────────────

el("h-k")?.addEventListener("input", () => {
  el("h-k-val").textContent = el("h-k").value;
});

el("hnsw-form")?.addEventListener("submit", async e => {
  e.preventDefault();
  const btn = el("btn-hnsw");
  btn.textContent = "Searching…";
  btn.disabled = true;

  const zoneId  = el("h-zone-id").value;
  const k       = parseInt(el("h-k").value);
  const vtype   = el("h-venue-type").value;

  // Since HNSW is a service-level call, we demo it via the SLM tool endpoint
  // In production this calls HNSWSearchService.zone_semantic_search directly
  const mockResults = Array.from({ length: k }, (_, i) => ({
    zone_id:    `${zoneId}-similar-${String.fromCharCode(65 + i)}`,
    similarity: +(0.94 - i * 0.08).toFixed(3),
  }));

  el("hnsw-results").innerHTML = mockResults.map((r, idx) => `
    <div class="hnsw-result-item" style="animation-delay:${idx * 60}ms">
      <div class="hnsw-rank">${idx + 1}</div>
      <div style="flex:1;min-width:0">
        <div style="font-size:0.82rem;font-family:'JetBrains Mono',monospace;
          color:var(--accent-cyan);margin-bottom:6px">${r.zone_id}</div>
        <div class="hnsw-sim-bar">
          <div class="hnsw-sim-fill" style="width:${Math.round(r.similarity * 100)}%"></div>
        </div>
      </div>
      <div class="hnsw-sim-val">${(r.similarity * 100).toFixed(1)}%</div>
    </div>`).join("");

  btn.textContent = "Search Similar Zones →";
  btn.disabled = false;
});

// ─────────────────────────────────────────────────────────────────────────────
// SLM Tools Panel
// ─────────────────────────────────────────────────────────────────────────────

function loadToolSchemas() {
  // Display the OpenAI-compatible tool schemas
  const schemas = [
    {
      type: "function",
      function: {
        name: "zone_semantic_search",
        description: "Find zones with similar characteristics to the current zone.",
        parameters: {
          type: "object",
          properties: {
            current_zone_id: { type: "string" },
            k:               { type: "integer", default: 3 },
            max_distance_m:  { type: "number" },
            venue_type:      { type: "string" },
          },
          required: ["current_zone_id"],
        },
      },
    },
    {
      type: "function",
      function: {
        name: "get_zone_forecast",
        description: "Get fill-probability forecast for a zone at estimated arrival.",
        parameters: {
          type: "object",
          properties: {
            zone_id:           { type: "string" },
            time_horizon_mins: { type: "integer", default: 30 },
          },
          required: ["zone_id"],
        },
      },
    },
    {
      type: "function",
      function: {
        name: "log_outcome",
        description: "Record actual parking outcome (opt-in telemetry).",
        parameters: {
          type: "object",
          properties: {
            zone_id:              { type: "string" },
            timestamp:            { type: "string", format: "date-time" },
            actual_availability:  { type: "number" },
            rider_satisfaction:   { type: "integer", minimum: 1, maximum: 5 },
          },
          required: ["zone_id", "timestamp", "actual_availability"],
        },
      },
    },
  ];
  el("tool-schemas").textContent = JSON.stringify(schemas, null, 2);
}

// Tool tabs
document.querySelectorAll(".tool-tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tool-tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".tool-pane").forEach(p => p.classList.remove("active"));
    tab.classList.add("active");
    const pane = el(`pane-${tab.dataset.tool}`);
    if (pane) pane.classList.add("active");
  });
});

// zone_semantic_search tool call
el("btn-slm-search")?.addEventListener("click", async () => {
  const payload = {
    current_zone_id: el("slm-search-zone").value,
    k:               parseInt(el("slm-search-k").value),
  };
  el("slm-search-req").textContent = JSON.stringify(payload, null, 2);

  try {
    // Stub: calls /zones/ to simulate similar zones
    const zones = await apiFetch("/zones/");
    const results = zones.slice(0, payload.k).map((z, i) => ({
      zone_id:    z.zone_id,
      similarity: +(0.92 - i * 0.07).toFixed(3),
      occupancy_profile: [],
    }));
    el("slm-search-resp").textContent = JSON.stringify(results, null, 2);
  } catch (err) {
    el("slm-search-resp").textContent = `Error: ${err.message}`;
  }
});

// get_zone_forecast tool call
el("btn-slm-forecast")?.addEventListener("click", async () => {
  const payload = {
    zone_id:           el("slm-fc-zone").value,
    time_horizon_mins: parseInt(el("slm-fc-horizon").value),
  };
  el("slm-fc-req").textContent = JSON.stringify(payload, null, 2);

  try {
    const result = await apiFetch("/predict/", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ zone_id: payload.zone_id, lookahead_mins: payload.time_horizon_mins }),
    });
    const forecast = {
      zone_id:            payload.zone_id,
      arrival_time:       new Date(Date.now() + payload.time_horizon_mins * 60000).toISOString(),
      expected_fill:      result.fill_probability,
      confidence:         result.confidence,
      recommended_action: result.fill_probability < 0.70 ? "park here" : `reroute — ${formatPct(result.fill_probability)} full`,
    };
    el("slm-fc-resp").textContent = JSON.stringify(forecast, null, 2);
  } catch (err) {
    el("slm-fc-resp").textContent = `Error: ${err.message}`;
  }
});

// log_outcome tool call
el("btn-slm-outcome")?.addEventListener("click", () => {
  const payload = {
    zone_id:              el("slm-lo-zone").value,
    timestamp:            new Date().toISOString(),
    actual_availability:  parseFloat(el("slm-lo-avail").value),
    rider_satisfaction:   parseInt(el("slm-lo-sat").value),
  };
  el("slm-lo-req").textContent  = JSON.stringify(payload, null, 2);
  // Stub response (no backend endpoint for telemetry yet)
  el("slm-lo-resp").textContent = JSON.stringify({ status: "ok", message: "Outcome logged (stub)" }, null, 2);
});

// ─────────────────────────────────────────────────────────────────────────────
// Refresh button
// ─────────────────────────────────────────────────────────────────────────────

el("btn-refresh")?.addEventListener("click", () => {
  checkHealth();
  loadOverview();
});

// ─────────────────────────────────────────────────────────────────────────────
// Boot
// ─────────────────────────────────────────────────────────────────────────────

(function init() {
  checkHealth();
  loadOverview();
  // Auto-refresh health every 30s
  setInterval(checkHealth, 30_000);
})();
