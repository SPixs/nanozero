// dashboard.js — single-page synthétique + jobserver proxy

const evtSource = new EventSource('/api/stream');
let lossChart = null;
let sprtChart = null;
let chartGen = -1;

// --- SSE local state (run_state.yaml + versions.yaml) ---
evtSource.addEventListener('state', (e) => {
  const data = JSON.parse(e.data);
  updateSummary(data);
  updateTrainProgress(data);
  updateEval(data);
});
evtSource.addEventListener('error', () => {});

function updateSummary(data) {
  const rs = data.run_state;
  if (!rs) {
    document.getElementById('run-id').textContent = '(no active run)';
    return;
  }
  document.getElementById('run-id').textContent = rs.run_id;
  const status = document.getElementById('run-status');
  status.textContent = rs.status;
  status.className = 'badge ' + rs.status;
  document.getElementById('run-phase').textContent = rs.phase;
  document.getElementById('run-gen').textContent = `gen ${rs.current_gen}`;
}

function updateTrainProgress(data) {
  const rs = data.run_state;
  if (!rs) return;
  const tr = rs.train;
  document.getElementById('train-text').textContent =
    `epoch ${tr.current_epoch} / ${tr.total_epochs || '?'}`;
  const bar = document.getElementById('train-bar');
  if (tr.total_epochs > 0) {
    bar.style.width = `${100 * tr.current_epoch / tr.total_epochs}%`;
  }
  refreshLossChart(rs.current_gen);
}

function updateEval(data) {
  const rs = data.run_state;
  if (!rs) return;
  const ev = rs.eval;
  document.getElementById('eval-challenger').textContent = ev.challenger || '—';
  document.getElementById('eval-baseline').textContent = ev.baseline || '—';
  document.getElementById('eval-games').textContent = ev.games_played ?? '—';
  const dec = document.getElementById('eval-decision');
  dec.textContent = ev.last_decision || '—';
  dec.className = 'decision ' + (ev.last_decision || '');
  refreshSprtChart(rs.current_gen);
}

// --- Versions table (5 dernières) ---
function refreshVersions() {
  fetch('/api/versions').then(r => r.json()).then(vs => {
    const tbody = document.getElementById('versions-tbody');
    tbody.innerHTML = '';
    const last5 = vs.all.slice(-5).reverse();
    for (const v of last5) {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${v.name}</td>
        <td>${v.type}</td>
        <td>${v.promoted ? 'yes' : 'no'}</td>
        <td>${v.sprt_result || '—'}</td>
      `;
      tbody.appendChild(tr);
    }
  }).catch(() => {});
}

// --- Charts ---
function initCharts() {
  const cOpts = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { labels: { font: { size: 10 }, boxWidth: 12 } } },
    scales: {
      x: { ticks: { font: { size: 9 } } },
      y: { beginAtZero: true, ticks: { font: { size: 9 } } },
    },
    animation: false,
  };
  lossChart = new Chart(document.getElementById('loss-chart'), {
    type: 'line',
    data: { labels: [], datasets: [
      { label: 'policy', data: [], borderColor: '#4a8aff', borderWidth: 1.5, pointRadius: 0 },
      { label: 'value', data: [], borderColor: '#ff8a4a', borderWidth: 1.5, pointRadius: 0 },
      { label: 'total', data: [], borderColor: '#8a4aff', borderWidth: 1.5, pointRadius: 0 },
    ]},
    options: cOpts,
  });
  sprtChart = new Chart(document.getElementById('sprt-chart'), {
    type: 'line',
    data: { labels: [], datasets: [
      { label: 'LLR', data: [], borderColor: '#4aff8a', borderWidth: 1.5, pointRadius: 0 },
    ]},
    options: { ...cOpts, scales: { ...cOpts.scales, y: { ticks: { font: { size: 9 } } } } },
  });
}

function refreshLossChart(gen) {
  if (gen == null || gen < 1) return;
  fetch(`/api/metrics/${gen}`).then(r => r.json()).then(rows => {
    lossChart.data.labels = rows.map(r => `e${r.epoch}`);
    lossChart.data.datasets[0].data = rows.map(r => parseFloat(r.policy_loss));
    lossChart.data.datasets[1].data = rows.map(r => parseFloat(r.value_loss));
    lossChart.data.datasets[2].data = rows.map(r => parseFloat(r.total_loss));
    lossChart.update('none');
  }).catch(() => {});
}

function refreshSprtChart(gen) {
  if (gen == null || gen < 1) return;
  fetch(`/api/sprt/${gen}`).then(r => r.json()).then(rows => {
    sprtChart.data.labels = rows.map(r => r.games_played);
    sprtChart.data.datasets[0].data = rows.map(r => parseFloat(r.llr) || 0);
    sprtChart.update('none');
  }).catch(() => {});
}

// --- Jobserver proxy (self-play distribué) ---
let lastJsCompleted = null;
let lastJsTime = null;
function refreshJobserver() {
  fetch('/api/jobserver').then(r => {
    if (!r.ok) throw new Error('jobserver unreachable');
    return r.json();
  }).then(d => {
    const j = d.jobs || {};
    const total = (j.pending || 0) + (j.claimed || 0) + (j.completed || 0);
    const done = j.completed || 0;
    const pct = total > 0 ? (100 * done / total) : 0;
    document.getElementById('jobs-bar').style.width = `${pct}%`;
    document.getElementById('jobs-text').textContent =
      `${done.toLocaleString()} / ${total.toLocaleString()}  (${pct.toFixed(1)}%)`;
    document.getElementById('js-positions').textContent =
      (d.positions_total || 0).toLocaleString();
    document.getElementById('js-current-model').textContent =
      `v${d.current_model_version || '?'}`;

    // workers + débit
    if (d.workers !== undefined) {
      document.getElementById('js-workers').textContent = d.workers;
    }
    const now = Date.now() / 1000;
    if (lastJsCompleted !== null && lastJsTime !== null) {
      const dt = now - lastJsTime;
      const dc = done - lastJsCompleted;
      if (dt > 0) {
        const rate = (dc / dt) * 60;  // jobs/min
        document.getElementById('js-rate').textContent =
          `${rate.toFixed(1)} jobs/min`;
        const remaining = (j.pending || 0) + (j.claimed || 0);
        if (rate > 0) {
          const etaMin = remaining / rate;
          let etaTxt;
          if (etaMin < 1) etaTxt = '< 1 min';
          else if (etaMin < 60) etaTxt = `${Math.round(etaMin)} min`;
          else etaTxt = `${(etaMin/60).toFixed(1)} h`;
          document.getElementById('js-eta').textContent = etaTxt;
        }
      }
    }
    lastJsCompleted = done;
    lastJsTime = now;
  }).catch(() => {
    document.getElementById('jobs-text').textContent = '(jobserver injoignable)';
  });
}

// --- bootstrap ---
initCharts();
refreshJobserver();
refreshVersions();
setInterval(refreshJobserver, 5000);
setInterval(refreshVersions, 15000);
