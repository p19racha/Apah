/* Apah Control Dashboard Frontend Application Logic */

document.addEventListener('DOMContentLoaded', () => {
  // State variables
  let allModels = [];
  let loadedModelName = null;
  let selectedModel = null;
  let wsStats = null;
  let wsLogs = null;
  let auditLogsHistory = [];

  // DOM Elements
  const navItems = document.querySelectorAll('.nav-item');
  const viewSections = document.querySelectorAll('.view-section');
  const detailPanel = document.getElementById('detail-panel');
  const btnClosePanel = document.getElementById('btn-close-panel');
  const btnRefreshAll = document.getElementById('btn-refresh-all');

  // Navigation View Switching
  navItems.forEach(item => {
    item.addEventListener('click', () => {
      const targetView = item.getAttribute('data-view');
      navItems.forEach(i => i.classList.remove('active'));
      item.classList.add('active');

      viewSections.forEach(sec => {
        if (sec.id === `view-${targetView}`) {
          sec.classList.add('active');
        } else {
          sec.classList.remove('active');
        }
      });
    });
  });

  // Close Detail Panel
  if (btnClosePanel) {
    btnClosePanel.addEventListener('click', () => {
      detailPanel.classList.remove('open');
      document.querySelectorAll('#models-tbody tr').forEach(r => r.classList.remove('selected'));
    });
  }

  // Panel Tabs Switching
  const panelTabs = document.querySelectorAll('.panel-tab');
  panelTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const ptab = tab.getAttribute('data-ptab');
      panelTabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');

      document.getElementById('panel-body-overview').style.display = ptab === 'overview' ? 'flex' : 'none';
      document.getElementById('panel-body-activity').style.display = ptab === 'activity' ? 'flex' : 'none';
      document.getElementById('panel-body-manifest').style.display = ptab === 'manifest' ? 'block' : 'none';
    });
  });

  // Helper: Format Bytes to Human Size
  function formatBytes(bytes) {
    if (!bytes || bytes === 0) return '0.0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  // Fetch Models Manifest List
  async function fetchModels() {
    try {
      const res = await fetch('/models');
      if (res.ok) {
        allModels = await res.json();
        renderModelsTable(allModels);
        updateModelsSummaryCards(allModels);
      }
    } catch (e) {
      console.warn('Error fetching /models:', e);
    }
  }

  // Fetch Loaded Process Status (/ps)
  async function fetchProcessStatus() {
    try {
      const res = await fetch('/ps');
      if (res.ok) {
        const psList = await res.json();
        if (psList && psList.length > 0) {
          loadedModelName = psList[0].name;
          document.getElementById('card-active-model-name').textContent = loadedModelName;
          document.getElementById('card-active-model-sub').textContent = `GPU VRAM: ${psList[0].gpu_memory_mb.toFixed(1)} MB | Uptime: ${Math.round(psList[0].uptime_seconds)}s`;
        } else {
          loadedModelName = null;
          document.getElementById('card-active-model-name').textContent = 'None';
          document.getElementById('card-active-model-sub').textContent = 'GPU VRAM: 0.0 MB';
        }
        renderModelsTable(allModels);
      }
    } catch (e) {
      console.warn('Error fetching /ps:', e);
    }
  }

  // Render Models Table
  function renderModelsTable(models) {
    const tbody = document.getElementById('models-tbody');
    const countLabel = document.getElementById('models-table-count');
    if (!tbody) return;

    if (countLabel) countLabel.textContent = `${models.length} models registered`;

    if (!models || models.length === 0) {
      tbody.innerHTML = `
        <tr>
          <td colspan="7" style="text-align: center; color: var(--text-muted); padding: 30px;">
            No model manifests found in ~/.apah/models/. Pull models using 'apah pull'.
          </td>
        </tr>
      `;
      return;
    }

    tbody.innerHTML = '';
    models.forEach(m => {
      const isLoaded = loadedModelName && (loadedModelName === m.name || loadedModelName.includes(m.name));
      const tr = document.createElement('tr');
      if (selectedModel && selectedModel.name === m.name) tr.classList.add('selected');

      const statusBadge = isLoaded
        ? '<span class="badge badge-success">Loaded</span>'
        : '<span class="badge badge-neutral">Unloaded</span>';

      const pulledDate = m.pulled_at ? new Date(m.pulled_at).toLocaleDateString() : '-';

      tr.innerHTML = `
        <td><strong style="color: var(--text-main);">${m.name}</strong></td>
        <td><code style="font-size: 11.5px; color: var(--text-muted);">${m.version || 'v1.0.0'}</code></td>
        <td><span class="badge badge-info">${m.quant || 'none'}</span></td>
        <td>${formatBytes(m.size_bytes)}</td>
        <td>${statusBadge}</td>
        <td style="color: var(--text-muted);">${pulledDate}</td>
        <td><button class="btn btn-secondary" style="padding: 3px 8px; font-size: 11px;">Details</button></td>
      `;

      tr.addEventListener('click', () => {
        document.querySelectorAll('#models-tbody tr').forEach(r => r.classList.remove('selected'));
        tr.classList.add('selected');
        openModelDetailPanel(m, isLoaded);
      });

      tbody.appendChild(tr);
    });
  }

  // Update Summary Cards
  function updateModelsSummaryCards(models) {
    document.getElementById('card-total-models-count').textContent = models.length;
    const totalBytes = models.reduce((acc, m) => acc + (m.size_bytes || 0), 0);
    document.getElementById('card-total-size-text').textContent = `Total size: ${formatBytes(totalBytes)}`;
  }

  // Open Model Detail Panel
  function openModelDetailPanel(model, isLoaded) {
    selectedModel = model;
    document.getElementById('panel-model-name').textContent = model.name;

    const statusEl = document.getElementById('panel-model-status');
    if (isLoaded) {
      statusEl.textContent = 'Loaded';
      statusEl.className = 'badge badge-success';
    } else {
      statusEl.textContent = 'Unloaded';
      statusEl.className = 'badge badge-neutral';
    }

    document.getElementById('panel-val-version').textContent = model.version || 'v1.0.0';
    document.getElementById('panel-val-quant').textContent = model.quant || 'none';
    document.getElementById('panel-val-size').textContent = formatBytes(model.size_bytes);
    document.getElementById('panel-val-arch').textContent = model.architecture || 'unknown';
    document.getElementById('panel-val-pulled').textContent = model.pulled_at || '-';
    document.getElementById('panel-val-checksum').textContent = model.checksum_sha256 ? model.checksum_sha256.substring(0, 16) + '...' : '-';

    document.getElementById('panel-manifest-json').textContent = JSON.stringify(model, null, 2);

    const btnLoad = document.getElementById('btn-panel-action-load');
    if (isLoaded) {
      btnLoad.textContent = 'Unload Model';
      btnLoad.className = 'btn btn-danger';
      btnLoad.onclick = () => unloadModel();
    } else {
      btnLoad.textContent = 'Load Model';
      btnLoad.className = 'btn btn-primary';
      btnLoad.onclick = () => loadModel(model.name);
    }

    // Filter mini activity feed
    updatePanelActivityFeed(model.name);

    detailPanel.classList.add('open');
  }

  // Action: Load Model
  async function loadModel(modelName) {
    try {
      const res = await fetch('/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_path: modelName })
      });
      const data = await res.json();
      if (res.ok) {
        alert(`Success: ${data.message || 'Model loaded.'}`);
        fetchProcessStatus();
      } else {
        alert(`Error: ${data.error || 'Failed to load model.'}`);
      }
    } catch (e) {
      alert(`Error calling /load: ${e}`);
    }
  }

  // Action: Unload Model
  async function unloadModel() {
    try {
      const res = await fetch('/unload', { method: 'POST' });
      const data = await res.json();
      if (res.ok) {
        alert(`Success: ${data.message || 'Model unloaded.'}`);
        fetchProcessStatus();
      } else {
        alert(`Error: ${data.error || 'Failed to unload model.'}`);
      }
    } catch (e) {
      alert(`Error calling /unload: ${e}`);
    }
  }

  // Copy Model Name
  document.getElementById('btn-panel-copy-name').addEventListener('click', () => {
    if (selectedModel) {
      navigator.clipboard.writeText(selectedModel.name);
      alert(`Copied "${selectedModel.name}" to clipboard.`);
    }
  });

  // WebSocket 1: Stats (/ws/stats)
  function connectWebSocketStats() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${location.host}/ws/stats`;
    wsStats = new WebSocket(wsUrl);

    wsStats.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        renderStats(data);
      } catch (e) {
        console.warn('Invalid stats JSON:', e);
      }
    };

    wsStats.onclose = () => {
      setTimeout(connectWebSocketStats, 2000);
    };
  }

  function renderStats(data) {
    // GPU cards
    const container = document.getElementById('gpu-cards-container');
    if (container && data.gpu) {
      document.getElementById('card-gpu-count-text').textContent = `${data.gpu.length} Visible GPU${data.gpu.length === 1 ? '' : 's'}`;
      if (data.gpu.length > 0) {
        document.getElementById('card-gpu-temp-text').textContent = `Temp: ${data.gpu[0].temperature_c}°C | Util: ${data.gpu[0].utilization_pct}%`;
        
        // Update sidebar widgets
        const g0 = data.gpu[0];
        document.getElementById('widget-vram-text').textContent = `${g0.memory_used_mb.toFixed(0)} / ${g0.memory_total_mb.toFixed(0)} MB`;
        const vramPct = (g0.memory_used_mb / (g0.memory_total_mb || 1)) * 100;
        document.getElementById('widget-vram-bar').style.width = `${Math.min(100, vramPct)}%`;

        document.getElementById('widget-gpu-util').textContent = `${g0.utilization_pct}%`;
        document.getElementById('widget-gpu-bar').style.width = `${Math.min(100, g0.utilization_pct)}%`;
      }

      container.innerHTML = '';
      data.gpu.forEach(g => {
        const div = document.createElement('div');
        div.className = 'card';
        div.innerHTML = `
          <div class="card-header-small">
            <span>GPU [${g.index}] ${g.name}</span>
            <span>🌡️ ${g.temperature_c}°C</span>
          </div>
          <div class="card-value">${g.utilization_pct}% Util</div>
          <div class="progress-group" style="margin-top: 8px;">
            <div class="progress-label">
              <span>VRAM Usage</span>
              <span>${g.memory_used_mb.toFixed(0)} / ${g.memory_total_mb.toFixed(0)} MB</span>
            </div>
            <div class="progress-bar-bg">
              <div class="progress-bar-fill" style="width: ${((g.memory_used_mb / (g.memory_total_mb || 1)) * 100).toFixed(1)}%;"></div>
            </div>
          </div>
        `;
        container.appendChild(div);
      });
    }

    // Scheduler stats
    if (data.scheduler) {
      document.getElementById('perf-active-batch').textContent = data.scheduler.active_batch_size || 0;
      document.getElementById('perf-waiting-queue').textContent = data.scheduler.waiting_queue_length || 0;
      const rate = data.scheduler.aggregate_throughput_tok_s || 0.0;
      document.getElementById('perf-throughput').textContent = `${rate.toFixed(1)} tok/s`;
    }
  }

  // WebSocket 2: Logs (/ws/logs)
  function connectWebSocketLogs() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${location.host}/ws/logs`;
    wsLogs = new WebSocket(wsUrl);

    wsLogs.onopen = () => {
      const badge = document.getElementById('log-ws-status');
      if (badge) {
        badge.textContent = 'Connected';
        badge.className = 'badge badge-success';
      }
    };

    wsLogs.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        auditLogsHistory.push(payload);
        appendLogLine(payload);
        if (selectedModel) updatePanelActivityFeed(selectedModel.name);
      } catch (e) {
        // Plain text log fallback
        appendLogLine({ event_type: 'raw', details: { text: event.data } });
      }
    };

    wsLogs.onclose = () => {
      const badge = document.getElementById('log-ws-status');
      if (badge) {
        badge.textContent = 'Reconnecting...';
        badge.className = 'badge badge-warning';
      }
      setTimeout(connectWebSocketLogs, 2000);
    };
  }

  function appendLogLine(logObj) {
    const term = document.getElementById('log-terminal-container');
    if (!term) return;

    const div = document.createElement('div');
    div.className = 'log-line';

    const ts = logObj.timestamp ? new Date(logObj.timestamp).toLocaleTimeString() : new Date().toLocaleTimeString();
    const eventType = logObj.event_type || 'event';
    const detailsStr = logObj.details ? JSON.stringify(logObj.details) : '';

    div.innerHTML = `<span class="log-ts">[${ts}]</span> <span class="log-event">${eventType}</span> ${detailsStr}`;
    term.appendChild(div);
    term.scrollTop = term.scrollHeight;
  }

  function updatePanelActivityFeed(modelName) {
    const feed = document.getElementById('panel-activity-feed');
    if (!feed) return;

    const matching = auditLogsHistory.filter(l => {
      const d = l.details || {};
      return d.model === modelName || d.model_path === modelName || d.model_name === modelName;
    });

    if (matching.length === 0) {
      feed.innerHTML = '<div style="color: var(--text-muted); font-size: 12px;">No recent request activity logged for this model.</div>';
      return;
    }

    feed.innerHTML = '';
    matching.slice(-5).reverse().forEach(item => {
      const div = document.createElement('div');
      div.className = 'feed-item';
      const ts = item.timestamp ? new Date(item.timestamp).toLocaleTimeString() : '';
      div.innerHTML = `
        <div class="feed-item-header">
          <span>${item.event_type}</span>
          <span style="color: var(--text-sub);">${ts}</span>
        </div>
        <div style="color: var(--text-muted); font-family: var(--font-mono); font-size: 11px; word-break: break-all;">
          ${JSON.stringify(item.details || {})}
        </div>
      `;
      feed.appendChild(div);
    });
  }

  // Button Refresh
  if (btnRefreshAll) {
    btnRefreshAll.addEventListener('click', () => {
      fetchModels();
      fetchProcessStatus();
    });
  }

  // Initial load
  fetchModels();
  fetchProcessStatus();
  connectWebSocketStats();
  connectWebSocketLogs();
});
