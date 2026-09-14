/* Apah Sovereign AI Control Dashboard JavaScript — App Logic */

document.addEventListener('DOMContentLoaded', () => {
  // Global State
  let state = {
    models: [],
    loadedModelStatus: null,
    gpuStats: [],
    selectedModel: null,
    logStreamPaused: false,
    logEntries: [],
    wsLogs: null,
    wsStats: null,
  };

  // DOM Elements
  const navItems = document.querySelectorAll('.nav-item');
  const viewSections = document.querySelectorAll('.view-section');
  const detailPanel = document.getElementById('detailPanel');
  const btnClosePanel = document.getElementById('btnClosePanel');
  const panelTabs = document.querySelectorAll('.panel-tab');

  // Modals
  const addModelModal = document.getElementById('addModelModal');
  const manifestModal = document.getElementById('manifestModal');
  const btnOpenAddModel = document.getElementById('btnOpenAddModel');
  const btnOpenAddModel2 = document.getElementById('btnOpenAddModel2');
  const btnCloseAddModel = document.getElementById('btnCloseAddModel');
  const btnCancelAddModel = document.getElementById('btnCancelAddModel');
  const formAddModel = document.getElementById('formAddModel');
  const selectSource = document.getElementById('selectSource');
  const groupLocalPath = document.getElementById('groupLocalPath');

  // Navigation Setup
  navItems.forEach(item => {
    item.addEventListener('click', () => {
      const targetView = item.getAttribute('data-view');
      navItems.forEach(n => n.classList.remove('active'));
      item.classList.add('active');

      viewSections.forEach(sec => {
        sec.style.display = sec.id === `view-${targetView}` ? 'block' : 'none';
      });
    });
  });

  // Panel Tabs Setup
  panelTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetPtab = tab.getAttribute('data-ptab');
      panelTabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');

      document.getElementById('panelTabDetails').style.display = targetPtab === 'details' ? 'block' : 'none';
      document.getElementById('panelTabActivity').style.display = targetPtab === 'activity' ? 'block' : 'none';
    });
  });

  btnClosePanel.addEventListener('click', () => {
    detailPanel.classList.add('hidden');
    document.querySelectorAll('#modelsTableBody tr').forEach(r => r.classList.remove('selected'));
  });

  // Modal Open/Close handlers
  const openAddModelModal = () => addModelModal.classList.remove('hidden');
  const closeAddModelModal = () => addModelModal.classList.add('hidden');
  btnOpenAddModel?.addEventListener('click', openAddModelModal);
  btnOpenAddModel2?.addEventListener('click', openAddModelModal);
  btnCloseAddModel?.addEventListener('click', closeAddModelModal);
  btnCancelAddModel?.addEventListener('click', closeAddModelModal);

  selectSource.addEventListener('change', () => {
    groupLocalPath.style.display = selectSource.value === 'local' ? 'block' : 'none';
  });

  document.getElementById('btnCloseManifest')?.addEventListener('click', () => {
    manifestModal.classList.add('hidden');
  });

  // Helper formatting
  function formatBytes(bytes) {
    if (!bytes || bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  function formatDate(isoStr) {
    if (!isoStr) return '-';
    try {
      const d = new Date(isoStr);
      return d.toLocaleString();
    } catch (e) {
      return isoStr;
    }
  }

  // Data Fetching: Models & Process Status
  async function loadModelsData() {
    try {
      const [modelsResp, psResp] = await Promise.all([
        fetch('/models').then(r => r.ok ? r.json() : []),
        fetch('/ps').then(r => r.ok ? r.json() : []),
      ]);

      state.models = modelsResp;
      state.loadedModelStatus = psResp.length > 0 ? psResp[0] : null;

      renderModelsView();
    } catch (err) {
      console.error('Error fetching models:', err);
    }
  }

  function renderModelsView() {
    const tbody = document.getElementById('modelsTableBody');
    const tableCount = document.getElementById('modelsTableCount');
    const activeModelVal = document.getElementById('activeModelVal');
    const activeModelSub = document.getElementById('activeModelSub');
    const totalModelsVal = document.getElementById('totalModelsVal');
    const totalSizeSub = document.getElementById('totalSizeSub');

    totalModelsVal.textContent = state.models.length;
    tableCount.textContent = `${state.models.length} models`;

    const totalBytes = state.models.reduce((sum, m) => sum + (m.size_bytes || 0), 0);
    totalSizeSub.textContent = `Total Size: ${formatBytes(totalBytes)}`;

    if (state.loadedModelStatus && state.loadedModelStatus.loaded) {
      activeModelVal.textContent = state.loadedModelStatus.name;
      activeModelSub.textContent = `${state.loadedModelStatus.gpu_memory_mb.toFixed(1)} MB GPU VRAM • Uptime ${state.loadedModelStatus.uptime_seconds.toFixed(0)}s`;
    } else {
      activeModelVal.textContent = 'None Loaded';
      activeModelSub.textContent = '0 MB GPU VRAM';
    }

    if (state.models.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-muted); padding: 24px;">No local models found in ~/.apah/models/. Use "+ Add Model" to pull one.</td></tr>`;
      return;
    }

    const searchTerm = document.getElementById('searchModels').value.toLowerCase();
    const filteredModels = state.models.filter(m => 
      m.name.toLowerCase().includes(searchTerm) || (m.quant && m.quant.toLowerCase().includes(searchTerm))
    );

    tbody.innerHTML = filteredModels.map(m => {
      const isLoaded = state.loadedModelStatus && (state.loadedModelStatus.name === m.name || state.loadedModelStatus.name === `${m.name}:${m.version}`);
      const statusBadge = isLoaded 
        ? `<span class="badge badge-success">Loaded</span>` 
        : `<span class="badge badge-secondary">Unloaded</span>`;

      return `
        <tr data-model-name="${m.name}" data-version="${m.version}">
          <td><strong>${m.name}</strong></td>
          <td>${m.version || 'v1.0.0'}</td>
          <td>${m.quant || 'none'}</td>
          <td>${formatBytes(m.size_bytes)}</td>
          <td>${statusBadge}</td>
          <td>${formatDate(m.pulled_at)}</td>
          <td><button class="btn btn-sm btn-select-row">Details</button></td>
        </tr>
      `;
    }).join('');

    // Attach row click events
    tbody.querySelectorAll('tr').forEach(row => {
      row.addEventListener('click', () => {
        const name = row.getAttribute('data-model-name');
        const ver = row.getAttribute('data-version');
        const modelObj = state.models.find(m => m.name === name && m.version === ver);
        if (modelObj) {
          selectModelRow(modelObj, row);
        }
      });
    });
  }

  function selectModelRow(modelObj, rowElement) {
    state.selectedModel = modelObj;
    document.querySelectorAll('#modelsTableBody tr').forEach(r => r.classList.remove('selected'));
    if (rowElement) rowElement.classList.add('selected');

    // Populate Right Panel
    document.getElementById('panelModelName').textContent = modelObj.name;
    document.getElementById('panelVersion').textContent = modelObj.version || 'v1.0.0';
    document.getElementById('panelQuant').textContent = modelObj.quant || 'none';
    document.getElementById('panelSize').textContent = formatBytes(modelObj.size_bytes);
    document.getElementById('panelSha256').textContent = modelObj.checksum_sha256 ? modelObj.checksum_sha256.substring(0, 16) + '...' : '-';

    const isLoaded = state.loadedModelStatus && (state.loadedModelStatus.name === modelObj.name || state.loadedModelStatus.name === `${modelObj.name}:${modelObj.version}`);
    const badgeContainer = document.getElementById('panelStatusBadge');
    const loadBtn = document.getElementById('btnPanelLoadUnload');

    if (isLoaded) {
      badgeContainer.innerHTML = `<span class="badge badge-success">LOADED</span>`;
      loadBtn.textContent = 'Unload Model';
      loadBtn.className = 'btn btn-danger';
    } else {
      badgeContainer.innerHTML = `<span class="badge badge-secondary">UNLOADED</span>`;
      loadBtn.textContent = 'Load Model';
      loadBtn.className = 'btn btn-primary';
    }

    detailPanel.classList.remove('hidden');
  }

  // Model Actions: Load / Unload / Manifest / Copy
  document.getElementById('btnPanelLoadUnload').addEventListener('click', async () => {
    if (!state.selectedModel) return;
    const isLoaded = state.loadedModelStatus && (state.loadedModelStatus.name === state.selectedModel.name || state.loadedModelStatus.name === `${state.selectedModel.name}:${state.selectedModel.version}`);

    if (isLoaded) {
      // Unload
      try {
        const res = await fetch('/unload', { method: 'POST' }).then(r => r.json());
        alert(res.message || 'Model unloaded.');
        await loadModelsData();
        if (state.selectedModel) selectModelRow(state.selectedModel, null);
      } catch (err) {
        alert('Failed to unload model: ' + err.message);
      }
    } else {
      // Load
      try {
        const payload = { model_path: state.selectedModel.name };
        const res = await fetch('/load', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        }).then(r => r.json());

        if (res.error) {
          alert('Load failed: ' + (res.error.message || res.error));
        } else {
          alert(res.message || 'Model loaded successfully.');
          await loadModelsData();
          if (state.selectedModel) selectModelRow(state.selectedModel, null);
        }
      } catch (err) {
        alert('Failed to load model: ' + err.message);
      }
    }
  });

  document.getElementById('btnPanelShowManifest').addEventListener('click', () => {
    if (!state.selectedModel) return;
    document.getElementById('manifestModalTitle').textContent = `Manifest: ${state.selectedModel.name}`;
    document.getElementById('manifestJsonContent').textContent = JSON.stringify(state.selectedModel, null, 2);
    manifestModal.classList.remove('hidden');
  });

  document.getElementById('btnPanelCopyName').addEventListener('click', () => {
    if (!state.selectedModel) return;
    navigator.clipboard.writeText(state.selectedModel.name);
    alert(`Copied model name '${state.selectedModel.name}' to clipboard!`);
  });

  // Add / Pull Model Form Submission
  formAddModel.addEventListener('submit', async (e) => {
    e.preventDefault();
    const modelName = document.getElementById('inputModelName').value.trim();
    const source = selectSource.value;
    const localPath = document.getElementById('inputLocalPath').value.trim();
    const revision = document.getElementById('inputRevision').value.trim();

    const btnSubmit = document.getElementById('btnSubmitPull');
    btnSubmit.disabled = true;
    btnSubmit.textContent = 'Pulling...';

    try {
      const payload = {
        model: modelName,
        source: source,
        local_path: localPath || null,
        revision: revision || 'main'
      };

      const res = await fetch('/pull', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      }).then(r => r.json());

      if (res.detail) {
        alert('Pull error: ' + res.detail);
      } else {
        alert(`Successfully pulled '${modelName}'!`);
        closeAddModelModal();
        await loadModelsData();
      }
    } catch (err) {
      alert('Error during pull: ' + err.message);
    } finally {
      btnSubmit.disabled = false;
      btnSubmit.textContent = 'Pull Model';
    }
  });

  // Settings View: Fetch & Save Config
  async function loadConfigData() {
    try {
      const cfg = await fetch('/config').then(r => r.json());
      document.getElementById('settingIdleTimeout').value = cfg.idle_timeout_seconds;
      document.getElementById('settingAutoUnload').checked = cfg.auto_unload_enabled;

      const airgapBadge = document.getElementById('cfgAirgapStatus');
      airgapBadge.innerHTML = cfg.airgap_mode_enabled 
        ? `<span class="badge badge-success">ENABLED (Socket Isolation)</span>` 
        : `<span class="badge badge-secondary">DISABLED</span>`;

      document.getElementById('cfgAuditContent').textContent = cfg.audit_log_content ? 'Full Content' : 'Metadata Only';
      document.getElementById('cfgAuditPath').textContent = cfg.audit_log_path;
    } catch (err) {
      console.error('Error fetching config:', err);
    }
  }

  document.getElementById('btnSaveConfig').addEventListener('click', async () => {
    const idleVal = parseInt(document.getElementById('settingIdleTimeout').value, 10);
    const autoVal = document.getElementById('settingAutoUnload').checked;

    try {
      const res = await fetch('/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          idle_timeout_seconds: idleVal,
          auto_unload_enabled: autoVal
        })
      }).then(r => r.json());

      const statusEl = document.getElementById('configSavedStatus');
      statusEl.textContent = 'Settings saved successfully!';
      setTimeout(() => { statusEl.textContent = ''; }, 3000);
    } catch (err) {
      alert('Failed to save config: ' + err.message);
    }
  });

  // Search Filter Handler
  document.getElementById('searchModels').addEventListener('input', renderModelsView);

  // WebSockets Setup: /ws/logs and /ws/stats
  function connectLogsWebSocket() {
    const wsProtocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${location.host}/ws/logs`;

    state.wsLogs = new WebSocket(wsUrl);
    const logContainer = document.getElementById('logContainer');
    const countBadge = document.getElementById('logCountBadge');

    state.wsLogs.onopen = () => {
      console.log('Connected to /ws/logs');
    };

    state.wsLogs.onmessage = (event) => {
      if (state.logStreamPaused) return;

      const rawLine = event.data;
      state.logEntries.push(rawLine);
      countBadge.textContent = `${state.logEntries.length} events`;

      renderLogs();
    };

    state.wsLogs.onclose = () => {
      console.warn('WebSocket /ws/logs closed. Reconnecting in 2s...');
      setTimeout(connectLogsWebSocket, 2000);
    };
  }

  function renderLogs() {
    const logContainer = document.getElementById('logContainer');
    const filterType = document.getElementById('logTypeFilter').value;
    const filterSearch = document.getElementById('logSearch').value.toLowerCase();

    const filtered = state.logEntries.filter(line => {
      if (filterType !== 'ALL' && !line.includes(`"event_type": "${filterType}"`)) return false;
      if (filterSearch && !line.toLowerCase().includes(filterSearch)) return false;
      return true;
    });

    logContainer.textContent = filtered.join('\n');
    logContainer.scrollTop = logContainer.scrollHeight;
  }

  document.getElementById('logTypeFilter').addEventListener('change', renderLogs);
  document.getElementById('logSearch').addEventListener('input', renderLogs);
  document.getElementById('btnClearLogs').addEventListener('click', () => {
    state.logEntries = [];
    document.getElementById('logCountBadge').textContent = '0 events';
    document.getElementById('logContainer').textContent = '';
  });

  document.getElementById('btnToggleLogPause').addEventListener('click', function() {
    state.logStreamPaused = !state.logStreamPaused;
    this.textContent = state.logStreamPaused ? 'Resume Stream' : 'Pause Stream';
  });

  function connectStatsWebSocket() {
    const wsProtocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${location.host}/ws/stats`;

    state.wsStats = new WebSocket(wsUrl);

    state.wsStats.onopen = () => {
      document.getElementById('serverStatusDot').className = 'status-dot';
      document.getElementById('serverStatusText').textContent = 'Server Online';
    };

    state.wsStats.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        updateTelemetryUI(data);
      } catch (err) {
        console.error('Error parsing stats JSON:', err);
      }
    };

    state.wsStats.onclose = () => {
      document.getElementById('serverStatusDot').className = 'status-dot offline';
      document.getElementById('serverStatusText').textContent = 'Disconnected';
      setTimeout(connectStatsWebSocket, 2000);
    };
  }

  function updateTelemetryUI(data) {
    const server = data.server || {};
    const gpuList = data.gpu || [];
    const sched = data.scheduler || {};

    // Update Topbar & Sidebar VRAM
    let totalUsedMb = 0;
    let totalMaxMb = 0;
    gpuList.forEach(g => {
      totalUsedMb += (g.memory_used_mb || 0);
      totalMaxMb += (g.memory_total_mb || 0);
    });

    if (totalMaxMb === 0 && server.gpu_memory_mb) {
      totalUsedMb = server.gpu_memory_mb;
      totalMaxMb = 24576; // Default fallback estimate if NVML not queried
    }

    document.getElementById('topbarVramText').textContent = `${totalUsedMb.toFixed(0)} / ${totalMaxMb.toFixed(0)} MB`;
    
    const pct = totalMaxMb > 0 ? (totalUsedMb / totalMaxMb) * 100 : 0;
    document.getElementById('sidebarVramBar').style.width = `${pct.toFixed(1)}%`;
    document.getElementById('sidebarVramVal').textContent = `${(totalUsedMb/1024).toFixed(1)} / ${(totalMaxMb/1024).toFixed(1)} GB (${pct.toFixed(0)}%)`;

    // Telemetry Summary Card
    const avgUtil = gpuList.length > 0 ? (gpuList.reduce((acc, g) => acc + g.utilization_pct, 0) / gpuList.length) : 0;
    const avgTemp = gpuList.length > 0 ? (gpuList.reduce((acc, g) => acc + g.temperature_c, 0) / gpuList.length) : 0;
    document.getElementById('gpuTelemetryVal').textContent = `${avgUtil.toFixed(0)}% Util`;
    document.getElementById('gpuTempSub').textContent = `${avgTemp.toFixed(0)} °C Avg Temp`;

    // Performance View Cards Grid
    const gpuGrid = document.getElementById('gpuCardsGrid');
    if (gpuList.length > 0) {
      gpuGrid.innerHTML = gpuList.map(g => `
        <div class="card">
          <div class="card-label">GPU ${g.index}: ${g.name}</div>
          <div class="card-value">${g.utilization_pct}% Util</div>
          <div class="progress-bar">
            <div class="progress-fill" style="width: ${((g.memory_used_mb/g.memory_total_mb)*100).toFixed(1)}%;"></div>
          </div>
          <div class="card-sub">${g.memory_used_mb.toFixed(0)} / ${g.memory_total_mb.toFixed(0)} MB VRAM • ${g.temperature_c} °C</div>
        </div>
      `).join('');
    } else {
      gpuGrid.innerHTML = `
        <div class="card">
          <div class="card-label">GPU Device</div>
          <div class="card-value">CPU / Virtual Mode</div>
          <div class="card-sub">No NVML GPU detected</div>
        </div>
      `;
    }

    // Performance Scheduler metrics
    document.getElementById('perfActiveBatch').textContent = sched.active_batch_size || 0;
    document.getElementById('perfWaitQueue').textContent = sched.waiting_queue_length || 0;
    document.getElementById('perfThroughput').textContent = `${(sched.aggregate_throughput_tok_s || 0).toFixed(1)} tok/s`;
    document.getElementById('perfTotalTokens').textContent = (sched.total_tokens_generated || 0).toLocaleString();
  }

  // Initial Initialization
  loadModelsData();
  loadConfigData();
  connectLogsWebSocket();
  connectStatsWebSocket();

  // Periodic Refresh
  setInterval(loadModelsData, 5000);
});
