/* Apah Sovereign AI Control Dashboard JavaScript — Minimalist & Robust App Engine */

document.addEventListener('DOMContentLoaded', () => {
  // App State
  const state = {
    models: [],
    loadedModelStatus: null,
    gpuStats: [],
    selectedModel: null,
    logStreamPaused: false,
    logEntries: [],
    wsLogs: null,
    wsStats: null,
  };

  // Toast Notifications
  function showToast(message, isError = false) {
    const container = document.getElementById('toastContainer');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = 'toast';
    if (isError) toast.style.borderColor = '#71717a';
    toast.textContent = message;

    container.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(-10px)';
      setTimeout(() => toast.remove(), 200);
    }, 3500);
  }

  // Navigation Setup
  const navItems = document.querySelectorAll('.nav-item');
  const viewSections = document.querySelectorAll('.view-section');
  const detailPanel = document.getElementById('detailPanel');
  const btnClosePanel = document.getElementById('btnClosePanel');
  const panelTabs = document.querySelectorAll('.panel-tab');

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

  panelTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetPtab = tab.getAttribute('data-ptab');
      panelTabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');

      document.getElementById('panelTabDetails').style.display = targetPtab === 'details' ? 'block' : 'none';
      document.getElementById('panelTabActivity').style.display = targetPtab === 'activity' ? 'block' : 'none';
    });
  });

  btnClosePanel?.addEventListener('click', () => {
    detailPanel.classList.add('hidden');
    document.querySelectorAll('#modelsTableBody tr').forEach(r => r.classList.remove('selected'));
  });

  // Modal Handlers
  const addModelModal = document.getElementById('addModelModal');
  const manifestModal = document.getElementById('manifestModal');
  const selectSource = document.getElementById('selectSource');
  const groupLocalPath = document.getElementById('groupLocalPath');

  const openAddModelModal = () => addModelModal.classList.remove('hidden');
  const closeAddModelModal = () => addModelModal.classList.add('hidden');

  document.getElementById('btnOpenAddModel')?.addEventListener('click', openAddModelModal);
  document.getElementById('btnOpenAddModel2')?.addEventListener('click', openAddModelModal);
  document.getElementById('btnCloseAddModel')?.addEventListener('click', closeAddModelModal);
  document.getElementById('btnCancelAddModel')?.addEventListener('click', closeAddModelModal);

  selectSource?.addEventListener('change', () => {
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
      return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch (e) {
      return isoStr;
    }
  }

  // Fetch Models & Process Status
  async function loadModelsData() {
    try {
      const [modelsResp, psResp] = await Promise.all([
        fetch('/models').then(r => r.ok ? r.json() : []),
        fetch('/ps').then(r => r.ok ? r.json() : []),
      ]);

      state.models = Array.isArray(modelsResp) ? modelsResp : [];
      state.loadedModelStatus = (Array.isArray(psResp) && psResp.length > 0) ? psResp[0] : null;

      renderModelsView();
    } catch (err) {
      console.error('Error fetching models data:', err);
    }
  }

  function renderModelsView() {
    const tbody = document.getElementById('modelsTableBody');
    const tableCount = document.getElementById('modelsTableCount');
    const activeModelVal = document.getElementById('activeModelVal');
    const activeModelSub = document.getElementById('activeModelSub');
    const totalModelsVal = document.getElementById('totalModelsVal');
    const totalSizeSub = document.getElementById('totalSizeSub');

    if (!tbody) return;

    totalModelsVal.textContent = state.models.length;
    tableCount.textContent = `${state.models.length} MODELS`;

    const totalBytes = state.models.reduce((sum, m) => sum + (m.size_bytes || 0), 0);
    totalSizeSub.textContent = `Total Size: ${formatBytes(totalBytes)}`;

    if (state.loadedModelStatus && state.loadedModelStatus.loaded) {
      activeModelVal.textContent = state.loadedModelStatus.name;
      activeModelSub.textContent = `${state.loadedModelStatus.gpu_memory_mb.toFixed(1)} MB VRAM • Uptime ${state.loadedModelStatus.uptime_seconds.toFixed(0)}s`;
    } else {
      activeModelVal.textContent = 'NONE';
      activeModelSub.textContent = '0 MB GPU VRAM';
    }

    if (state.models.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-muted); padding: 24px;">No local model manifests found in ~/.apah/models/. Use "+ ADD MODEL" to pull one.</td></tr>`;
      return;
    }

    const searchTerm = (document.getElementById('searchModels')?.value || '').toLowerCase();
    const filteredModels = state.models.filter(m => 
      m.name.toLowerCase().includes(searchTerm) || (m.quant && m.quant.toLowerCase().includes(searchTerm))
    );

    tbody.innerHTML = filteredModels.map(m => {
      const isLoaded = state.loadedModelStatus && (
        state.loadedModelStatus.name === m.name || 
        state.loadedModelStatus.name === `${m.name}:${m.version}`
      );
      const statusBadge = isLoaded 
        ? `<span class="badge badge-mono-active">LOADED</span>` 
        : `<span class="badge badge-mono-idle">UNLOADED</span>`;

      return `
        <tr data-model-name="${m.name}" data-version="${m.version}">
          <td><strong>${m.name}</strong></td>
          <td>${m.version || 'v1.0.0'}</td>
          <td>${m.quant || 'none'}</td>
          <td>${formatBytes(m.size_bytes)}</td>
          <td>${statusBadge}</td>
          <td>${formatDate(m.pulled_at)}</td>
          <td><button class="btn btn-sm">DETAILS</button></td>
        </tr>
      `;
    }).join('');

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

    document.getElementById('panelModelName').textContent = modelObj.name;
    document.getElementById('panelVersion').textContent = modelObj.version || 'v1.0.0';
    document.getElementById('panelQuant').textContent = modelObj.quant || 'none';
    document.getElementById('panelSize').textContent = formatBytes(modelObj.size_bytes);
    document.getElementById('panelSha256').textContent = modelObj.checksum_sha256 ? modelObj.checksum_sha256.substring(0, 16) + '...' : '-';

    const isLoaded = state.loadedModelStatus && (
      state.loadedModelStatus.name === modelObj.name || 
      state.loadedModelStatus.name === `${modelObj.name}:${modelObj.version}`
    );
    const badgeContainer = document.getElementById('panelStatusBadge');
    const loadBtn = document.getElementById('btnPanelLoadUnload');

    if (isLoaded) {
      badgeContainer.innerHTML = `<span class="badge badge-mono-active">LOADED</span>`;
      loadBtn.textContent = 'UNLOAD MODEL';
    } else {
      badgeContainer.innerHTML = `<span class="badge badge-mono-idle">UNLOADED</span>`;
      loadBtn.textContent = 'LOAD MODEL';
    }

    detailPanel.classList.remove('hidden');
  }

  // Load / Unload / Manifest / Copy Actions
  document.getElementById('btnPanelLoadUnload')?.addEventListener('click', async () => {
    if (!state.selectedModel) return;
    const isLoaded = state.loadedModelStatus && (
      state.loadedModelStatus.name === state.selectedModel.name || 
      state.loadedModelStatus.name === `${state.selectedModel.name}:${state.selectedModel.version}`
    );

    const btn = document.getElementById('btnPanelLoadUnload');
    btn.disabled = true;

    if (isLoaded) {
      try {
        const res = await fetch('/unload', { method: 'POST' }).then(r => r.json());
        showToast(res.message || 'Model unloaded successfully.');
        await loadModelsData();
        if (state.selectedModel) selectModelRow(state.selectedModel, null);
      } catch (err) {
        showToast('Failed to unload model: ' + err.message, true);
      } finally {
        btn.disabled = false;
      }
    } else {
      try {
        const payload = { model_path: state.selectedModel.name };
        const res = await fetch('/load', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        }).then(r => r.json());

        if (res.error) {
          showToast('Load failed: ' + (res.error.message || res.error), true);
        } else {
          showToast(res.message || 'Model loaded successfully.');
          await loadModelsData();
          if (state.selectedModel) selectModelRow(state.selectedModel, null);
        }
      } catch (err) {
        showToast('Failed to load model: ' + err.message, true);
      } finally {
        btn.disabled = false;
      }
    }
  });

  document.getElementById('btnPanelShowManifest')?.addEventListener('click', () => {
    if (!state.selectedModel) return;
    document.getElementById('manifestModalTitle').textContent = `MANIFEST: ${state.selectedModel.name}`;
    document.getElementById('manifestJsonContent').textContent = JSON.stringify(state.selectedModel, null, 2);
    manifestModal.classList.remove('hidden');
  });

  document.getElementById('btnPanelCopyName')?.addEventListener('click', () => {
    if (!state.selectedModel) return;
    navigator.clipboard.writeText(state.selectedModel.name);
    showToast(`Copied '${state.selectedModel.name}' to clipboard`);
  });

  // Form Add / Pull Model Submission
  document.getElementById('formAddModel')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const modelName = document.getElementById('inputModelName').value.trim();
    const source = selectSource.value;
    const localPath = document.getElementById('inputLocalPath').value.trim();
    const revision = document.getElementById('inputRevision').value.trim();

    const btnSubmit = document.getElementById('btnSubmitPull');
    btnSubmit.disabled = true;
    btnSubmit.textContent = 'PULLING...';

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
        showToast('Pull failed: ' + res.detail, true);
      } else {
        showToast(`Successfully pulled '${modelName}'`);
        closeAddModelModal();
        await loadModelsData();
      }
    } catch (err) {
      showToast('Error during pull: ' + err.message, true);
    } finally {
      btnSubmit.disabled = false;
      btnSubmit.textContent = 'PULL MODEL';
    }
  });

  // Settings view
  async function loadConfigData() {
    try {
      const cfg = await fetch('/config').then(r => r.json());
      document.getElementById('settingIdleTimeout').value = cfg.idle_timeout_seconds;
      document.getElementById('settingAutoUnload').checked = cfg.auto_unload_enabled;

      const airgapBadge = document.getElementById('cfgAirgapStatus');
      airgapBadge.innerHTML = cfg.airgap_mode_enabled 
        ? `<span class="badge badge-mono-active">ENABLED</span>` 
        : `<span class="badge badge-mono-idle">DISABLED</span>`;

      document.getElementById('cfgAuditContent').textContent = cfg.audit_log_content ? 'Full Content' : 'Metadata Only';
      document.getElementById('cfgAuditPath').textContent = cfg.audit_log_path;
    } catch (err) {
      console.error('Error fetching config:', err);
    }
  }

  document.getElementById('btnSaveConfig')?.addEventListener('click', async () => {
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

      showToast('Configuration updated successfully.');
    } catch (err) {
      showToast('Failed to save config: ' + err.message, true);
    }
  });

  document.getElementById('searchModels')?.addEventListener('input', renderModelsView);

  // WebSockets Setup
  function connectLogsWebSocket() {
    const wsProtocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${location.host}/ws/logs`;

    state.wsLogs = new WebSocket(wsUrl);
    const countBadge = document.getElementById('logCountBadge');

    state.wsLogs.onopen = () => {
      console.log('Connected to /ws/logs');
    };

    state.wsLogs.onmessage = (event) => {
      if (state.logStreamPaused) return;

      const rawLine = event.data;
      state.logEntries.push(rawLine);
      if (countBadge) countBadge.textContent = `${state.logEntries.length} EVENTS`;

      renderLogs();
    };

    state.wsLogs.onclose = () => {
      setTimeout(connectLogsWebSocket, 2000);
    };
  }

  function renderLogs() {
    const logContainer = document.getElementById('logContainer');
    if (!logContainer) return;

    const filterType = document.getElementById('logTypeFilter').value;
    const filterSearch = (document.getElementById('logSearch')?.value || '').toLowerCase();

    const filtered = state.logEntries.filter(line => {
      if (filterType !== 'ALL' && !line.includes(`"event_type": "${filterType}"`)) return false;
      if (filterSearch && !line.toLowerCase().includes(filterSearch)) return false;
      return true;
    });

    logContainer.textContent = filtered.join('\n');
    logContainer.scrollTop = logContainer.scrollHeight;
  }

  document.getElementById('logTypeFilter')?.addEventListener('change', renderLogs);
  document.getElementById('logSearch')?.addEventListener('input', renderLogs);
  document.getElementById('btnClearLogs')?.addEventListener('click', () => {
    state.logEntries = [];
    if (document.getElementById('logCountBadge')) document.getElementById('logCountBadge').textContent = '0 EVENTS';
    if (document.getElementById('logContainer')) document.getElementById('logContainer').textContent = '';
  });

  document.getElementById('btnToggleLogPause')?.addEventListener('click', function() {
    state.logStreamPaused = !state.logStreamPaused;
    this.textContent = state.logStreamPaused ? 'RESUME STREAM' : 'PAUSE STREAM';
  });

  function connectStatsWebSocket() {
    const wsProtocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${location.host}/ws/stats`;

    state.wsStats = new WebSocket(wsUrl);

    state.wsStats.onopen = () => {
      const dot = document.getElementById('serverStatusDot');
      const text = document.getElementById('serverStatusText');
      if (dot) dot.className = 'dot active';
      if (text) text.textContent = 'ONLINE';
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
      const dot = document.getElementById('serverStatusDot');
      const text = document.getElementById('serverStatusText');
      if (dot) dot.className = 'dot';
      if (text) text.textContent = 'DISCONNECTED';
      setTimeout(connectStatsWebSocket, 2000);
    };
  }

  function updateTelemetryUI(data) {
    const server = data.server || {};
    const gpuList = data.gpu || [];
    const sched = data.scheduler || {};

    let totalUsedMb = 0;
    let totalMaxMb = 0;
    gpuList.forEach(g => {
      totalUsedMb += (g.memory_used_mb || 0);
      totalMaxMb += (g.memory_total_mb || 0);
    });

    if (totalMaxMb === 0 && server.gpu_memory_mb) {
      totalUsedMb = server.gpu_memory_mb;
      totalMaxMb = 24576;
    }

    const topbarVram = document.getElementById('topbarVramText');
    if (topbarVram) topbarVram.textContent = `${totalUsedMb.toFixed(0)} / ${totalMaxMb.toFixed(0)} MB`;

    const pct = totalMaxMb > 0 ? (totalUsedMb / totalMaxMb) * 100 : 0;
    const sidebarVramBar = document.getElementById('sidebarVramBar');
    const sidebarVramVal = document.getElementById('sidebarVramVal');
    if (sidebarVramBar) sidebarVramBar.style.width = `${pct.toFixed(1)}%`;
    if (sidebarVramVal) sidebarVramVal.textContent = `${(totalUsedMb/1024).toFixed(1)} / ${(totalMaxMb/1024).toFixed(1)} GB (${pct.toFixed(0)}%)`;

    const avgUtil = gpuList.length > 0 ? (gpuList.reduce((acc, g) => acc + g.utilization_pct, 0) / gpuList.length) : 0;
    const avgTemp = gpuList.length > 0 ? (gpuList.reduce((acc, g) => acc + g.temperature_c, 0) / gpuList.length) : 0;
    const telemetryVal = document.getElementById('gpuTelemetryVal');
    const tempSub = document.getElementById('gpuTempSub');
    if (telemetryVal) telemetryVal.textContent = `${avgUtil.toFixed(0)}% UTIL`;
    if (tempSub) tempSub.textContent = `${avgTemp.toFixed(0)} °C Avg Temp`;

    const gpuGrid = document.getElementById('gpuCardsGrid');
    if (gpuGrid) {
      if (gpuList.length > 0) {
        gpuGrid.innerHTML = gpuList.map(g => `
          <div class="card">
            <div class="card-title">GPU ${g.index}: ${g.name}</div>
            <div class="card-val">${g.utilization_pct}% UTIL</div>
            <div class="progress-track">
              <div class="progress-bar" style="width: ${((g.memory_used_mb/g.memory_total_mb)*100).toFixed(1)}%;"></div>
            </div>
            <div class="card-sub">${g.memory_used_mb.toFixed(0)} / ${g.memory_total_mb.toFixed(0)} MB VRAM • ${g.temperature_c} °C</div>
          </div>
        `).join('');
      } else {
        gpuGrid.innerHTML = `
          <div class="card">
            <div class="card-title">GPU DEVICE</div>
            <div class="card-val" style="font-size: 1.1rem;">CPU / VIRTUAL MODE</div>
            <div class="card-sub">No NVML GPU detected</div>
          </div>
        `;
      }
    }

    if (document.getElementById('perfActiveBatch')) document.getElementById('perfActiveBatch').textContent = sched.active_batch_size || 0;
    if (document.getElementById('perfWaitQueue')) document.getElementById('perfWaitQueue').textContent = sched.waiting_queue_length || 0;
    if (document.getElementById('perfThroughput')) document.getElementById('perfThroughput').textContent = `${(sched.aggregate_throughput_tok_s || 0).toFixed(1)} TOK/S`;
    if (document.getElementById('perfTotalTokens')) document.getElementById('perfTotalTokens').textContent = (sched.total_tokens_generated || 0).toLocaleString();
  }

  // Initialization
  loadModelsData();
  loadConfigData();
  connectLogsWebSocket();
  connectStatsWebSocket();

  setInterval(loadModelsData, 5000);
});
