"""Small dependency-free administration page for the local plugin catalog."""

PLUGIN_MARKETPLACE_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OSagent Plugin Marketplace</title>
  <style>
    :root { color-scheme: light dark; font: 14px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; --bg: #fff; --panel: #f7f7f8; --line: #e5e7eb; --muted: #6b7280; --text: #111827; --accent: #2563eb; }
    @media (prefers-color-scheme: dark) { :root { --bg: #171717; --panel: #242424; --line: #3f3f46; --muted: #a1a1aa; --text: #f4f4f5; --accent: #60a5fa; } }
    * { box-sizing: border-box; }
    body { max-width: 1120px; margin: 0 auto; padding: 30px 28px 40px; background: var(--bg); color: var(--text); }
    header { display: flex; align-items: flex-start; justify-content: space-between; gap: 20px; margin-bottom: 26px; }
    h1 { margin: 0 0 5px; font-size: 22px; letter-spacing: 0; }
    .subtitle { margin: 0; color: var(--muted); }
    .toolbar { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }
    input { min-width: 220px; flex: 1; padding: 10px 12px; color: var(--text); background: var(--bg); border: 1px solid var(--line); border-radius: 8px; outline: none; }
    input:focus { border-color: var(--accent); box-shadow: 0 0 0 2px color-mix(in srgb, var(--accent) 20%, transparent); }
    button { padding: 9px 13px; color: var(--text); background: var(--bg); border: 1px solid var(--line); border-radius: 8px; cursor: pointer; font: inherit; }
    button:hover { background: var(--panel); }
    button.primary { color: #fff; background: var(--accent); border-color: var(--accent); }
    button:disabled { cursor: wait; opacity: .6; }
    table { width: 100%; border-collapse: separate; border-spacing: 0 8px; }
    th { padding: 0 12px 4px; color: var(--muted); font-size: 12px; font-weight: 500; text-align: left; }
    td { padding: 14px 12px; background: var(--panel); border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); text-align: left; vertical-align: middle; }
    td:first-child { border-left: 1px solid var(--line); border-radius: 8px 0 0 8px; }
    td:last-child { border-right: 1px solid var(--line); border-radius: 0 8px 8px 0; }
    .status { color: var(--muted); font-weight: 600; text-transform: capitalize; }
    .status::before { content: ""; display: inline-block; width: 7px; height: 7px; margin: 0 7px 1px 0; border-radius: 50%; background: currentColor; }
    .actions { display: flex; gap: 6px; flex-wrap: wrap; justify-content: flex-end; }
    #message { min-height: 1.4em; margin: 8px 0; color: var(--muted); }
    @media (max-width: 700px) { body { padding: 22px 14px; } th:nth-child(3), td:nth-child(3) { display: none; } td { padding: 12px 8px; } }
  </style>
</head>
<body>
  <header><div><h1>OSagent Plugin Marketplace</h1><p class="subtitle">Verified extensions for your OSagent workspace</p></div></header>
  <div class="toolbar">
    <input id="token" type="password" autocomplete="off" placeholder="Service API key">
    <input id="query" type="search" placeholder="Search plugins">
    <button id="refresh" type="button" title="Reload plugin catalog">Refresh</button>
  </div>
  <div id="message" role="status"></div>
  <table>
    <thead><tr><th>Plugin</th><th>Status</th><th>Description</th><th>Actions</th></tr></thead>
    <tbody id="plugins"><tr><td colspan="4">Enter the service key and refresh.</td></tr></tbody>
  </table>
  <script>
    const token = document.querySelector('#token');
    const query = document.querySelector('#query');
    const message = document.querySelector('#message');
    const tbody = document.querySelector('#plugins');
    const base = window.location.pathname.startsWith('/osagent/') ? '/osagent' : '';
    token.value = sessionStorage.getItem('osagent-service-key') || '';
    token.addEventListener('change', () => sessionStorage.setItem('osagent-service-key', token.value));
    const headers = () => ({'Authorization': `Bearer ${token.value}`, 'Content-Type': 'application/json'});
    async function api(path, options = {}) {
      const response = await fetch(path, { ...options, headers: {...headers(), ...(options.headers || {})} });
      const payload = await response.json().catch(() => ({ok:false, error: response.statusText}));
      if (!response.ok || !payload.ok) throw new Error(payload.error || `HTTP ${response.status}`);
      return payload.data;
    }
    function setMessage(text, error = false) { message.textContent = text; message.style.color = error ? '#c33' : ''; }
    function button(label, title, handler) {
      const el = document.createElement('button'); el.type = 'button'; el.textContent = label; el.title = title;
      if (label === 'Install' || label === 'Enable') el.className = 'primary';
      el.addEventListener('click', handler); return el;
    }
    function render(items) {
      tbody.replaceChildren();
      if (!items.length) { tbody.innerHTML = '<tr><td colspan="4">No plugins found.</td></tr>'; return; }
      for (const item of items) {
        const row = document.createElement('tr');
        row.innerHTML = `<td><strong>${escapeHtml(item.name)}</strong><br><small>v${escapeHtml(item.version)}</small></td>` +
          `<td class="status">${escapeHtml(item.status)}</td><td>${escapeHtml(item.description || '')}</td>`;
        const actions = document.createElement('td'); actions.className = 'actions';
        if (!item.installed) actions.append(button('Install', 'Install verified plugin bundle', () => run(item.name, 'install')));
        else actions.append(button(item.enabled ? 'Disable' : 'Enable', item.enabled ? 'Disable plugin' : 'Enable plugin', () => run(item.name, item.enabled ? 'disable' : 'enable')));
        if (item.installed) actions.append(button('Uninstall', 'Remove installed plugin bundle', () => run(item.name, 'uninstall')));
        row.append(actions); tbody.append(row);
      }
    }
    function escapeHtml(value) { return String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
    async function load() { try { setMessage('Loading...'); const data = await api(`${base}/v1/plugins?query=${encodeURIComponent(query.value)}`); render(data.plugins || []); setMessage(`${(data.plugins || []).length} plugin(s)`); } catch (e) { setMessage(e.message, true); } }
    async function run(name, operation) {
      try { setMessage(`${operation} ${name}...`); const path = operation === 'install' ? `${base}/v1/plugins/${encodeURIComponent(name)}/install` : operation === 'uninstall' ? `${base}/v1/plugins/${encodeURIComponent(name)}` : `${base}/v1/plugins/${encodeURIComponent(name)}/${operation}`; await api(path, {method: operation === 'uninstall' ? 'DELETE' : 'POST', body: '{}'}); await load(); } catch (e) { setMessage(e.message, true); }
    }
    document.querySelector('#refresh').addEventListener('click', load);
    query.addEventListener('keydown', e => { if (e.key === 'Enter') load(); });
  </script>
</body>
</html>"""
