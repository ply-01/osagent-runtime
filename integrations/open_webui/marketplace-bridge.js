/* Injected by the OSagent gateway into Open WebUI's workspace shell. */
(function () {
  'use strict';
  const entryId = 'osagent-marketplace-entry';
  function installEntry() {
    if (!location.pathname.startsWith('/workspace/')) return;
    // Newer OSagent Open WebUI builds include the route natively. Keep this
    // bridge as a compatibility fallback for the stock container image.
    if (document.querySelector('a[href="/workspace/plugins"]')) return;
    if (document.getElementById(entryId)) return;
    const anchor = document.querySelector('a[href="/workspace/tools"], a[href="/workspace/skills"]');
    if (!anchor || !anchor.parentElement) return;
    const entry = anchor.cloneNode(true);
    entry.id = entryId; entry.href = '/workspace/plugins';
    entry.setAttribute('aria-label', 'OSagent Plugin Marketplace');
    const labels = entry.querySelectorAll('span');
    if (labels.length) labels[0].textContent = '插件';
    if (labels.length > 1) labels[1].remove();
    anchor.parentElement.parentElement?.appendChild(entry);
  }

  new MutationObserver(installEntry).observe(document.documentElement, {childList: true, subtree: true});
  installEntry();
})();
