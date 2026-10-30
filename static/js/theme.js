// Theme switcher — persists choice in localStorage.
(function () {
  const STORAGE_KEY = 'metarecom-theme';
  const VALID = ['dark', 'light', 'gradient'];

  function applyTheme(theme) {
    if (!VALID.includes(theme)) theme = 'dark';
    document.documentElement.setAttribute('data-theme', theme);
    document.querySelectorAll('.theme-switcher button').forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.themeSet === theme);
    });
    try { localStorage.setItem(STORAGE_KEY, theme); } catch (e) {}
  }

  // Init from storage (or current attribute)
  let saved = null;
  try { saved = localStorage.getItem(STORAGE_KEY); } catch (e) {}
  applyTheme(saved || document.documentElement.getAttribute('data-theme') || 'dark');

  // Wire up buttons
  document.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-theme-set]');
    if (!btn) return;
    applyTheme(btn.dataset.themeSet);
  });
})();
