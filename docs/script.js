/* Apah Landing Page Interactions: Theme Toggle & Copy Buttons */

document.addEventListener('DOMContentLoaded', () => {
  const lightBtn = document.getElementById('lightbtn');
  const storedTheme = localStorage.getItem('apah-theme');

  if (storedTheme) {
    document.documentElement.setAttribute('data-theme', storedTheme);
    if (lightBtn) {
      lightBtn.textContent = storedTheme === 'light' ? '🌙 dark mode' : '☀ light mode';
    }
  }

  if (lightBtn) {
    lightBtn.addEventListener('click', () => {
      const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
      const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

      document.documentElement.setAttribute('data-theme', newTheme);
      localStorage.setItem('apah-theme', newTheme);
      lightBtn.textContent = newTheme === 'light' ? '🌙 dark mode' : '☀ light mode';
    });
  }
});

function copyCode(btn, codeId) {
  const codeEl = document.getElementById(codeId);
  if (!codeEl) return;

  const text = codeEl.innerText.trim();
  navigator.clipboard.writeText(text).then(() => {
    const orig = btn.innerText;
    btn.innerText = '[copied!]';
    btn.style.color = 'var(--amber)';
    btn.style.borderColor = 'var(--amber)';
    setTimeout(() => {
      btn.innerText = orig;
      btn.style.color = '';
      btn.style.borderColor = '';
    }, 2000);
  });
}
