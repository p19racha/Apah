/* Apah Landing Page Interactions: Copy buttons & Theme Toggle (Zero Dependencies) */

document.addEventListener('DOMContentLoaded', () => {
  // 1. Theme Toggle Setup
  const themeToggleBtn = document.getElementById('theme-toggle');
  const storedTheme = localStorage.getItem('apah-theme');
  
  if (storedTheme) {
    document.documentElement.setAttribute('data-theme', storedTheme);
    if (themeToggleBtn) {
      themeToggleBtn.textContent = storedTheme === 'light' ? '[theme: dark]' : '[theme: light]';
    }
  }

  if (themeToggleBtn) {
    themeToggleBtn.addEventListener('click', () => {
      const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
      const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
      
      document.documentElement.setAttribute('data-theme', newTheme);
      localStorage.setItem('apah-theme', newTheme);
      themeToggleBtn.textContent = newTheme === 'light' ? '[theme: dark]' : '[theme: light]';
    });
  }
});

// 2. Clipboard Copy Helper
function copyCode(buttonElement, targetId) {
  const codeElement = document.getElementById(targetId);
  if (!codeElement) return;

  const textToCopy = codeElement.innerText.trim();
  
  navigator.clipboard.writeText(textToCopy).then(() => {
    const originalText = buttonElement.innerText;
    buttonElement.innerText = '[copied!]';
    buttonElement.style.borderColor = 'var(--accent)';
    buttonElement.style.color = 'var(--accent)';
    
    setTimeout(() => {
      buttonElement.innerText = originalText;
      buttonElement.style.borderColor = '';
      buttonElement.style.color = '';
    }, 2000);
  }).catch((err) => {
    console.error('Failed to copy code: ', err);
  });
}
