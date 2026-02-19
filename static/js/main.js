// ===== 初始化 =====
document.addEventListener("DOMContentLoaded", () => {
  currentLang = localStorage.getItem("lang") || "zh";
  updateI18n();
  loadSavedVoices();
  loadHistory();
  renderHistory();
  updateCharCount();
  restoreSession(); // 恢复上次会话
});
