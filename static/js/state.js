// ===== 状态管理 =====
let currentMode = "preset"; // preset, clone
let currentLang = "zh";
let selectedVoiceId = null;
let savedVoices = [];

// 录音相关
let mediaRecorder = null;
let audioChunks = [];
let recordedBlob = null;
let selectedFile = null;
let recordingStartTime = null;
let timerInterval = null;

// 单句重新生成相关
let sentenceAudios = []; // 每句音频 base64 数组
let sentenceTexts = []; // 每句文本数组
let sentenceInstructs = []; // 每句情感指令（仅 preset 模式有意义）
let sentenceVoiceConfigs = []; // 每句声音配置（null=默认，{type,speaker/voice_id,label}=覆盖）
let sentenceParagraphBreaks = []; // 段落边界标记：true=该句是段落开头
let sentenceSpeedRates = []; // 每句语速倍率（默认 1.0，范围 0.5~2.0）
let lastGenerateParams = null; // {mode, speaker, language, instruct, voice_id, clone_prompt_id}
let clonePromptId = null; // clone 模式的 session ID

// 分句预览模式（无音频，纯文本编辑）
let isPreviewing = false;

// 撤销栈
let undoStack = []; // [{index, audio, text}]

// 生成进度
let generatingProgress = -1; // 生成中：已完成句数（0-based index of current），非生成：-1

// 单句试听
let sentencePreviewIndex = -1;

// 统计数据（数值，语言无关）
let lastStatsData = null; // {char_count, sentence_count, elapsed, avg_per_char}

function renderStats() {
  if (!lastStatsData) return;
  const s = lastStatsData;
  if (s.sentence_count != null) {
    document.getElementById("stats-chars").textContent =
      `${s.char_count} ${t("stats.chars")} · ${s.sentence_count} ${t("stats.sentences")} · ${s.elapsed}s`;
  } else {
    document.getElementById("stats-chars").textContent =
      `${s.char_count} ${t("stats.chars")} · ${s.elapsed}s`;
  }
  document.getElementById("stats-speed").textContent =
    `${s.avg_per_char}s/${t("stats.chars")}`;
}

// 根据当前 sentenceTexts 重算统计并更新显示
function refreshStatsFromSentences() {
  if (!lastStatsData || !sentenceTexts.length) return;
  const charCount = sentenceTexts.join("").length;
  lastStatsData.char_count = charCount;
  lastStatsData.sentence_count = sentenceTexts.length;
  renderStats();
  // 同步右上角字数
  document.getElementById("char-count").innerHTML =
    `${charCount} <span data-i18n="stats.chars">${t("stats.chars")}</span>`;
}

// 生成历史
let generationHistory = [];

function loadHistory() {
  try {
    generationHistory = JSON.parse(
      localStorage.getItem("vibevoice_history") || "[]",
    );
  } catch (e) {
    generationHistory = [];
  }
}

function saveToHistory(text, mode) {
  generationHistory.unshift({
    id: Date.now().toString(),
    text: text.slice(0, 200),
    mode,
    timestamp: Date.now(),
  });
  if (generationHistory.length > 50) generationHistory.length = 50;
  localStorage.setItem("vibevoice_history", JSON.stringify(generationHistory));
  renderHistory();
}

function renderHistory() {
  const container = document.getElementById("history-list");
  if (!container) return;
  if (generationHistory.length === 0) {
    container.innerHTML = `<div class="text-center text-charcoal/50 text-sm py-4">${t("history.empty")}</div>`;
    return;
  }
  container.innerHTML = generationHistory
    .map((item) => {
      const preview =
        item.text.length > 40 ? item.text.slice(0, 40) + "..." : item.text;
      const modeKey =
        "tab." + (item.mode === "saved_voice" ? "library" : item.mode);
      const modeLabel = t(modeKey);
      const timeStr = formatTimeAgo(item.timestamp);
      return `<div class="history-item" onclick="loadHistoryItem('${item.id}')">
      <div class="text-sm truncate" style="color:#2d3748">${escapeHtmlSimple(preview)}</div>
      <div class="flex items-center justify-between mt-1">
        <span class="text-xs" style="color:#a0aec0">${modeLabel}</span>
        <span class="text-xs" style="color:#cbd5e0">${timeStr}</span>
      </div>
    </div>`;
    })
    .join("");
}

function loadHistoryItem(id) {
  const item = generationHistory.find((h) => h.id === id);
  if (!item) return;
  document.getElementById("text-input").value = item.text;
  updateCharCount();
}

function formatTimeAgo(ts) {
  const diff = Date.now() - ts;
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "<1m";
  if (mins < 60) return mins + "m";
  const hours = Math.floor(mins / 60);
  if (hours < 24) return hours + "h";
  const days = Math.floor(hours / 24);
  return days + "d";
}

// 简单HTML转义（不依赖DOM，state.js加载时editor.js的escapeHtml尚未定义）
function escapeHtmlSimple(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// 句间停顿
let pausePaceMultiplier = 1.0;
let decodedPcmCache = []; // 缓存解码后的 PCM，避免重复 atob

// ===== 会话持久化 (IndexedDB) =====
const SESSION_DB = "vibevoice_session";
const SESSION_STORE = "session";
const SESSION_KEY = "current";

function openSessionDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(SESSION_DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(SESSION_STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function saveSession() {
  if (!sentenceAudios.length) return;
  try {
    const db = await openSessionDB();
    const tx = db.transaction(SESSION_STORE, "readwrite");
    tx.objectStore(SESSION_STORE).put(
      {
        sentenceAudios,
        sentenceTexts,
        sentenceInstructs,
        sentenceVoiceConfigs,
        sentenceSpeedRates,
        lastGenerateParams,
        clonePromptId,
        currentSubtitles,
        pausePaceMultiplier,
        inputText:
          sentenceTexts.length > 0
            ? sentenceTexts.join("")
            : document.getElementById("text-input").value,
        statsData: lastStatsData,
        timestamp: Date.now(),
      },
      SESSION_KEY,
    );
    db.close();
  } catch (e) {
    console.warn("saveSession failed:", e);
  }
}

async function loadSession() {
  try {
    const db = await openSessionDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(SESSION_STORE, "readonly");
      const req = tx.objectStore(SESSION_STORE).get(SESSION_KEY);
      req.onsuccess = () => {
        db.close();
        resolve(req.result || null);
      };
      req.onerror = () => {
        db.close();
        resolve(null);
      };
    });
  } catch (e) {
    return null;
  }
}

async function clearSession() {
  try {
    const db = await openSessionDB();
    const tx = db.transaction(SESSION_STORE, "readwrite");
    tx.objectStore(SESSION_STORE).delete(SESSION_KEY);
    db.close();
  } catch (e) {}
}

async function restoreSession() {
  const session = await loadSession();
  if (!session || !session.sentenceAudios || !session.sentenceAudios.length)
    return;
  // 恢复状态
  sentenceAudios = session.sentenceAudios;
  sentenceTexts = session.sentenceTexts;
  lastGenerateParams = session.lastGenerateParams;
  sentenceInstructs =
    session.sentenceInstructs ||
    sentenceTexts.map(() => lastGenerateParams?.instruct || "");
  sentenceVoiceConfigs =
    session.sentenceVoiceConfigs || sentenceTexts.map(() => null);
  sentenceSpeedRates =
    session.sentenceSpeedRates || sentenceTexts.map(() => 1.0);
  clonePromptId = session.clonePromptId;
  currentSubtitles = session.currentSubtitles;
  pausePaceMultiplier = session.pausePaceMultiplier ?? 1.0;
  decodedPcmCache = [];
  // 恢复输入框
  if (session.inputText) {
    document.getElementById("text-input").value = session.inputText;
    updateCharCount();
  }
  // 重建音频
  const merged = mergeAllSentenceAudios();
  currentSubtitles = merged.subtitles;
  audioElement.src = URL.createObjectURL(merged.blob);
  loadWaveform();
  // 恢复 stats 显示
  if (session.statsData) {
    lastStatsData = session.statsData;
  }
  refreshStatsFromSentences();
  // 显示播放器和句子视图（始终进入句子编辑器）
  document.getElementById("player-section").classList.remove("hidden");
  selectedSentenceIndex = -1;
  showSentenceEditorView();
}
