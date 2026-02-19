// ===== 模式切换 =====
function switchMode(mode) {
  currentMode = mode;

  // 更新 tab 状态
  document.querySelectorAll(".mode-tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.mode === mode);
  });

  // 显示/隐藏配置面板
  document.getElementById("config-preset").classList.toggle("hidden", mode !== "preset");
  document.getElementById("config-clone").classList.toggle("hidden", mode !== "clone");
  document.getElementById("config-design").classList.toggle("hidden", mode !== "design");
  document.getElementById("config-library").classList.toggle("hidden", mode !== "library");

  // 切换到声音库时刷新列表
  if (mode === "library") {
    renderVoiceList();
  } else {
    // 离开声音库时清除选中
    selectedVoiceId = null;
  }

  // 保存区域：clone 模式下有音频时显示，其他模式隐藏
  if (mode === "clone" && (recordedBlob || selectedFile)) {
    document.getElementById("save-voice-section").classList.remove("hidden");
  } else {
    document.getElementById("save-voice-section").classList.add("hidden");
  }
}

// ===== 页面切换 =====
function showPage(page) {
  document
    .getElementById("api-overlay")
    .classList.toggle("hidden", page !== "api");
}

// ===== 字数统计 =====
let langDetectTimer = null;
function updateCharCount() {
  const text = document.getElementById("text-input").value;
  document.getElementById("char-count").innerHTML =
    `${text.length} <span data-i18n="stats.chars">${t("stats.chars")}</span>`;

  // 文本为空时禁用生成按钮
  const btn = document.getElementById("generate-btn");
  if (!isGenerating) {
    btn.disabled = text.trim().length === 0;
  }

  // 自动检测语言（防抖500ms）
  clearTimeout(langDetectTimer);
  langDetectTimer = setTimeout(() => detectAndSetLanguage(text), 500);
}

function detectAndSetLanguage(text) {
  if (!text.trim()) return;
  const counts = { zh: 0, ja: 0, ko: 0, en: 0 };
  for (const c of text) {
    const code = c.charCodeAt(0);
    if (code >= 0x4e00 && code <= 0x9fff) counts.zh++;
    else if (
      (code >= 0x3040 && code <= 0x30ff) ||
      (code >= 0x31f0 && code <= 0x31ff)
    )
      counts.ja++;
    else if (code >= 0xac00 && code <= 0xd7af) counts.ko++;
    else if ((code >= 0x41 && code <= 0x5a) || (code >= 0x61 && code <= 0x7a))
      counts.en++;
  }
  const total = counts.zh + counts.ja + counts.ko + counts.en;
  if (total < 5) return;
  let lang = null;
  if (counts.ja > 0 && counts.ja >= counts.zh * 0.1) lang = "Japanese";
  else if (counts.ko > total * 0.3) lang = "Korean";
  else if (counts.zh > total * 0.3) lang = "Chinese";
  else if (counts.en > total * 0.5) lang = "English";
  if (lang) {
    ["language-preset", "language-clone", "language-design", "language-library"].forEach((id) => {
      document.getElementById(id).value = lang;
    });
  }
}

// 前端分句（与后端保持一致）
function splitTextToSentences(text, minLength = 10) {
  const pattern = /([。！？；.!?;][。！？；.!?;"\u201C\u201D\u300C\u300D'\u2018\u2019]*|[：:](?=["\u201C\u201D\u300C\u300D'\u2018\u2019])|\n)/;
  const parts = text.split(pattern);

  // split 带捕获组：奇数索引是分隔符，用索引判定（避免 lookahead 在隔离部分失效）
  const rawSentences = [];
  let current = "";
  for (let i = 0; i < parts.length; i++) {
    current += parts[i];
    if (i % 2 === 1) {
      if (current.trim()) {
        rawSentences.push(current.trim());
      }
      current = "";
    }
  }
  if (current.trim()) {
    rawSentences.push(current.trim());
  }

  if (rawSentences.length === 0) {
    return text.trim() ? [text] : [];
  }

  // 合并过短的句子（以冒号结尾的句子强制独立，用于旁白/引语分离）
  const merged = [];
  let buffer = "";
  for (const sentence of rawSentences) {
    buffer += sentence;
    if (buffer.length >= minLength || /[：:]$/.test(buffer)) {
      merged.push(buffer);
      buffer = "";
    }
  }
  if (buffer) {
    if (merged.length > 0) {
      merged[merged.length - 1] += buffer;
    } else {
      merged.push(buffer);
    }
  }
  return merged;
}

// 更新生成进度（DOM操作，不重新渲染）
function updateGeneratingProgress(current) {
  generatingProgress = current;
  const items = document.querySelectorAll(".sentence-editor-item");
  items.forEach((el, i) => {
    el.classList.remove("gen-done", "gen-current");
    if (i < current) {
      el.classList.add("gen-done");
    } else if (i === current) {
      el.classList.add("gen-current");
    }
  });
  // 滚动当前句子到可见区域
  const currentItem = items[current];
  if (currentItem) {
    currentItem.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
}

// 隐藏进度视图
function hideProgressView() {
  generatingProgress = -1;
  const textInput = document.getElementById("text-input");
  const progressView = document.getElementById("progress-view");
  textInput.classList.remove("hidden");
  progressView.style.display = "";
  progressView.style.flexDirection = "";
  progressView.style.overflow = "";
  progressView.classList.add("hidden");
  updateCharCount();
  // 恢复生成按钮（分句预览按钮），隐藏句子工具栏
  const genBtn = document.getElementById("generate-btn");
  genBtn.style.display = "";
  genBtn.onclick = enterPreviewMode;
  genBtn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg><span>${t("btn.previewSentences")}</span>`;
  const toolbar = document.getElementById("sentence-toolbar");
  toolbar.style.display = "none";
  toolbar.classList.add("hidden");
}

// 显示句子编辑视图
let selectedSentenceIndex = -1;

function showSentenceEditorView() {
  const textInput = document.getElementById("text-input");
  const progressView = document.getElementById("progress-view");
  const generating = isGenerating;
  const previewing = isPreviewing;

  // 找出最近一次撤销对应的句子索引
  const lastUndoIndex =
    undoStack.length > 0 ? undoStack[undoStack.length - 1].index : -1;

  // 判断是否为 preset 模式（预编辑时用 currentMode，生成后用 lastGenerateParams）
  const globalIsPreset = previewing
    ? currentMode === "preset"
    : lastGenerateParams && lastGenerateParams.mode === "preset";

  let html =
    `<div style="flex:1;min-height:0;overflow-y:auto" class="scrollbar-thin"><ul class="sentence-editor-list${generating ? " generating" : ""}${previewing ? " previewing" : ""}">`;
  // 插入按钮（仅非生成模式）
  if (!generating) {
    html += `<li class="sentence-insert-row"><button class="sentence-insert-btn" onclick="event.stopPropagation(); showInsertForm(0)" title="${t("btn.addSentence")}">＋</button></li>`;
  }
  sentenceTexts.forEach((text, index) => {
    const isSelected = !generating && index === selectedSentenceIndex;
    const isPreviewPlaying = !generating && !previewing && sentencePreviewIndex === index;
    const hasUndo = !generating && !previewing && index === lastUndoIndex;
    const instruct = sentenceInstructs[index] || "";

    // 生成中的进度 CSS 类
    let genClass = "";
    if (generating) {
      if (index < generatingProgress) genClass = "gen-done";
      else if (index === generatingProgress) genClass = "gen-current";
    }

    // 逐句声音配置判断
    const voiceConfig = sentenceVoiceConfigs[index];
    const effectiveIsPreset = voiceConfig
      ? voiceConfig.type === "preset"
      : globalIsPreset;

    // 声音标签（非生成中时始终显示）
    const voiceLabel = voiceConfig ? voiceConfig.label : t("label.voiceDefault");
    const voiceOverrideClass = voiceConfig ? " voice-override" : "";
    const voiceTag = !generating
      ? `<div class="sentence-voice-tag" id="sent-voice-${index}" onclick="event.stopPropagation(); editSentenceVoice(${index})"><span class="sentence-voice-label">${t("label.voiceLabel")}:</span> <span class="sentence-voice-value${voiceOverrideClass}">${escapeHtml(voiceLabel)}</span> <span class="sentence-voice-edit">✏</span></div>`
      : "";

    // 情感标签：预编辑和生成后都显示（该句使用 preset 声音时）
    const instructTag = !generating && effectiveIsPreset
      ? `<div class="sentence-instruct-tag" id="sent-instruct-${index}" onclick="event.stopPropagation(); editSentenceInstruct(${index})"><span class="sentence-instruct-label">${t("label.instructLabel")}:</span> <span class="sentence-instruct-value">${instruct ? escapeHtml(instruct) : t("label.instructEmpty")}</span> <span class="sentence-instruct-edit">✏</span></div>`
      : "";

    let actionsHtml = "";
    if (generating) {
      actionsHtml = "";
    } else if (previewing) {
      // 预编辑模式：只有删除按钮（无音频，不能试听/重新生成）
      actionsHtml = `<span class="sentence-editor-actions">
                <button class="sentence-del-btn" onclick="event.stopPropagation(); deleteSentence(${index})" title="删除">✕</button>
            </span>`;
    } else {
      // 生成后完整操作按钮
      actionsHtml = `<span class="sentence-editor-actions">
                ${hasUndo ? '<button class="sentence-regen-btn" onclick="event.stopPropagation(); undoRegenerate()" title="' + t("btn.undo") + '" style="border-color:#f6ad55;color:#dd6b20">↩</button>' : ""}
                <button class="sentence-play-btn ${isPreviewPlaying ? "playing-now" : ""}" onclick="event.stopPropagation(); previewSentenceAudio(${index})" title="试听">${isPreviewPlaying ? "⏸" : "▶"}</button>
                <button class="sentence-regen-btn" onclick="event.stopPropagation(); regenerateSentence(${index})">${t("btn.regenerate")}</button>
                <button class="sentence-del-btn" onclick="event.stopPropagation(); deleteSentence(${index})" title="删除">✕</button>
            </span>`;
    }

    const interactAttrs = generating
      ? ""
      : `onclick="selectSentenceItem(${index}, event)" ondblclick="editSentenceItem(${index})"`;

    html += `<li class="sentence-editor-item ${isSelected ? "selected" : ""} ${genClass}"
            id="sent-item-${index}"
            ${interactAttrs}>
            <span class="sentence-editor-index">${index + 1}</span>
            <div style="flex:1;min-width:0">
                <span class="sentence-editor-text" id="sent-text-${index}">${escapeHtml(text)}</span>
                ${voiceTag}
                ${instructTag}
            </div>
            ${actionsHtml}
        </li>`;
    // 插入按钮（仅非生成模式）
    if (!generating) {
      html += `<li class="sentence-insert-row"><button class="sentence-insert-btn" onclick="event.stopPropagation(); showInsertForm(${index + 1})" title="${t("btn.addSentence")}">＋</button></li>`;
    }
  });
  html += "</ul></div>";

  progressView.innerHTML = html;
  textInput.classList.add("hidden");
  progressView.style.display = "flex";
  progressView.style.flexDirection = "column";
  progressView.style.overflow = "hidden";
  progressView.classList.remove("hidden");

  if (generating) {
    // 生成中：不触碰生成按钮（generation.js 已将其改为停止按钮），不显示句子工具栏
    const toolbar = document.getElementById("sentence-toolbar");
    toolbar.style.display = "none";
    toolbar.classList.add("hidden");
  } else if (previewing) {
    // 预编辑模式：显示"生成语音"按钮和"返回编辑"按钮
    const btn = document.getElementById("generate-btn");
    btn.style.display = "";
    btn.disabled = false;
    btn.onclick = generate;
    btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg><span>${t("btn.startGenerate")}</span>`;
    // 显示"返回编辑"在状态消息区
    document.getElementById("status-message").innerHTML =
      `<button onclick="exitPreviewMode()" style="background:none;border:none;cursor:pointer;color:#E07A5F;font-size:13px">← ${t("btn.backToInput")}</button>`;
    // 不显示句子工具栏（无音频，停顿控件无意义）
    const toolbar = document.getElementById("sentence-toolbar");
    toolbar.style.display = "none";
    toolbar.classList.add("hidden");
  } else {
    // 生成后：清除状态消息，隐藏生成按钮，显示句子工具栏
    document.getElementById("status-message").textContent = "";
    document.getElementById("generate-btn").style.display = "none";
    const toolbar = document.getElementById("sentence-toolbar");
    toolbar.classList.remove("hidden");
    toolbar.style.display = "flex";
    // 同步停顿控件值
    document.getElementById("st-pace-label").textContent = t("label.pace");
    document.getElementById("st-pace-value").textContent =
      pausePaceMultiplier === 0
        ? t("label.paceOff")
        : pausePaceMultiplier.toFixed(1) + "x";
    document.getElementById("st-pace-range").value = pausePaceMultiplier;
  }
}

function clearAndRestart() {
  if (!confirm(t("confirm.newProject"))) return;
  finishEditing();
  // 清空所有状态
  sentenceAudios = [];
  sentenceTexts = [];
  sentenceInstructs = [];
  sentenceVoiceConfigs = [];
  decodedPcmCache = [];
  currentSubtitles = null;
  lastGenerateParams = null;
  clonePromptId = null;
  selectedSentenceIndex = -1;
  undoStack = [];
  sentencePreviewIndex = -1;
  lastStatsData = null;
  isPreviewing = false;
  if (_sentencePreviewEndHandler) {
    audioElement.removeEventListener("timeupdate", _sentencePreviewEndHandler);
    _sentencePreviewEndHandler = null;
  }
  // 清空 textarea
  document.getElementById("text-input").value = "";
  // 隐藏播放器
  document.getElementById("player-section").classList.add("hidden");
  // 清除持久化
  clearSession();
  // 回到 textarea 视图
  hideProgressView();
}

function selectSentenceItem(index, event) {
  // 如果点击发生在正在编辑的 contenteditable 内，不处理（让光标自由移动）
  if (event && event.target.closest('[contenteditable="true"]')) return;

  // 如果有正在编辑的句子，先保存
  finishEditing();

  selectedSentenceIndex = selectedSentenceIndex === index ? -1 : index;
  // 更新选中状态
  document.querySelectorAll(".sentence-editor-item").forEach((el, i) => {
    el.classList.toggle("selected", i === selectedSentenceIndex);
  });
}

function editSentenceItem(index) {
  // 如果已经在编辑这句了，不重复处理
  const textEl = document.getElementById(`sent-text-${index}`);
  if (textEl && textEl.contentEditable === "true") return;

  finishEditing();
  selectedSentenceIndex = index;
  document.querySelectorAll(".sentence-editor-item").forEach((el, i) => {
    el.classList.toggle("selected", i === index);
  });
  if (textEl) {
    textEl.contentEditable = "true";
    textEl.focus();
    // 光标放到末尾
    const range = document.createRange();
    range.selectNodeContents(textEl);
    range.collapse(false);
    const sel = window.getSelection();
    sel.removeAllRanges();
    sel.addRange(range);
    // Enter 键完成编辑
    textEl.onkeydown = (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        finishEditing();
      }
    };
  }
}

function finishEditing() {
  let changed = false;
  document
    .querySelectorAll('.sentence-editor-text[contenteditable="true"]')
    .forEach((el) => {
      el.contentEditable = "false";
      el.onkeydown = null;
      // 获取索引并更新 sentenceTexts
      const idMatch = el.id.match(/sent-text-(\d+)/);
      if (idMatch) {
        const idx = parseInt(idMatch[1]);
        const newText = el.textContent.trim();
        if (newText && newText !== sentenceTexts[idx]) {
          sentenceTexts[idx] = newText;
          changed = true;
        }
      }
    });
  if (changed) {
    refreshStatsFromSentences();
    saveSession(); // 文本编辑后持久化
  }
}

// ===== 逐句情感编辑 =====
function editSentenceInstruct(index) {
  const tag = document.getElementById(`sent-instruct-${index}`);
  if (!tag) return;
  if (tag.querySelector("input")) return; // 已在编辑中
  const currentVal = sentenceInstructs[index] || "";
  tag.innerHTML = `<input type="text" class="sentence-instruct-input"
        value="${escapeHtml(currentVal)}"
        placeholder="${t("label.instructEmpty")}"
        onblur="finishInstructEdit(${index}, this)"
        onkeydown="if(event.key==='Enter'){event.preventDefault();this.blur();}if(event.key==='Escape'){this.blur();}">`;
  const input = tag.querySelector("input");
  input.focus();
  input.select();
}

function finishInstructEdit(index, inputEl) {
  const newVal = inputEl.value.trim();
  sentenceInstructs[index] = newVal;
  saveSession();
  showSentenceEditorView();
}

// ===== 逐句声音选择 =====
function getDefaultVoiceLabel() {
  if (isPreviewing) {
    if (currentMode === "preset") {
      const sel = document.getElementById("speaker");
      return sel ? sel.options[sel.selectedIndex].text : "Vivian";
    }
    if (currentMode === "library") {
      const v = savedVoices.find(v => v.id === selectedVoiceId);
      return v ? v.name : t("label.voiceDefault");
    }
    return t("label.voiceDefault");
  }
  if (!lastGenerateParams) return t("label.voiceDefault");
  if (lastGenerateParams.mode === "preset") {
    const sel = document.getElementById("speaker");
    if (sel) {
      for (const opt of sel.options) {
        if (opt.value === lastGenerateParams.speaker) return opt.text;
      }
    }
    return lastGenerateParams.speaker || t("label.voiceDefault");
  }
  if (lastGenerateParams.mode === "saved_voice") {
    const v = savedVoices.find(v => v.id === lastGenerateParams.voice_id);
    return v ? v.name : t("label.voiceDefault");
  }
  return t("label.voiceDefault");
}

function editSentenceVoice(index) {
  const tag = document.getElementById(`sent-voice-${index}`);
  if (!tag) return;
  if (tag.querySelector("select")) return; // 已在编辑中

  const currentConfig = sentenceVoiceConfigs[index];

  // 构建 <select>
  let optionsHtml = `<option value="">${t("label.voiceDefault")} (${getDefaultVoiceLabel()})</option>`;

  // 预设说话人选项组
  const speakerSelect = document.getElementById("speaker");
  if (speakerSelect) {
    optionsHtml += `<optgroup label="${t("tab.preset")}">`;
    for (const opt of speakerSelect.options) {
      const selected = currentConfig && currentConfig.type === "preset" && currentConfig.speaker === opt.value ? " selected" : "";
      optionsHtml += `<option value="preset:${opt.value}"${selected}>${opt.text}</option>`;
    }
    optionsHtml += `</optgroup>`;
  }

  // 声音库选项组
  if (savedVoices.length > 0) {
    optionsHtml += `<optgroup label="${t("tab.library")}">`;
    for (const voice of savedVoices) {
      const selected = currentConfig && currentConfig.type === "saved_voice" && currentConfig.voice_id === voice.id ? " selected" : "";
      optionsHtml += `<option value="saved:${voice.id}"${selected}>${escapeHtml(voice.name)}</option>`;
    }
    optionsHtml += `</optgroup>`;
  }

  tag.innerHTML = `<select class="sentence-voice-select" onchange="finishVoiceEdit(${index}, this)" onblur="finishVoiceEdit(${index}, this)">${optionsHtml}</select>`;
  const selectEl = tag.querySelector("select");
  selectEl.focus();
}

function finishVoiceEdit(index, selectEl) {
  if (!selectEl.isConnected) return;
  selectEl.onchange = null;
  selectEl.onblur = null;
  const val = selectEl.value;
  if (!val) {
    sentenceVoiceConfigs[index] = null;
  } else if (val.startsWith("preset:")) {
    const speaker = val.slice(7);
    const speakerSelect = document.getElementById("speaker");
    let label = speaker;
    if (speakerSelect) {
      for (const opt of speakerSelect.options) {
        if (opt.value === speaker) { label = opt.text; break; }
      }
    }
    sentenceVoiceConfigs[index] = { type: "preset", speaker, label };
  } else if (val.startsWith("saved:")) {
    const voiceId = val.slice(6);
    const voice = savedVoices.find(v => v.id === voiceId);
    sentenceVoiceConfigs[index] = {
      type: "saved_voice",
      voice_id: voiceId,
      label: voice ? voice.name : voiceId,
    };
    sentenceInstructs[index] = "";
  }
  saveSession();
  setTimeout(() => showSentenceEditorView(), 0);
}

// ===== 单句试听 =====
let _sentencePreviewEndHandler = null;

function previewSentenceAudio(index) {
  // 清除之前的句尾监听
  if (_sentencePreviewEndHandler) {
    audioElement.removeEventListener("timeupdate", _sentencePreviewEndHandler);
    _sentencePreviewEndHandler = null;
  }
  // 如果正在播放同一句，停止
  if (sentencePreviewIndex === index && !audioElement.paused) {
    audioElement.pause();
    sentencePreviewIndex = -1;
    showSentenceEditorView();
    return;
  }

  const sub = currentSubtitles && currentSubtitles[index];
  if (!sub) return;

  // 暂停当前播放
  if (!audioElement.paused) audioElement.pause();

  sentencePreviewIndex = index;
  showSentenceEditorView();

  // seek 到句子起点并播放
  audioElement.currentTime = sub.start;
  audioElement.play();

  // 监听 timeupdate，到句尾自动停止
  const endTime = sub.end;
  _sentencePreviewEndHandler = () => {
    if (audioElement.currentTime >= endTime) {
      audioElement.pause();
      audioElement.removeEventListener(
        "timeupdate",
        _sentencePreviewEndHandler,
      );
      _sentencePreviewEndHandler = null;
      sentencePreviewIndex = -1;
      showSentenceEditorView();
    }
  };
  audioElement.addEventListener("timeupdate", _sentencePreviewEndHandler);
}

// ===== 撤销重新生成 =====
function undoRegenerate() {
  if (undoStack.length === 0) return;
  const last = undoStack.pop();
  sentenceAudios[last.index] = last.audio;
  sentenceTexts[last.index] = last.text;
  if (last.instruct !== undefined)
    sentenceInstructs[last.index] = last.instruct;
  if (last.voiceConfig !== undefined)
    sentenceVoiceConfigs[last.index] = last.voiceConfig;
  // 重新合并
  rebuildAudioAndSubtitles();
  saveSession(); // 持久化
  selectedSentenceIndex = last.index;
  showSentenceEditorView();
  const statusEl = document.getElementById("status-message");
  statusEl.innerHTML = `<span class="text-yellow-600">${t("btn.undo")}</span>`;
}

// ===== 删除句子 =====
function deleteSentence(index) {
  if (sentenceTexts.length <= 1) {
    // 预编辑模式下只剩一句，退出预览
    if (isPreviewing) exitPreviewMode();
    return;
  }
  if (!confirm(t("confirm.deleteSentence"))) return;
  finishEditing();
  sentenceTexts.splice(index, 1);
  sentenceInstructs.splice(index, 1);
  sentenceVoiceConfigs.splice(index, 1);
  if (isPreviewing) {
    // 预编辑模式：无音频，跳过 rebuildAudioAndSubtitles
    if (selectedSentenceIndex >= sentenceTexts.length)
      selectedSentenceIndex = sentenceTexts.length - 1;
    if (selectedSentenceIndex === index) selectedSentenceIndex = -1;
    showSentenceEditorView();
    return;
  }
  sentenceAudios.splice(index, 1);
  decodedPcmCache = [];
  rebuildAudioAndSubtitles();
  saveSession(); // 持久化
  if (selectedSentenceIndex >= sentenceTexts.length)
    selectedSentenceIndex = sentenceTexts.length - 1;
  if (selectedSentenceIndex === index) selectedSentenceIndex = -1;
  refreshStatsFromSentences();
  showSentenceEditorView();
}

// ===== 插入句子 =====
function showInsertForm(afterIndex) {
  if (!isPreviewing && !lastGenerateParams) return;
  finishEditing();
  // 取消已有的插入表单
  const existing = document.querySelector(".insert-form-row");
  if (existing) existing.remove();

  const editorList = document.querySelector(".sentence-editor-list");
  if (!editorList) return;

  const isPreset = isPreviewing
    ? currentMode === "preset"
    : lastGenerateParams && lastGenerateParams.mode === "preset";
  const defaultInstruct = isPreviewing
    ? (document.getElementById("instruct")?.value?.trim() || "")
    : (lastGenerateParams?.instruct || "");
  const instructRow = isPreset
    ? `<div style="display:flex;align-items:center;gap:6px">
        <span style="font-size:11px;color:#A0AEC0;white-space:nowrap">${t("label.instructLabel")}:</span>
        <input type="text" id="insert-instruct-input" value="${escapeHtml(defaultInstruct)}" placeholder="${t("label.instructEmpty")}" style="flex:1">
    </div>`
    : "";

  const formHtml = `<li class="insert-form-row"><div class="sentence-insert-form">
        <input type="text" id="insert-text-input" placeholder="${t("btn.insertHint")}" autofocus>
        ${instructRow}
        <div class="sentence-insert-form-actions">
            <button onclick="cancelInsertForm()">${t("btn.stop")}</button>
            <button class="confirm-btn" onclick="confirmInsert(${afterIndex})">${t("btn.addSentence")}</button>
        </div>
    </div></li>`;

  // 找到 afterIndex 对应的插入按钮行
  const items = editorList.children;
  const insertBtnIndex = afterIndex * 2;
  if (insertBtnIndex >= 0 && insertBtnIndex < items.length) {
    items[insertBtnIndex].insertAdjacentHTML("afterend", formHtml);
  } else {
    editorList.insertAdjacentHTML("afterbegin", formHtml);
  }

  const textInput = document.getElementById("insert-text-input");
  textInput.focus();
  // Enter 确认，Escape 取消
  const handleKey = (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      confirmInsert(afterIndex);
    }
    if (e.key === "Escape") {
      e.preventDefault();
      cancelInsertForm();
    }
  };
  textInput.addEventListener("keydown", handleKey);
  const instructInput = document.getElementById("insert-instruct-input");
  if (instructInput) instructInput.addEventListener("keydown", handleKey);
}

function cancelInsertForm() {
  const row = document.querySelector(".insert-form-row");
  if (row) row.remove();
}

async function confirmInsert(afterIndex) {
  const textInput = document.getElementById("insert-text-input");
  const instructInput = document.getElementById("insert-instruct-input");
  const newText = textInput ? textInput.value.trim() : "";
  const newInstruct = instructInput
    ? instructInput.value.trim()
    : (lastGenerateParams?.instruct || "");
  if (!newText) {
    textInput && textInput.focus();
    return;
  }

  // 预编辑模式：纯文本插入，不调 API
  if (isPreviewing) {
    cancelInsertForm();
    sentenceTexts.splice(afterIndex, 0, newText);
    sentenceInstructs.splice(afterIndex, 0, newInstruct);
    sentenceVoiceConfigs.splice(afterIndex, 0, null);
    selectedSentenceIndex = afterIndex;
    showSentenceEditorView();
    return;
  }

  // 移除表单，显示占位行
  cancelInsertForm();

  const statusEl = document.getElementById("status-message");
  const btn = document.getElementById("generate-btn");
  const editorList = document.querySelector(".sentence-editor-list");

  if (editorList) {
    editorList.classList.add("inserting");
    const placeholderHtml = `<li class="inserting-row"><div class="sentence-inserting-placeholder">
            <span class="spinner" style="width:14px;height:14px;border-width:2px"></span>
            <span>${escapeHtml(newText.length > 30 ? newText.slice(0, 30) + "..." : newText)}</span>
        </div></li>`;
    const items = editorList.children;
    const insertBtnIndex = afterIndex * 2;
    if (insertBtnIndex >= 0 && insertBtnIndex < items.length) {
      items[insertBtnIndex].insertAdjacentHTML("afterend", placeholderHtml);
    } else {
      editorList.insertAdjacentHTML("afterbegin", placeholderHtml);
    }
  }

  const originalBtnHtml = btn.innerHTML;
  const originalBtnOnclick = btn.onclick;
  btn.disabled = true;

  try {
    const formData = new FormData();
    formData.append("sentence_text", newText);
    formData.append("mode", lastGenerateParams.mode);
    formData.append("language", lastGenerateParams.language);
    if (lastGenerateParams.speaker)
      formData.append("speaker", lastGenerateParams.speaker);
    // 使用新句子指定的 instruct
    const instruct =
      lastGenerateParams.mode === "preset" && newInstruct
        ? newInstruct
        : lastGenerateParams.instruct;
    if (instruct) formData.append("instruct", instruct);
    if (lastGenerateParams.voice_id)
      formData.append("voice_id", lastGenerateParams.voice_id);
    if (lastGenerateParams.clone_prompt_id)
      formData.append("clone_prompt_id", lastGenerateParams.clone_prompt_id);

    const response = await fetch("/regenerate", {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || "Generate failed");
    }
    const data = await response.json();

    sentenceAudios.splice(afterIndex, 0, data.audio);
    sentenceTexts.splice(afterIndex, 0, newText);
    sentenceInstructs.splice(afterIndex, 0, newInstruct);
    sentenceVoiceConfigs.splice(afterIndex, 0, null);
    decodedPcmCache = [];
    rebuildAudioAndSubtitles();
    saveSession();
    selectedSentenceIndex = afterIndex;
    refreshStatsFromSentences();
    showSentenceEditorView();

    const subtitle = currentSubtitles[afterIndex];
    if (subtitle) {
      audioElement.addEventListener("loadedmetadata", function jumpToNew() {
        audioElement.currentTime = subtitle.start;
        audioElement.play();
        audioElement.removeEventListener("loadedmetadata", jumpToNew);
      });
    }
  } catch (error) {
    statusEl.innerHTML = `<span class="text-red-600">${t("status.failed")}: ${error.message}</span>`;
    if (editorList) {
      const placeholder = editorList.querySelector(".inserting-row");
      if (placeholder) placeholder.remove();
      editorList.classList.remove("inserting");
    }
  } finally {
    btn.disabled = false;
    btn.innerHTML = originalBtnHtml;
    btn.onclick = originalBtnOnclick;
  }
}

// ===== 分句预览模式 =====
function enterPreviewMode() {
  const text = document.getElementById("text-input").value.trim();
  if (!text) {
    document.getElementById("status-message").textContent = t("status.enterText");
    return;
  }
  sentenceTexts = splitTextToSentences(text);
  const instruct = document.getElementById("instruct")?.value?.trim() || "";
  sentenceInstructs = sentenceTexts.map(() =>
    currentMode === "preset" ? instruct : ""
  );
  sentenceVoiceConfigs = sentenceTexts.map(() => null);
  sentenceAudios = [];
  isPreviewing = true;
  selectedSentenceIndex = -1;
  showSentenceEditorView();
}

function exitPreviewMode() {
  // 同步文本回 textarea
  finishEditing();
  if (sentenceTexts.length > 0) {
    document.getElementById("text-input").value = sentenceTexts.join("");
  }
  isPreviewing = false;
  sentenceTexts = [];
  sentenceInstructs = [];
  sentenceVoiceConfigs = [];
  hideProgressView();
  // 恢复按钮为"分句预览"
  resetToPreviewButton();
}

function resetToPreviewButton() {
  const btn = document.getElementById("generate-btn");
  btn.onclick = enterPreviewMode;
  btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg><span>${t("btn.previewSentences")}</span>`;
}

// HTML 转义
function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}
