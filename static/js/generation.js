// ===== 生成语音 =====

// 分句进度生成
let currentEventSource = null;
let isGenerating = false;
let currentSubtitles = null;

function stopGeneration() {
  if (currentEventSource) {
    currentEventSource.close();
    currentEventSource = null;
  }
  isGenerating = false;
  isPreviewing = false;
  generatingProgress = -1;
  currentSubtitles = null;
  sentenceAudios = [];
  sentenceTexts = [];
  sentenceInstructs = [];
  sentenceVoiceConfigs = [];
  decodedPcmCache = [];
  selectedSentenceIndex = -1;
  undoStack = [];
  sentencePreviewIndex = -1;
  if (_sentencePreviewEndHandler) {
    audioElement.removeEventListener("timeupdate", _sentencePreviewEndHandler);
    _sentencePreviewEndHandler = null;
  }
  // 隐藏进度视图
  const textInput = document.getElementById("text-input");
  const progressView = document.getElementById("progress-view");
  textInput.classList.remove("hidden");
  progressView.classList.add("hidden");

  const btn = document.getElementById("generate-btn");
  const statusEl = document.getElementById("status-message");
  btn.disabled = false;
  btn.onclick = enterPreviewMode;
  btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg><span>${t("btn.previewSentences")}</span>`;
  statusEl.innerHTML = `<span class="text-yellow-600">${t("status.stopped")}</span>`;
  updateCharCount();
}

async function regenerateSentence(index) {
  if (!lastGenerateParams) return;
  finishEditing();

  // 保存到撤销栈
  undoStack.push({
    index,
    audio: sentenceAudios[index],
    text: sentenceTexts[index],
    instruct: sentenceInstructs[index],
    voiceConfig: sentenceVoiceConfigs[index],
  });

  const text = sentenceTexts[index];
  const item = document.getElementById(`sent-item-${index}`);
  const actionsEl = item
    ? item.querySelector(".sentence-editor-actions")
    : null;
  let originalActionsHtml = "";
  if (actionsEl) {
    originalActionsHtml = actionsEl.innerHTML;
    actionsEl.innerHTML = `<span class="spinner" style="width:16px;height:16px;border-width:2px;color:#E07A5F"></span><span style="font-size:12px;color:#E07A5F">${t("status.regenerating")}</span>`;
    actionsEl.style.opacity = "1";
  }
  if (item) item.classList.add("regenerating");

  const statusEl = document.getElementById("status-message");
  statusEl.textContent = t("status.regenerating");

  try {
    const formData = new FormData();
    formData.append("sentence_text", text);

    // 逐句声音配置优先
    const vc = sentenceVoiceConfigs[index];
    if (vc) {
      formData.append("mode", vc.type === "preset" ? "preset" : "saved_voice");
      formData.append("language", lastGenerateParams.language);
      if (vc.type === "preset") {
        formData.append("speaker", vc.speaker);
        const instruct = sentenceInstructs[index] || "";
        if (instruct) formData.append("instruct", instruct);
      } else if (vc.type === "saved_voice") {
        formData.append("voice_id", vc.voice_id);
      }
    } else {
      formData.append("mode", lastGenerateParams.mode);
      formData.append("language", lastGenerateParams.language);
      if (lastGenerateParams.speaker)
        formData.append("speaker", lastGenerateParams.speaker);
      // 逐句情感指令（preset 模式优先使用逐句值）
      const instruct =
        lastGenerateParams.mode === "preset" && sentenceInstructs[index]
          ? sentenceInstructs[index]
          : lastGenerateParams.instruct;
      if (instruct) formData.append("instruct", instruct);
      if (lastGenerateParams.voice_id)
        formData.append("voice_id", lastGenerateParams.voice_id);
      if (lastGenerateParams.clone_prompt_id)
        formData.append("clone_prompt_id", lastGenerateParams.clone_prompt_id);
    }

    const response = await fetch("/regenerate", {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || "Regenerate failed");
    }

    const data = await response.json();

    // 替换该句的音频和文本
    sentenceAudios[index] = data.audio;
    sentenceTexts[index] = text;

    // 前端合并所有句子音频 + 重算字幕
    rebuildAudioAndSubtitles();

    // 跳转到该句开头播放
    const subtitle = currentSubtitles[index];
    if (subtitle) {
      audioElement.addEventListener(
        "loadedmetadata",
        function jumpToSentence() {
          audioElement.currentTime = subtitle.start;
          audioElement.play();
          audioElement.removeEventListener("loadedmetadata", jumpToSentence);
        },
      );
    }

    statusEl.innerHTML = `<span class="text-green-600">${t("status.success")}</span>`;
    saveSession(); // 持久化

    // 重新渲染句子列表
    selectedSentenceIndex = index;
    showSentenceEditorView();
  } catch (error) {
    // 回滚撤销栈（生成失败，无需撤销）
    undoStack.pop();
    statusEl.innerHTML = `<span class="text-red-600">${t("status.failed")}: ${error.message}</span>`;
    // 恢复按钮
    if (actionsEl) {
      actionsEl.innerHTML = originalActionsHtml;
      actionsEl.style.opacity = "";
    }
    if (item) item.classList.remove("regenerating");
  }
}

async function generateWithProgress(url, btn, statusEl) {
  isGenerating = true;

  // 获取文本并分句显示
  const text = document.getElementById("text-input").value.trim();
  sentenceTexts = splitTextToSentences(text);
  generatingProgress = 0;
  showSentenceEditorView();

  return new Promise((resolve, reject) => {
    const eventSource = new EventSource(url);
    currentEventSource = eventSource;
    let totalSentences = 0;

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.started) {
          totalSentences = data.total;
          statusEl.textContent = `${t("status.generating")} 0/${totalSentences} ${t("stats.sentences")} (0%)`;
          // 显示停止按钮
          btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="6" width="12" height="12"></rect></svg><span>${t("btn.stop")}</span>`;
          btn.disabled = false;
          btn.onclick = stopGeneration;
          // 标记第一句正在生成
          updateGeneratingProgress(0);
        }

        if (data.progress) {
          const { current, total, percent } = data.progress;
          btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="6" width="12" height="12"></rect></svg><span>${current}/${total} ${t("stats.sentences")} (${percent}%)</span>`;
          statusEl.textContent = `${t("status.generating")} ${current}/${total} ${t("stats.sentences")} (${percent}%)`;
          // 更新句子样式
          updateGeneratingProgress(current);
        }

        if (data.done) {
          eventSource.close();
          currentEventSource = null;
          isGenerating = false;
          currentSubtitles = data.subtitles || null;

          // 保存每句音频和文本
          if (data.sentence_audios) {
            sentenceAudios = data.sentence_audios;
            sentenceTexts = (data.subtitles || []).map((s) => s.text);
            sentenceInstructs = sentenceTexts.map(
              () => lastGenerateParams?.instruct || "",
            );
            sentenceVoiceConfigs = sentenceTexts.map(() => null);
          }
          if (data.clone_prompt_id) {
            clonePromptId = data.clone_prompt_id;
            if (lastGenerateParams)
              lastGenerateParams.clone_prompt_id = data.clone_prompt_id;
          }
          saveSession(); // 持久化

          // 始终显示句子编辑视图
          selectedSentenceIndex = -1;
          showSentenceEditorView();

          // 将 base64 转为 blob URL
          const binaryString = atob(data.audio);
          const bytes = new Uint8Array(binaryString.length);
          for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
          }
          const blob = new Blob([bytes], { type: "audio/wav" });
          const audioUrl = URL.createObjectURL(blob);

          // 恢复按钮
          btn.onclick = enterPreviewMode;
          resolve({
            audioUrl,
            stats: data.stats,
          });
        }

        if (data.error) {
          eventSource.close();
          currentEventSource = null;
          isGenerating = false;
          hideProgressView();
          btn.onclick = enterPreviewMode;
          reject(new Error(data.error));
        }
      } catch (e) {
        console.error("Parse error:", e);
      }
    };

    eventSource.onerror = (error) => {
      eventSource.close();
      currentEventSource = null;
      isGenerating = false;
      hideProgressView();
      btn.onclick = enterPreviewMode;
      reject(new Error(t("status.failed")));
    };
  }).catch((error) => {
    hideProgressView();
    statusEl.innerHTML = `<span class="text-red-600">${t("status.failed")}: ${error.message}</span>`;
    btn.disabled = false;
    btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg><span>${t("btn.previewSentences")}</span>`;
    return null;
  });
}

// POST版本的进度生成（用于克隆，需要上传文件）
async function generateWithProgressPost(url, formData, btn, statusEl) {
  isGenerating = true;
  const text = document.getElementById("text-input").value.trim();
  sentenceTexts = splitTextToSentences(text);
  generatingProgress = 0;
  showSentenceEditorView();

  try {
    const response = await fetch(url, { method: "POST", body: formData });
    if (!response.ok) throw new Error("Request failed");

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let result = null;

    btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="6" width="12" height="12"></rect></svg><span>${t("btn.stop")}</span>`;
    btn.disabled = false;
    btn.onclick = () => {
      reader.cancel();
      stopGeneration();
    };

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          try {
            const data = JSON.parse(line.slice(6));
            if (data.started) {
              statusEl.textContent = `${t("status.generating")} 0/${data.total} ${t("stats.sentences")} (0%)`;
              updateGeneratingProgress(0);
            }
            if (data.progress) {
              const { current, total, percent } = data.progress;
              btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="6" width="12" height="12"></rect></svg><span>${current}/${total} ${t("stats.sentences")} (${percent}%)</span>`;
              statusEl.textContent = `${t("status.generating")} ${current}/${total} ${t("stats.sentences")} (${percent}%)`;
              updateGeneratingProgress(current);
            }
            if (data.done) {
              currentSubtitles = data.subtitles || null;
              // 保存每句音频和文本
              if (data.sentence_audios) {
                sentenceAudios = data.sentence_audios;
                sentenceTexts = (data.subtitles || []).map((s) => s.text);
                sentenceInstructs = sentenceTexts.map(
                  () => lastGenerateParams?.instruct || "",
                );
                sentenceVoiceConfigs = sentenceTexts.map(() => null);
              }
              if (data.clone_prompt_id) {
                clonePromptId = data.clone_prompt_id;
                if (lastGenerateParams)
                  lastGenerateParams.clone_prompt_id = data.clone_prompt_id;
              }
              saveSession(); // 持久化
              const binaryString = atob(data.audio);
              const bytes = new Uint8Array(binaryString.length);
              for (let i = 0; i < binaryString.length; i++)
                bytes[i] = binaryString.charCodeAt(i);
              result = {
                audioUrl: URL.createObjectURL(
                  new Blob([bytes], { type: "audio/wav" }),
                ),
                stats: data.stats,
              };
            }
            if (data.error) throw new Error(data.error);
          } catch (e) {
            if (e.message !== "Unexpected end of JSON input") throw e;
          }
        }
      }
    }

    isGenerating = false;
    // 始终显示句子编辑视图
    selectedSentenceIndex = -1;
    showSentenceEditorView();
    btn.onclick = enterPreviewMode;
    return result;
  } catch (error) {
    isGenerating = false;
    hideProgressView();
    btn.onclick = enterPreviewMode;
    statusEl.innerHTML = `<span class="text-red-600">${t("status.failed")}: ${error.message}</span>`;
    btn.disabled = false;
    btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg><span>${t("btn.previewSentences")}</span>`;
    return null;
  }
}

async function generate() {
  // 从预编辑模式启动生成
  if (isPreviewing) {
    return generateFromPreview();
  }

  const text = document.getElementById("text-input").value.trim();
  const btn = document.getElementById("generate-btn");
  const statusEl = document.getElementById("status-message");

  if (!text) {
    statusEl.textContent = t("status.enterText");
    return;
  }

  // 验证克隆模式
  if (currentMode === "clone" && !recordedBlob && !selectedFile) {
    statusEl.textContent = t("status.needAudio");
    return;
  }

  // 验证声音库模式
  if (currentMode === "library" && !selectedVoiceId) {
    statusEl.textContent = t("status.selectVoice");
    return;
  }

  // 验证声音设计模式
  if (currentMode === "design") {
    const desc = document.getElementById("voice-desc").value.trim();
    if (!desc) {
      statusEl.textContent = t("status.needDesc");
      return;
    }
  }

  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span><span>${t("status.generating")}</span>`;
  statusEl.textContent = "";

  // 清除上次的句子编辑状态
  sentenceAudios = [];
  sentenceTexts = [];
  sentenceInstructs = [];
  sentenceVoiceConfigs = [];
  decodedPcmCache = [];
  selectedSentenceIndex = -1;
  clonePromptId = null;
  undoStack = [];
  clearSession(); // 清除持久化

  // 确保模型加载
  const modelType =
    currentMode === "preset"
      ? "custom"
      : currentMode === "library"
        ? "clone"
        : currentMode;
  const modelReady = await ensureModelLoaded(modelType);
  if (!modelReady) {
    hideProgressView();
    btn.disabled = false;
    return;
  }

  try {
    let response;
    let stats = null;

    if (currentMode === "preset") {
      const speaker = document.getElementById("speaker").value;
      const language = document.getElementById("language-preset").value;
      const instruct = document.getElementById("instruct").value.trim();

      lastGenerateParams = {
        mode: "preset",
        speaker,
        language,
        instruct: instruct || null,
      };

      // 使用分句进度生成
      const params = new URLSearchParams({ text, speaker, language });
      if (instruct) params.append("instruct", instruct);

      const result = await generateWithProgress(
        `/tts/progress?${params.toString()}`,
        btn,
        statusEl,
      );
      if (!result) return;

      audioElement.src = result.audioUrl;
      stats = result.stats;

      // 跳过后面的 response 处理
      loadWaveform();
      audioElement.play();
      document.getElementById("player-section").classList.remove("hidden");

      if (stats) {
        lastStatsData = stats;
        renderStats();
      }

      if (sentenceTexts.length <= 1) {
        statusEl.innerHTML = `<span class="text-green-600">${t("status.success")}</span>`;
      }
      document.getElementById("save-voice-section").classList.add("hidden");

      saveToHistory(text, "preset");
      btn.disabled = false;
      btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg><span>${t("btn.previewSentences")}</span>`;
      return;
    } else if (currentMode === "library") {
      const language = document.getElementById("language-library").value;

      lastGenerateParams = {
        mode: "saved_voice",
        language,
        voice_id: selectedVoiceId,
      };

      const params = new URLSearchParams({ text });
      if (language) params.append("language", language);

      const result = await generateWithProgress(
        `/voices/${selectedVoiceId}/tts/progress?${params.toString()}`,
        btn,
        statusEl,
      );
      if (!result) return;

      audioElement.src = result.audioUrl;
      stats = result.stats;

      loadWaveform();
      audioElement.play();
      document.getElementById("player-section").classList.remove("hidden");

      if (stats) {
        lastStatsData = stats;
        renderStats();
      }

      if (sentenceTexts.length <= 1) {
        statusEl.innerHTML = `<span class="text-green-600">${t("status.success")}</span>`;
      }
      document.getElementById("save-voice-section").classList.add("hidden");

      saveToHistory(text, "saved_voice");
      btn.disabled = false;
      btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg><span>${t("btn.previewSentences")}</span>`;
      return;
    } else if (currentMode === "clone") {
      const language = document.getElementById("language-clone").value;
      const refText = document.getElementById("ref-text").value.trim();

      lastGenerateParams = { mode: "clone", language, clone_prompt_id: null };

      const audioFile = recordedBlob
        ? new File([recordedBlob], "recording.webm", { type: "audio/webm" })
        : selectedFile;
      const formData = new FormData();
      formData.append("audio", audioFile);
      formData.append("text", text);
      formData.append("language", language);
      formData.append("ref_text", refText);

      const result = await generateWithProgressPost(
        "/clone/progress",
        formData,
        btn,
        statusEl,
      );
      if (!result) return;

      audioElement.src = result.audioUrl;
      stats = result.stats;

      loadWaveform();
      audioElement.play();
      document.getElementById("player-section").classList.remove("hidden");

      if (stats) {
        lastStatsData = stats;
        renderStats();
      }

      if (sentenceTexts.length <= 1) {
        statusEl.innerHTML = `<span class="text-green-600">${t("status.success")}</span>`;
      }
      saveToHistory(text, "clone");
      btn.disabled = false;
      btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg><span>${t("btn.previewSentences")}</span>`;
      return;
    } else if (currentMode === "design") {
      const language = document.getElementById("language-design").value;
      const desc = document.getElementById("voice-desc").value.trim();

      lastGenerateParams = { mode: "design", language, instruct: desc };

      // 使用声音设计的分句进度生成
      const params = new URLSearchParams({ text, language, instruct: desc });

      const result = await generateWithProgress(
        `/design/progress?${params.toString()}`,
        btn,
        statusEl,
      );
      if (!result) return;

      audioElement.src = result.audioUrl;
      stats = result.stats;

      loadWaveform();
      audioElement.play();
      document.getElementById("player-section").classList.remove("hidden");

      if (stats) {
        lastStatsData = stats;
        renderStats();
      }

      if (sentenceTexts.length <= 1) {
        statusEl.innerHTML = `<span class="text-green-600">${t("status.success")}</span>`;
      }
      document.getElementById("save-voice-section").classList.add("hidden");

      saveToHistory(text, "design");
      btn.disabled = false;
      btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg><span>${t("btn.previewSentences")}</span>`;
      return;
    }

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || t("status.failed"));
    }

    // 播放音频
    const blob = await response.blob();
    audioElement.src = URL.createObjectURL(blob);
    loadWaveform();
    audioElement.play();

    // 显示播放器
    document.getElementById("player-section").classList.remove("hidden");

    // 显示统计
    const charCount = response.headers.get("X-Char-Count");
    const elapsed = response.headers.get("X-Elapsed");
    const avgPerChar = response.headers.get("X-Avg-Per-Char");

    if (charCount && elapsed) {
      lastStatsData = {
        char_count: charCount,
        sentence_count: null,
        elapsed,
        avg_per_char: avgPerChar,
      };
      renderStats();
    }

    statusEl.innerHTML = `<span class="text-green-600">${t("status.success")}</span>`;
  } catch (error) {
    statusEl.innerHTML = `<span class="text-red-600">${t("status.failed")}: ${error.message}</span>`;
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg><span>${t("btn.previewSentences")}</span>`;
  }
}

function buildLastGenerateParams() {
  if (currentMode === "preset") {
    const speaker = document.getElementById("speaker").value;
    const language = document.getElementById("language-preset").value;
    const instruct = document.getElementById("instruct").value.trim();
    lastGenerateParams = {
      mode: "preset",
      speaker,
      language,
      instruct: instruct || null,
    };
  } else if (currentMode === "library") {
    const language = document.getElementById("language-library").value;
    lastGenerateParams = {
      mode: "saved_voice",
      language,
      voice_id: selectedVoiceId,
    };
  } else if (currentMode === "clone") {
    const language = document.getElementById("language-clone").value;
    lastGenerateParams = {
      mode: "clone",
      language,
      clone_prompt_id: clonePromptId || null,
    };
  } else if (currentMode === "design") {
    const language = document.getElementById("language-design").value;
    const desc = document.getElementById("voice-desc").value.trim();
    lastGenerateParams = { mode: "design", language, instruct: desc };
  }
}

async function generateMixedVoices(texts, instructs, voiceConfigs) {
  const btn = document.getElementById("generate-btn");
  const statusEl = document.getElementById("status-message");

  // 确定需要哪些模型
  const modelsNeeded = new Set();
  for (const vc of voiceConfigs) {
    if (vc) {
      if (vc.type === "preset") modelsNeeded.add("custom");
      else if (vc.type === "saved_voice") modelsNeeded.add("clone");
    }
  }
  // 默认声音也需要对应模型
  if (lastGenerateParams) {
    if (lastGenerateParams.mode === "preset") modelsNeeded.add("custom");
    else if (
      lastGenerateParams.mode === "saved_voice" ||
      lastGenerateParams.mode === "clone"
    )
      modelsNeeded.add("clone");
    else if (lastGenerateParams.mode === "design") modelsNeeded.add("design");
  }

  for (const modelType of modelsNeeded) {
    const ready = await ensureModelLoaded(modelType);
    if (!ready) {
      hideProgressView();
      btn.disabled = false;
      return;
    }
  }

  isGenerating = true;
  sentenceTexts = texts;
  sentenceInstructs = instructs;
  sentenceVoiceConfigs = voiceConfigs;
  sentenceAudios = new Array(texts.length).fill(null);
  decodedPcmCache = [];
  generatingProgress = 0;

  // 显示句子编辑器（生成中模式）
  showSentenceEditorView();

  // 显示停止按钮
  btn.disabled = false;
  btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="6" width="12" height="12"></rect></svg><span>${t("btn.stop")}</span>`;
  btn.onclick = stopGeneration;
  statusEl.textContent = `${t("status.generating")} 0/${texts.length} ${t("stats.sentences")} (0%)`;

  const startTime = Date.now();

  try {
    for (let i = 0; i < texts.length; i++) {
      if (!isGenerating) return; // 用户按了停止

      updateGeneratingProgress(i);
      statusEl.textContent = `${t("status.generating")} ${i}/${texts.length} ${t("stats.sentences")} (${Math.round((i / texts.length) * 100)}%)`;
      btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="6" width="12" height="12"></rect></svg><span>${i}/${texts.length} ${t("stats.sentences")}</span>`;

      const formData = new FormData();
      formData.append("sentence_text", texts[i]);

      const vc = voiceConfigs[i];
      if (vc) {
        formData.append(
          "mode",
          vc.type === "preset" ? "preset" : "saved_voice",
        );
        formData.append("language", lastGenerateParams.language);
        if (vc.type === "preset") {
          formData.append("speaker", vc.speaker);
          const inst = instructs[i] || "";
          if (inst) formData.append("instruct", inst);
        } else if (vc.type === "saved_voice") {
          formData.append("voice_id", vc.voice_id);
        }
      } else {
        formData.append("mode", lastGenerateParams.mode);
        formData.append("language", lastGenerateParams.language);
        if (lastGenerateParams.speaker)
          formData.append("speaker", lastGenerateParams.speaker);
        const inst =
          lastGenerateParams.mode === "preset" && instructs[i]
            ? instructs[i]
            : lastGenerateParams.instruct;
        if (inst) formData.append("instruct", inst);
        if (lastGenerateParams.voice_id)
          formData.append("voice_id", lastGenerateParams.voice_id);
        if (lastGenerateParams.clone_prompt_id)
          formData.append(
            "clone_prompt_id",
            lastGenerateParams.clone_prompt_id,
          );
      }

      const response = await fetch("/regenerate", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || `Sentence ${i + 1} failed`);
      }
      const data = await response.json();
      sentenceAudios[i] = data.audio;
    }

    // 全部完成
    isGenerating = false;
    generatingProgress = -1;

    // 重建音频
    rebuildAudioAndSubtitles();
    loadWaveform();
    audioElement.play();
    document.getElementById("player-section").classList.remove("hidden");

    // 统计
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    const charCount = texts.join("").length;
    lastStatsData = {
      char_count: charCount,
      sentence_count: texts.length,
      elapsed,
      avg_per_char: (parseFloat(elapsed) / charCount).toFixed(2),
    };
    renderStats();

    saveSession();
    saveToHistory(texts.join(""), "mixed");

    selectedSentenceIndex = -1;
    showSentenceEditorView();

    statusEl.innerHTML = `<span class="text-green-600">${t("status.success")}</span>`;
    btn.disabled = false;
    btn.onclick = enterPreviewMode;
    btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg><span>${t("btn.previewSentences")}</span>`;
  } catch (error) {
    isGenerating = false;
    generatingProgress = -1;
    statusEl.innerHTML = `<span class="text-red-600">${t("status.failed")}: ${error.message}</span>`;
    btn.disabled = false;
    btn.onclick = enterPreviewMode;
    btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg><span>${t("btn.previewSentences")}</span>`;
  }
}

async function generateFromPreview() {
  // 收集编辑后的句子文本，写回 textarea
  finishEditing();
  const editedTexts = [...sentenceTexts];
  const editedInstructs = [...sentenceInstructs];
  const editedVoiceConfigs = [...sentenceVoiceConfigs];
  document.getElementById("text-input").value = editedTexts.join("");

  // 退出预编辑状态
  isPreviewing = false;

  // 检查是否有逐句声音覆盖
  const hasVoiceOverrides = editedVoiceConfigs.some((c) => c !== null);
  if (hasVoiceOverrides) {
    // 构建默认参数
    buildLastGenerateParams();
    // 走混合声音生成（逐句调 /regenerate）
    await generateMixedVoices(editedTexts, editedInstructs, editedVoiceConfigs);
    return;
  }

  // 无覆盖，走原有批量生成流程
  await generate();

  // 生成完成后，尝试对齐 sentenceInstructs 和 voiceConfigs
  if (sentenceTexts.length === editedInstructs.length) {
    sentenceInstructs = editedInstructs;
  }
  if (sentenceTexts.length === editedVoiceConfigs.length) {
    sentenceVoiceConfigs = editedVoiceConfigs;
  }
  // 数量不同则保留 generate() 里设的默认值

  saveSession();
}

async function ensureModelLoaded(modelType) {
  const statusEl = document.getElementById("status-message");

  try {
    const statusRes = await fetch("/model/status");
    const statusData = await statusRes.json();

    if (statusData.status[modelType] === "loaded") {
      return true;
    }

    if (statusData.status[modelType] === "unloaded") {
      statusEl.textContent = t("status.modelLoading");
      await fetch(`/model/load/${modelType}`, { method: "POST" });
    }

    // 等待加载完成
    for (let i = 0; i < 120; i++) {
      await new Promise((r) => setTimeout(r, 1000));
      const res = await fetch("/model/status");
      const data = await res.json();
      if (data.status[modelType] === "loaded") {
        statusEl.textContent = "";
        return true;
      }
      if (data.status[modelType] === "unloaded") {
        statusEl.textContent = t("status.failed");
        return false;
      }
    }

    statusEl.textContent = t("status.failed");
    return false;
  } catch (error) {
    statusEl.textContent = t("status.failed") + ": " + error.message;
    return false;
  }
}
