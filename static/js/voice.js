// ===== 声音库 =====
async function loadSavedVoices() {
  try {
    const response = await fetch("/voices");
    const data = await response.json();
    savedVoices = data.voices || [];
    renderVoiceList();
  } catch (error) {
    console.error("Failed to load voices:", error);
  }
}

// 声音库预览播放器
let previewAudio = null;
let previewingVoiceId = null;

function renderVoiceList() {
  const container = document.getElementById("voice-list");
  if (savedVoices.length === 0) {
    container.innerHTML = `<div class="text-center text-charcoal/50 text-sm py-4">暂无保存的声音</div>`;
    return;
  }

  container.innerHTML = savedVoices
    .map(
      (voice) => `
        <div class="voice-card ${selectedVoiceId === voice.id ? "selected" : ""}" onclick="selectVoice('${voice.id}')">
            <div class="flex items-center justify-between">
                <span class="text-sm font-medium truncate">${voice.name}</span>
                <div class="flex gap-1">
                    <button id="preview-btn-${voice.id}" onclick="event.stopPropagation(); previewVoice('${voice.id}')" class="p-1 hover:bg-warm-gray rounded" title="预览">${previewingVoiceId === voice.id ? "⏸" : "▶"}</button>
                    <button onclick="event.stopPropagation(); deleteVoice('${voice.id}')" class="p-1 hover:bg-red-500/20 hover:text-red-600 rounded" title="删除">✕</button>
                </div>
            </div>
            <div class="text-xs text-charcoal/50 mt-1">${voice.language}</div>
        </div>
    `,
    )
    .join("");
}

function selectVoice(voiceId) {
  selectedVoiceId = selectedVoiceId === voiceId ? null : voiceId;
  renderVoiceList();
}

function previewVoice(voiceId) {
  // 如果正在播放同一个声音，则停止
  if (previewingVoiceId === voiceId && previewAudio) {
    previewAudio.pause();
    previewAudio.currentTime = 0;
    previewAudio = null;
    previewingVoiceId = null;
    renderVoiceList();
    return;
  }

  // 如果正在播放其他声音，先停止
  if (previewAudio) {
    previewAudio.pause();
    previewAudio.currentTime = 0;
  }

  // 播放新声音
  previewAudio = new Audio(`/voices/${voiceId}/preview`);
  previewingVoiceId = voiceId;
  renderVoiceList();

  previewAudio.play();

  // 播放结束后重置状态
  previewAudio.onended = () => {
    previewingVoiceId = null;
    previewAudio = null;
    renderVoiceList();
  };
}

async function deleteVoice(voiceId) {
  if (!confirm(t("confirm.delete"))) return;
  try {
    await fetch(`/voices/${voiceId}`, { method: "DELETE" });
    if (selectedVoiceId === voiceId) selectedVoiceId = null;
    await loadSavedVoices();
  } catch (error) {
    alert("删除失败: " + error.message);
  }
}

async function saveVoice() {
  const name = document.getElementById("voice-name").value.trim();
  if (!name) return;

  const language = document.getElementById("language-clone").value;
  const refText = document.getElementById("ref-text").value.trim();

  let audioFile = recordedBlob
    ? new File([recordedBlob], "recording.webm", { type: "audio/webm" })
    : selectedFile;
  if (!audioFile) return;

  const formData = new FormData();
  formData.append("name", name);
  formData.append("language", language);
  formData.append("ref_text", refText);
  formData.append("audio", audioFile);

  const statusEl = document.getElementById("status-message");
  statusEl.textContent = t("status.saving");

  try {
    const response = await fetch("/voices/save", {
      method: "POST",
      body: formData,
    });
    if (!response.ok) throw new Error("Save failed");
    statusEl.textContent = t("status.saved");
    document.getElementById("voice-name").value = "";
    await loadSavedVoices();
    switchMode("library");
  } catch (error) {
    statusEl.textContent = t("status.failed") + ": " + error.message;
  }
}

// ===== 录音 =====
async function toggleRecording() {
  const btn = document.getElementById("record-btn");
  const icon = document.getElementById("record-icon");
  const text = document.getElementById("record-text");
  const timer = document.getElementById("record-timer");

  if (!mediaRecorder || mediaRecorder.state === "inactive") {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      audioChunks = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunks.push(e.data);
      };

      mediaRecorder.onstop = () => {
        stream.getTracks().forEach((track) => track.stop());
        recordedBlob = new Blob(audioChunks, { type: "audio/webm" });
        selectedFile = null;

        showAudioPreview(URL.createObjectURL(recordedBlob));

        icon.textContent = "🎙️";
        text.textContent = t("btn.record");
        timer.classList.add("hidden");
        btn.classList.remove("bg-red-500/20", "border-red-500");
        clearInterval(timerInterval);
      };

      mediaRecorder.start();
      recordingStartTime = Date.now();

      icon.innerHTML = '<div class="recording-indicator"></div>';
      text.textContent = t("btn.stopRecord");
      timer.classList.remove("hidden");
      timer.textContent = "00:00";
      btn.classList.add("bg-red-500/20", "border-red-500");

      timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
        timer.textContent = `${String(Math.floor(elapsed / 60)).padStart(2, "0")}:${String(elapsed % 60).padStart(2, "0")}`;
      }, 100);
    } catch (err) {
      alert("无法访问麦克风: " + err.message);
    }
  } else if (mediaRecorder.state === "recording") {
    mediaRecorder.stop();
  }
}

function handleFileSelect(input) {
  const file = input.files[0];
  if (file) {
    selectedFile = file;
    recordedBlob = null;
    showAudioPreview(URL.createObjectURL(file));
  }
}

function showAudioPreview(url) {
  document.getElementById("audio-preview-section").classList.remove("hidden");
  document.getElementById("audio-preview").src = url;
  document.getElementById("save-voice-section").classList.remove("hidden");
}

function clearAudio() {
  recordedBlob = null;
  selectedFile = null;
  document.getElementById("audio-preview-section").classList.add("hidden");
  document.getElementById("audio-preview").src = "";
  document.getElementById("audio-file").value = "";
  document.getElementById("save-voice-section").classList.add("hidden");
}
