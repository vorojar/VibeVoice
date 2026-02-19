// ===== 音频播放 =====
const audioElement = document.getElementById("audio-element");

// wavesurfer 波形可视化
let wavesurfer = null;
let wsRegions = null;

function initWavesurfer() {
  if (wavesurfer) return;
  wavesurfer = WaveSurfer.create({
    container: "#waveform",
    height: 48,
    waveColor: "#E8E4DD",
    progressColor: "#E07A5F",
    cursorColor: "#2D3748",
    cursorWidth: 2,
    barWidth: 2,
    barGap: 1,
    barRadius: 2,
    barMinHeight: 1,
    normalize: true,
    interact: true,
    media: audioElement,
  });
  wsRegions = wavesurfer.registerPlugin(WaveSurfer.Regions.create());
  wavesurfer.on("ready", () => {
    updateWaveformRegions();
  });
  wavesurfer.on("timeupdate", (currentTime) => {
    document.getElementById("current-time").textContent =
      formatTime(currentTime);
    updatePlaybackHighlight(currentTime);
  });
}

let _waveformLoading = false;
let _waveformLoadQueued = false;

function loadWaveform() {
  if (!audioElement.src) return;
  if (!wavesurfer) initWavesurfer();
  if (_waveformLoading) {
    _waveformLoadQueued = true;
    return;
  }
  _waveformLoading = true;
  _waveformLoadQueued = false;
  wavesurfer.load(audioElement.src).catch(() => {}).finally(() => {
    _waveformLoading = false;
    if (_waveformLoadQueued) {
      _waveformLoadQueued = false;
      loadWaveform();
    }
  });
}

function updateWaveformRegions() {
  if (!wsRegions) return;
  wsRegions.clearRegions();
  if (!currentSubtitles || !currentSubtitles.length) return;
  currentSubtitles.forEach((sub, i) => {
    wsRegions.addRegion({
      start: sub.start,
      end: sub.end,
      color: i % 2 === 0 ? "rgba(224,122,95,0.15)" : "rgba(224,122,95,0.05)",
      drag: false,
      resize: false,
    });
  });
}

// ===== 重建合并音频和字幕 =====
function rebuildAudioAndSubtitles() {
  const merged = mergeAllSentenceAudios();
  currentSubtitles = merged.subtitles;
  audioElement.src = URL.createObjectURL(merged.blob);
  loadWaveform();
}

// 解码 base64 WAV 为 PCM samples（Int16Array）和采样率
function decodeBase64Wav(b64) {
  if (!b64) {
    return { samples: new Int16Array(0), sampleRate: 24000, numChannels: 1 };
  }
  const binaryString = atob(b64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) bytes[i] = binaryString.charCodeAt(i);
  const wavData = parseWav(bytes.buffer);
  return wavData; // { samples: Int16Array, sampleRate, numChannels }
}

// 根据句尾标点判断停顿时长（秒），再乘以 pace 倍率
function getPauseDuration(text) {
  if (pausePaceMultiplier === 0) return 0;
  if (!text || !text.trim()) return 0;
  // 先检查原始文本是否以换行结尾（段落分隔）
  if (/[\n\r]\s*$/.test(text)) return 0.6 * pausePaceMultiplier;
  const trimmed = text.trimEnd();
  const lastChar = trimmed[trimmed.length - 1];
  let base = 0;
  if (/[。？！?!]/.test(lastChar)) base = 0.3;
  else if (/[；;，,、：:]/.test(lastChar)) base = 0.2;
  else base = 0.15; // 无标点默认短停顿
  return base * pausePaceMultiplier;
}

// 将所有 sentenceAudios 合并为一个 WAV Blob，同时重算字幕
function mergeAllSentenceAudios() {
  const allChunks = []; // { samples: Int16Array } 包含音频和静音片段
  let totalLength = 0;
  const subtitles = [];
  let currentTime = 0;
  let sampleRate = 24000;

  for (let i = 0; i < sentenceAudios.length; i++) {
    // 优先用缓存，避免重复 base64 解码
    if (!decodedPcmCache[i] || decodedPcmCache[i]._src !== sentenceAudios[i]) {
      const decoded = decodeBase64Wav(sentenceAudios[i]);
      decoded._src = sentenceAudios[i]; // 标记来源，音频变化时失效
      decodedPcmCache[i] = decoded;
    }
    const wav = decodedPcmCache[i];
    sampleRate = wav.sampleRate;

    // 添加音频
    allChunks.push(wav.samples);
    totalLength += wav.samples.length;

    const duration = wav.samples.length / sampleRate;
    subtitles.push({
      text: sentenceTexts[i],
      start: Math.round(currentTime * 1000) / 1000,
      end: Math.round((currentTime + duration) * 1000) / 1000,
    });
    currentTime += duration;

    // 句间插入静音（最后一句除外）
    if (i < sentenceAudios.length - 1) {
      const pauseSec = getPauseDuration(sentenceTexts[i]);
      if (pauseSec > 0) {
        const silenceSamples = Math.round(pauseSec * sampleRate);
        allChunks.push(new Int16Array(silenceSamples)); // 全零 = 静音
        totalLength += silenceSamples;
        currentTime += pauseSec;
      }
    }
  }

  // 合并 PCM
  const merged = new Int16Array(totalLength);
  let offset = 0;
  for (const samples of allChunks) {
    merged.set(samples, offset);
    offset += samples.length;
  }

  // 编码为 WAV
  const wavBlob = encodeWav(merged, sampleRate);
  return { blob: wavBlob, subtitles };
}

// 将 Int16Array PCM 编码为 WAV Blob
function encodeWav(samples, sampleRate) {
  const numChannels = 1;
  const bitsPerSample = 16;
  const byteRate = (sampleRate * numChannels * bitsPerSample) / 8;
  const blockAlign = (numChannels * bitsPerSample) / 8;
  const dataSize = samples.length * 2;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  // RIFF header
  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(view, 8, "WAVE");
  // fmt chunk
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);
  // data chunk
  writeString(view, 36, "data");
  view.setUint32(40, dataSize, true);
  // PCM data
  const output = new Int16Array(buffer, 44);
  output.set(samples);

  return new Blob([buffer], { type: "audio/wav" });
}

function writeString(view, offset, str) {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}

// ===== 音频播放控制 =====
function togglePlay() {
  if (audioElement.paused) {
    audioElement.play();
  } else {
    audioElement.pause();
  }
}

function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function toggleDownloadMenu() {
  const menu = document.getElementById("download-menu");
  menu.classList.toggle("hidden");
}

// 点击其他地方关闭下载菜单
document.addEventListener("click", (e) => {
  const menu = document.getElementById("download-menu");
  const btn = document.getElementById("download-btn");
  if (menu && !menu.contains(e.target) && !btn.contains(e.target)) {
    menu.classList.add("hidden");
  }
});

async function downloadAudio(format = "wav") {
  if (!audioElement.src) return;

  // 关闭菜单
  document.getElementById("download-menu").classList.add("hidden");

  const timestamp = Date.now();

  if (format === "wav") {
    const a = document.createElement("a");
    a.href = audioElement.src;
    a.download = `tts_${timestamp}.wav`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  } else if (format === "mp3") {
    try {
      // 显示转换中状态
      const btn = document.getElementById("download-btn");
      const originalHTML = btn.innerHTML;
      btn.innerHTML = '<span class="text-sm">转换中...</span>';
      btn.disabled = true;

      // 获取 WAV 数据
      const response = await fetch(audioElement.src);
      const arrayBuffer = await response.arrayBuffer();

      // 解析 WAV
      const wavData = parseWav(arrayBuffer);

      // 转换为 MP3
      const mp3Blob = await convertToMp3(wavData);

      // 下载
      const a = document.createElement("a");
      a.href = URL.createObjectURL(mp3Blob);
      a.download = `tts_${timestamp}.mp3`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);

      // 恢复按钮
      btn.innerHTML = originalHTML;
      btn.disabled = false;
    } catch (error) {
      console.error("MP3 转换失败:", error);
      alert("MP3 转换失败，请下载 WAV 格式");
      const btn = document.getElementById("download-btn");
      btn.innerHTML = originalHTML;
      btn.disabled = false;
    }
  }
}

function downloadSubtitle(format) {
  if (!currentSubtitles || !currentSubtitles.length) {
    alert("没有可用的字幕数据");
    return;
  }
  document.getElementById("download-menu").classList.add("hidden");
  const timestamp = Date.now();
  let content, filename;
  if (format === "srt") {
    content = currentSubtitles
      .map(
        (s, i) =>
          `${i + 1}\n${formatSrtTime(s.start)} --> ${formatSrtTime(s.end)}\n${s.text}\n`,
      )
      .join("\n");
    filename = `tts_${timestamp}.srt`;
  } else {
    content =
      "WEBVTT\n\n" +
      currentSubtitles
        .map(
          (s, i) =>
            `${i + 1}\n${formatVttTime(s.start)} --> ${formatVttTime(s.end)}\n${s.text}\n`,
        )
        .join("\n");
    filename = `tts_${timestamp}.vtt`;
  }
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
}

function formatSrtTime(seconds) {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  const ms = Math.round((seconds % 1) * 1000);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")},${String(ms).padStart(3, "0")}`;
}

function formatVttTime(seconds) {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  const ms = Math.round((seconds % 1) * 1000);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${String(ms).padStart(3, "0")}`;
}

// 解析 WAV 文件
function parseWav(arrayBuffer) {
  if (!arrayBuffer || arrayBuffer.byteLength < 44) {
    return { samples: new Int16Array(0), sampleRate: 24000, numChannels: 1 };
  }
  const dataView = new DataView(arrayBuffer);

  // 读取 WAV 头信息
  const numChannels = dataView.getUint16(22, true);
  const sampleRate = dataView.getUint32(24, true);
  const bitsPerSample = dataView.getUint16(34, true);

  // 找到 data chunk
  let offset = 12;
  while (offset < arrayBuffer.byteLength) {
    const chunkId = String.fromCharCode(
      dataView.getUint8(offset),
      dataView.getUint8(offset + 1),
      dataView.getUint8(offset + 2),
      dataView.getUint8(offset + 3),
    );
    const chunkSize = dataView.getUint32(offset + 4, true);

    if (chunkId === "data") {
      offset += 8;
      break;
    }
    offset += 8 + chunkSize;
  }

  // 读取 PCM 数据
  const samples = new Int16Array(arrayBuffer, offset);

  return { samples, sampleRate, numChannels };
}

// 转换为 MP3
function convertToMp3(wavData) {
  return new Promise((resolve) => {
    const { samples, sampleRate, numChannels } = wavData;

    const mp3encoder = new lamejs.Mp3Encoder(numChannels, sampleRate, 128);
    const mp3Data = [];

    const blockSize = 1152;
    const numSamples = samples.length / numChannels;

    if (numChannels === 1) {
      // 单声道
      for (let i = 0; i < numSamples; i += blockSize) {
        const chunk = samples.subarray(i, Math.min(i + blockSize, numSamples));
        const mp3buf = mp3encoder.encodeBuffer(chunk);
        if (mp3buf.length > 0) {
          mp3Data.push(mp3buf);
        }
      }
    } else {
      // 立体声 - 分离左右声道
      const left = new Int16Array(numSamples);
      const right = new Int16Array(numSamples);
      for (let i = 0; i < numSamples; i++) {
        left[i] = samples[i * 2];
        right[i] = samples[i * 2 + 1];
      }

      for (let i = 0; i < numSamples; i += blockSize) {
        const leftChunk = left.subarray(i, Math.min(i + blockSize, numSamples));
        const rightChunk = right.subarray(
          i,
          Math.min(i + blockSize, numSamples),
        );
        const mp3buf = mp3encoder.encodeBuffer(leftChunk, rightChunk);
        if (mp3buf.length > 0) {
          mp3Data.push(mp3buf);
        }
      }
    }

    // 完成编码
    const end = mp3encoder.flush();
    if (end.length > 0) {
      mp3Data.push(end);
    }

    resolve(new Blob(mp3Data, { type: "audio/mp3" }));
  });
}

let playingSentenceIndex = -1;

function updatePlaybackHighlight(currentTime) {
  if (!currentSubtitles || !currentSubtitles.length) return;

  // 找到当前播放的句子
  let newIndex = -1;
  for (let i = 0; i < currentSubtitles.length; i++) {
    const sub = currentSubtitles[i];
    if (currentTime >= sub.start && currentTime < sub.end) {
      newIndex = i;
      break;
    }
  }

  if (newIndex === playingSentenceIndex) return;
  playingSentenceIndex = newIndex;

  // 检查句子编辑视图是否可见
  const progressView = document.getElementById("progress-view");
  if (progressView.classList.contains("hidden")) return;
  const items = progressView.querySelectorAll(".sentence-editor-item");
  if (!items.length) return;

  items.forEach((el, i) => {
    // 跳过正在编辑的句子
    const textEl = el.querySelector('[contenteditable="true"]');
    if (textEl) return;

    el.classList.remove("playing", "played");
    if (i === newIndex) {
      el.classList.add("playing");
    } else if (newIndex >= 0 && i < newIndex) {
      el.classList.add("played");
    }
  });

  // 自动滚动到当前句子
  if (newIndex >= 0 && items[newIndex]) {
    items[newIndex].scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
}

function clearPlaybackHighlight() {
  playingSentenceIndex = -1;
  document
    .querySelectorAll(
      ".sentence-editor-item.playing, .sentence-editor-item.played",
    )
    .forEach((el) => {
      el.classList.remove("playing", "played");
    });
}

audioElement.addEventListener("timeupdate", () => {
  if (!wavesurfer) {
    document.getElementById("current-time").textContent = formatTime(
      audioElement.currentTime,
    );
    updatePlaybackHighlight(audioElement.currentTime);
  }
});

audioElement.addEventListener("loadedmetadata", () => {
  document.getElementById("duration").textContent = formatTime(
    audioElement.duration,
  );
});

audioElement.addEventListener("play", () => {
  document.getElementById("play-icon").innerHTML =
    '<rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect>';
  updatePlaybackHighlight(audioElement.currentTime);
});

audioElement.addEventListener("pause", () => {
  document.getElementById("play-icon").innerHTML =
    '<polygon points="5 3 19 12 5 21 5 3"></polygon>';
  clearPlaybackHighlight();
});

audioElement.addEventListener("ended", () => {
  document.getElementById("play-icon").innerHTML =
    '<polygon points="5 3 19 12 5 21 5 3"></polygon>';
  clearPlaybackHighlight();
  if (_sentencePreviewEndHandler) {
    audioElement.removeEventListener("timeupdate", _sentencePreviewEndHandler);
    _sentencePreviewEndHandler = null;
    sentencePreviewIndex = -1;
    showSentenceEditorView();
  }
});
