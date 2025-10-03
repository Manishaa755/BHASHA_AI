let mediaRecorder;
let audioChunks = [];

const recBtn = document.getElementById("recBtn");
const stopBtn = document.getElementById("stopBtn");
const statusEl = document.getElementById("status");
const transcriptEl = document.getElementById("transcript");
const langSelect = document.getElementById("langSelect");

// 🎙 Start recording
recBtn.addEventListener("click", async () => {
  console.log("Start Recording button clicked");
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    console.log("Microphone access granted");
    mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
    audioChunks = [];

    mediaRecorder.ondataavailable = (event) => {
      console.log("Data available:", event.data.size);
      if (event.data.size > 0) {
        audioChunks.push(event.data);
      }
    };

    mediaRecorder.onstart = () => {
      console.log("MediaRecorder started");
      statusEl.textContent = "Recording...";
      recBtn.disabled = true;
      stopBtn.disabled = false;
    };

    mediaRecorder.start();
  } catch (err) {
    console.error("Error accessing microphone:", err);
    statusEl.textContent = "Microphone access denied!";
  }
});

// ⏹ Stop & send to bot
stopBtn.addEventListener("click", () => {
  console.log("Stop button clicked");
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();

    mediaRecorder.onstop = async () => {
      console.log("MediaRecorder stopped, sending audio...");
      statusEl.textContent = "Processing audio...";

      const blob = new Blob(audioChunks, { type: "audio/webm" });
      console.log("Blob created:", blob);

      await uploadAudio(blob);

      recBtn.disabled = false;
      stopBtn.disabled = true;
    };
  } else {
    console.warn("Stop clicked but recorder not active");
  }
});

// 📤 Upload audio & get bot reply
async function uploadAudio(blob) {
  const formData = new FormData();
  formData.append("file", blob, "recording.webm");

  const lang = langSelect.value;
  console.log("Uploading to /chat with lang:", lang);

  try {
    const response = await fetch(`/chat/?lang=${lang}`, {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      const errData = await response.json();
      console.error("Server error:", errData);
      statusEl.textContent = `Error: ${errData.detail || response.statusText}`;
      return;
    }

    const result = await response.json();
    console.log("Server response:", result);

    transcriptEl.textContent =
      `You said: ${result.user_text || "(unrecognized)"}\n\nBot: ${result.bot_reply}`;
    statusEl.textContent = "Bot replied!";
  } catch (err) {
    console.error("Upload failed:", err);
    statusEl.textContent = "Upload failed!";
  }
}
