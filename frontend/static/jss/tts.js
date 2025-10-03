document.addEventListener("DOMContentLoaded", () => {
  const speakBtn = document.getElementById("speakBtn");
  const ttsInput = document.getElementById("ttsInput");

  speakBtn.addEventListener("click", () => {
    const text = ttsInput.value.trim();
    if (!text) {
      alert("⚠️ Please enter some text!");
      return;
    }

    const speech = new SpeechSynthesisUtterance(text);
    speech.lang = "en-US"; // You can allow user to select later
    window.speechSynthesis.speak(speech);
  });
});
