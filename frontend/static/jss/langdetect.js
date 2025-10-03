document.addEventListener("DOMContentLoaded", () => {
  const detectBtn = document.getElementById("detectBtn");
  const inputText = document.getElementById("inputText");
  const output = document.getElementById("langOutput");

  detectBtn.addEventListener("click", () => {
    const text = inputText.value.trim();
    if (!text) {
      output.innerText = "⚠️ Please enter some text.";
      return;
    }

    // Simple demo logic (rule-based)
    if (/[अ-ह]/.test(text)) {
      output.innerText = "Detected: Hindi 🇮🇳";
    } else if (/[а-яА-Я]/.test(text)) {
      output.innerText = "Detected: Russian 🇷🇺";
    } else if (/[ぁ-んァ-ン]/.test(text)) {
      output.innerText = "Detected: Japanese 🇯🇵";
    } else if (/[a-zA-Z]/.test(text)) {
      output.innerText = "Detected: English 🇺🇸";
    } else {
      output.innerText = "❓ Could not detect language.";
    }
  });
});
