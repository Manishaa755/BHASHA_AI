document.addEventListener("DOMContentLoaded", () => {
  const askBtn = document.getElementById("askBtn");
  const queryInput = document.getElementById("queryInput");
  const responseBox = document.getElementById("llmResponse");

  askBtn.addEventListener("click", () => {
    const query = queryInput.value.trim();
    if (!query) {
      responseBox.innerText = "⚠️ Please ask something!";
      return;
    }

    // Demo response (fake LLM reasoning)
    if (query.toLowerCase().includes("hello")) {
      responseBox.innerText = "👋 Hello! How can I assist you today?";
    } else if (query.toLowerCase().includes("india")) {
      responseBox.innerText = "🇮🇳 India is a diverse country with 22 official languages!";
    } else {
      responseBox.innerText = "🤖 (Demo) I processed your query: " + query;
    }
  });
});
