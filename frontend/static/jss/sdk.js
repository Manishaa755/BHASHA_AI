document.addEventListener("DOMContentLoaded", () => {
  const demoBtn = document.getElementById("demoBtn");
  const statusBox = document.getElementById("sdkStatus");

  demoBtn.addEventListener("click", () => {
    statusBox.innerText = "🚀 Running Voice SDK Demo...";
    setTimeout(() => {
      statusBox.innerText = "✅ Voice SDK connected successfully!";
    }, 2000);
  });
});
