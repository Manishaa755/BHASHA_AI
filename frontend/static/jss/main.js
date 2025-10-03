document.addEventListener("DOMContentLoaded", () => {
  const sttBtn = document.getElementById("sttBtn");
  if (sttBtn) {
    sttBtn.addEventListener("click", () => {
      window.location.href = "stt.html";
    });
  }
});
