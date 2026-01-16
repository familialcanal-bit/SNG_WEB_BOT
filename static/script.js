// ===============================
// SNG WEB BOT - FRONTEND SCRIPT
// ===============================

const input = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const chatBox = document.getElementById("chat-box");

// Ajouter message
function addMessage(role, text) {
  const msg = document.createElement("div");
  msg.className = role === "user" ? "msg user" : "msg bot";
  msg.innerText = text;
  chatBox.appendChild(msg);
  chatBox.scrollTop = chatBox.scrollHeight;
}

// Envoyer message
async function sendMessage() {
  const message = input.value.trim();
  if (!message) return;

  addMessage("user", message);
  input.value = "";

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ message })
    });

    const data = await res.json();
    console.log("API RESPONSE:", data);

    if (!data.ok) {
      throw new Error(data.error || "Erreur API");
    }

    addMessage("bot", data.reply);

  } catch (err) {
    console.error("CHAT ERROR:", err);
    addMessage("bot", "❌ Erreur serveur");
  }
}

// Bouton
sendBtn.addEventListener("click", sendMessage);

// Touche Entrée
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    sendMessage();
  }
});
