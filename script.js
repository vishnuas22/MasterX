// MasterX Lite – Advanced script.js

function autoExpand(textarea) {
  textarea.style.height = 'auto';
  textarea.style.height = textarea.scrollHeight + 'px';
}


const chat = document.getElementById("chat");
const userInputEl = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const modelSelectEl = document.getElementById("model-select");

let currentLesson = null;
let messageCache = [];
let lastUsedModel = "";

const modelAbilities = {
  "llama3": ["essay", "general question", "concept explanation"],
  "mistral": ["casual answer", "clarification", "short form", "chat"],
  "codellama": ["code", "debug", "programming", "script"],
  "gemma": ["creative", "example", "metaphor", "recursion"]
};

function addChatMessage(sender, message, meta = {}) {
  const messageEl = document.createElement("div");
  messageEl.classList.add("message", sender);

  const metaInfo = sender === "bot" && meta.model
    ? `<div class='meta'>Model: ${meta.model} | ⏱ ${meta.time} ms</div>`
    : "";

  messageEl.innerHTML = `<div class="bubble">${marked.parse(message)}</div>${metaInfo}`;
  chat.appendChild(messageEl);
  chat.scrollTop = chat.scrollHeight;
}

function getSelectedModel() {
  const dropdown = modelSelectEl.value;
  return dropdown !== "auto" ? dropdown : null;
}

function detectIntent(input) {
  const text = input.toLowerCase();
  if (text.includes("code") || text.includes("function") || text.includes("program")) return "code";
  if (text.includes("essay") || text.includes("history")) return "essay";
  if (text.includes("debug") || text.includes("error")) return "debug";
  if (text.includes("recursion") || text.includes("metaphor")) return "recursion";
  return "general question";
}

function suggestModel(input) {
  const intent = detectIntent(input);
  for (let model in modelAbilities) {
    if (modelAbilities[model].includes(intent)) return model;
  }
  return "llama3";
}

async function askOllama(prompt, model) {
  const res = await fetch("http://localhost:11434/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: model, prompt: prompt, stream: false })
  });
  const data = await res.json();
  return data.response;
}

async function getLessonReply(userInput) {
  addChatMessage("user", userInput);
  addChatMessage("bot", "Thinking...");

  const start = performance.now();

  const selectedModel = getSelectedModel();
  const model = selectedModel || suggestModel(userInput);
  lastUsedModel = model;

  let contextPrompt = currentLesson
    ? `You are a Python mentor helping a student learn programming.
Current Lesson: ${currentLesson.title}
Lesson Summary: ${currentLesson.summary}
Lesson Content: ${currentLesson.content}
Student Question: ${userInput}
Give a helpful, clear, and motivating response.`
    : `You are an intelligent and friendly AI mentor named MasterX helping students learn tech skills, including programming, data science, and AI.
Student Question: ${userInput}
Provide a helpful, thoughtful, and clear response. Include examples or code if useful.`;

  try {
    const reply = await askOllama(contextPrompt, model);
    const timeTaken = Math.round(performance.now() - start);
    chat.lastChild.remove();
    addChatMessage("bot", reply, { model, time: timeTaken });

    messageCache.push({ input: userInput, reply, model, timeTaken });
  } catch (err) {
    chat.lastChild.textContent = "⚠️ Error fetching response from model.";
  }
}

sendBtn.addEventListener("click", () => {
  const input = userInputEl.value.trim();
  if (input !== "") {
    getLessonReply(input);
    userInputEl.value = "";
  }
});

userInputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendBtn.click();
});
