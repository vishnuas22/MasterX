// Global state
let currentLesson = null;

// Sample Python lessons (customize or load dynamically later)
const lessons = [
  {
    id: "intro",
    title: "Intro to Python",
    content: "Welcome to Python! Let's explore the basics of programming.",
    summary: "Understand what Python is and how it’s used."
  },
  {
    id: "variables",
    title: "Variables in Python",
    content: "Variables are used to store data in Python. Example: `x = 5`.",
    summary: "Learn how to declare and use variables."
  }
];

// Start a track (e.g., Python)
function startTrack(trackName) {
  if (trackName === 'python') {
    renderLessons(lessons);
    document.getElementById("lesson-tabs").style.display = "block";
    document.getElementById("lesson-content").style.display = "block";
  } else {
    alert("Track not found: " + trackName);
  }
}

// Render lesson tabs
function renderLessons(lessonList) {
  const tabsContainer = document.getElementById("lesson-tabs");
  tabsContainer.innerHTML = "";

  lessonList.forEach((lesson, index) => {
    const tab = document.createElement("button");
    tab.textContent = lesson.title;
    tab.className = "lesson-tab";
    tab.onclick = () => {
      currentLesson = lesson;
      renderLessonContent(lesson);
    };
    tabsContainer.appendChild(tab);

    // Auto-select first lesson
    if (index === 0) {
      currentLesson = lesson;
      renderLessonContent(lesson);
    }
  });
}

// Display lesson content
function renderLessonContent(lesson) {
  const contentDiv = document.getElementById("lesson-content");
  contentDiv.innerHTML = `
    <h2>${lesson.title}</h2>
    <p>${lesson.content}</p>
  `;
}

// Send message
document.getElementById("send-btn").addEventListener("click", () => {
  const input = document.getElementById("user-input");
  const message = input.value.trim();
  if (message !== "") {
    addChatMessage("user", message);
    getLessonReply(message);
    input.value = "";
  }
});



// Ollama Model 

async function askOllama(prompt) {
  try {
    const response = await fetch("http://localhost:11434/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "mistral",
        messages: [{ role: "user", content: prompt }],
        stream: false  // turn off streaming for simplicity
      })
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const data = await response.json();

    // Some versions return `message.content`, others return `response.message`
    if (data.message && data.message.content) {
      return data.message.content;
    } else if (data.content) {
      return data.content;
    } else {
      throw new Error("Unexpected response format from Ollama");
    }
  } catch (err) {
    console.error("❌ Error in askOllama:", err);
    throw err;
  }
}




// Add message to chat box
function addChatMessage(sender, text) {
  const chat = document.getElementById("chat");
  const msg = document.createElement("div");
  msg.className = `chat-message ${sender}`;
  msg.innerHTML = marked.parse(text);

  chat.appendChild(msg);
  chat.scrollTop = chat.scrollHeight;
}

// Mock lesson-aware reply generator
async function getLessonReply(userInput) {
  const chat = document.getElementById("chat");

  // Add placeholder message and keep reference
  const thinkingMessage = document.createElement("div");
  thinkingMessage.className = "chat-message bot";
  thinkingMessage.textContent = "Thinking...";
  chat.appendChild(thinkingMessage);
  chat.scrollTop = chat.scrollHeight;

  let contextPrompt;

  if (currentLesson) {
    // Inside a lesson → contextual prompt
    contextPrompt = `
You are a Python mentor helping a student learn programming.
Current Lesson: ${currentLesson.title}
Lesson Summary: ${currentLesson.summary}
Lesson Content: ${currentLesson.content}

Student Question: ${userInput}

Give a helpful, clear, and motivating response.
    `;
  } else {
    // No lesson selected → general AI helper
    contextPrompt = `
You are an intelligent and friendly AI mentor named MasterX helping students learn tech skills, including programming, data science, and AI.

Student Question: ${userInput}

Provide a helpful, thoughtful, and clear response. Include examples or code if useful.
    `;
  }

  try {
    const reply = await askOllama(contextPrompt);
    thinkingMessage.textContent = reply; // Replace "Thinking..." with response
  } catch (err) {
    thinkingMessage.textContent = "⚠️ Error fetching response from model. Please ensure Ollama is running.";
  }
}




function updateModeIndicator() {
  const mode = document.getElementById("mode-indicator");
  mode.textContent = currentLesson ? `📘 Lesson: ${currentLesson.title}` : "💬 General Mode";
}



// Reset progress
function resetProgress() {
  localStorage.clear();
  currentLesson = null;
  document.getElementById("lesson-tabs").style.display = "none";
  document.getElementById("lesson-content").style.display = "none";
  document.getElementById("chat").innerHTML = "";
  addChatMessage("bot", "Progress has been reset.");
}

// Show help
function showHelp() {
  addChatMessage("bot", "🧠 You can start a track from the sidebar. Try the Python Track first!");
}
