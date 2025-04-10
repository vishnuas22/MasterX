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

// Add message to chat box
function addChatMessage(sender, text) {
  const chat = document.getElementById("chat");
  const msg = document.createElement("div");
  msg.className = `chat-message ${sender}`;
  msg.textContent = text;
  chat.appendChild(msg);
  chat.scrollTop = chat.scrollHeight;
}

// Mock lesson-aware reply generator
function getLessonReply(userInput) {
  if (!currentLesson) {
    addChatMessage("bot", "Please select a lesson first.");
    return;
  }

  // You can make this smarter later
  addChatMessage("bot", `You're asking about "${currentLesson.title}": ${userInput}`);
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
