const input = document.getElementById("videoInput");
const chooseBtn = document.getElementById("chooseBtn");
const fileNameText = document.getElementById("fileName");
const dropArea = document.getElementById("dropArea");

let selectedFile = null;

/* Choose button click */
chooseBtn.addEventListener("click", () => {
    input.click();
});

/* File selected from system */
input.addEventListener("change", () => {
    if (input.files.length > 0) {
        selectedFile = input.files[0];
        fileNameText.innerText = selectedFile.name;
    }
});

/* Drag & Drop */
dropArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropArea.classList.add("hover");
});

dropArea.addEventListener("dragleave", () => {
    dropArea.classList.remove("hover");
});

dropArea.addEventListener("drop", (e) => {
    e.preventDefault();
    dropArea.classList.remove("hover");

    if (e.dataTransfer.files.length > 0) {
        selectedFile = e.dataTransfer.files[0];
        fileNameText.innerText = selectedFile.name;
    }
});

/* Upload & Analyze */
async function uploadVideo() {

    if (!selectedFile) {
        alert("Please select a video first");
        return;
    }

    const loader = document.getElementById("loader");
    const resultDiv = document.getElementById("result");
    const progressBar = document.getElementById("progressBar");
    const card = document.querySelector(".glass");

    loader.classList.remove("hidden");
    resultDiv.innerHTML = "";
    progressBar.style.width = "0%";

    const formData = new FormData();
    formData.append("video", selectedFile);

    const xhr = new XMLHttpRequest();

    xhr.open("POST", "/upload", true);

    // 🔥 LIVE PROGRESS
    xhr.upload.onprogress = function (e) {
        if (e.lengthComputable) {
            let percent = (e.loaded / e.total) * 100;
            progressBar.style.width = percent + "%";
        }
    };

    xhr.onload = function () {

        loader.classList.add("hidden");

        const data = JSON.parse(xhr.responseText);

        showAnimatedResult(data);

        playNotification();
    };

    xhr.send(formData);
}

function animateCounter(element, target) {

    let count = 0;
    let speed = 20;
    let increment = target / 50;

    let interval = setInterval(() => {

        count += increment;

        if (count >= target) {
            count = target;
            clearInterval(interval);
        }

        element.innerText = count.toFixed(1) + "%";

    }, speed);
}

function showAnimatedResult(data) {

    const resultDiv = document.getElementById("result");
    const card = document.querySelector(".glass");

    let resultClass = data.result.toLowerCase().includes("fake")
        ? "fake"
        : "real";

    card.classList.remove("fake-border", "real-border");
    card.classList.add(resultClass + "-border");

    resultDiv.innerHTML = `
        <div class="${resultClass}">
            ${data.result}
            <br>
            Confidence: <span id="counter">0%</span>
        </div>
    `;

    const counterElement = document.getElementById("counter");
    animateCounter(counterElement, parseFloat(data.confidence));
}

function playNotification() {
    const audio = new Audio("/static/success.mp3");
    audio.play();
}

const toggleBtn = document.getElementById("themeToggle");

toggleBtn.addEventListener("click", () => {
    document.body.classList.toggle("light-mode");

    if (document.body.classList.contains("light-mode")) {
        toggleBtn.innerText = "☀";
    } else {
        toggleBtn.innerText = "🌙";
    }
});