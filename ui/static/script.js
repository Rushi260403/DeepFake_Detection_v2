let chart;

function selectFile() {
    document.getElementById("videoInput").click();
}

const dropArea = document.getElementById("dropArea");
const input = document.getElementById("videoInput");

dropArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropArea.classList.add("dragover");
});

dropArea.addEventListener("dragleave", () => {
    dropArea.classList.remove("dragover");
});

dropArea.addEventListener("drop", (e) => {
    e.preventDefault();
    dropArea.classList.remove("dragover");
    input.files = e.dataTransfer.files;
    dropArea.innerHTML = e.dataTransfer.files[0].name;
});

async function uploadVideo() {

    const file = input.files[0];

    if (!file) {
        alert("Please select a video");
        return;
    }

    const progressBar = document.getElementById("progressBar");
    const progressContainer = document.getElementById("progressContainer");
    const resultDiv = document.getElementById("result");

    progressContainer.classList.remove("hidden");
    progressBar.style.width = "0%";
    resultDiv.innerHTML = "";

    const formData = new FormData();
    formData.append("video", file);

    // Fake progress animation
    let width = 0;
    let interval = setInterval(() => {
        if (width >= 90) clearInterval(interval);
        width += 5;
        progressBar.style.width = width + "%";
    }, 200);

    const response = await fetch("/upload", {
        method: "POST",
        body: formData
    });

    const data = await response.json();

    clearInterval(interval);
    progressBar.style.width = "100%";

    resultDiv.innerHTML = `
        <h2 class="${data.result}">${data.result}</h2>
        <h3>${data.confidence}% Confidence</h3>
        <p>Real Frames: ${data.real_count}</p>
        <p>Fake Frames: ${data.fake_count}</p>
    `;

    createChart(data.real_percent, data.fake_percent);
}

function createChart(real, fake) {

    const ctx = document.getElementById("chartCanvas");

    if (chart) chart.destroy();

    chart = new Chart(ctx, {
        type: "doughnut",
        data: {
            labels: ["Real", "Fake"],
            datasets: [{
                data: [real, fake],
                backgroundColor: ["green", "red"]
            }]
        },
        options: {
            responsive: true
        }
    });
}