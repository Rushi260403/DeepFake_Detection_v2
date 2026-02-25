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
    dropArea.style.border = "2px solid #00ffcc";
});

dropArea.addEventListener("dragleave", () => {
    dropArea.style.border = "2px dashed white";
});

dropArea.addEventListener("drop", (e) => {
    e.preventDefault();
    dropArea.style.border = "2px dashed white";

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

    loader.classList.remove("hidden");
    resultDiv.innerHTML = "";

    const formData = new FormData();
    formData.append("video", selectedFile);

    try {

        const response = await fetch("/upload", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        loader.classList.add("hidden");

        resultDiv.innerHTML = `
            <h2>${data.result}</h2>
            <h3>Confidence: ${data.confidence}</h3>
        `;

    } catch (error) {
        loader.classList.add("hidden");
        resultDiv.innerHTML = "Error analyzing video";
        console.error(error);
    }
}