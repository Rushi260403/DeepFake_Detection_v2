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

        // Add animation
        resultDiv.style.opacity = "0";

        setTimeout(() => {

            if (data.result.toLowerCase().includes("fake")) {

                resultDiv.innerHTML = `
                    <div class="fake">
                        ⚠ FAKE VIDEO <br>
                        Confidence: ${data.confidence}
                    </div>
                `;

            } else {

                resultDiv.innerHTML = `
                    <div class="real">
                        ✔ REAL VIDEO <br>
                        Confidence: ${data.confidence}
                    </div>
                `;
            }

            resultDiv.style.opacity = "1";

        }, 200);

    } catch (error) {

        loader.classList.add("hidden");

        resultDiv.innerHTML = `
            <div class="fake">
                Error analyzing video
            </div>
        `;

        console.error(error);
    }
}