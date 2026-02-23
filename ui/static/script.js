async function uploadVideo() {

    const input = document.getElementById("videoInput");
    const file = input.files[0];

    if (!file) {
        alert("Please select a video");
        return;
    }

    const loader = document.getElementById("loader");
    const resultDiv = document.getElementById("result");

    // Show loader with progress bar
    loader.classList.remove("hidden");

    // Insert modern progress bar UI
    loader.innerHTML = `
        <div class="loader-box">
            <div class="progress-bar">
                <div class="progress-fill"></div>
            </div>
            <p>Analyzing video... Please wait</p>
        </div>
    `;

    resultDiv.innerHTML = "";

    const formData = new FormData();
    formData.append("video", file);

    try {

        const response = await fetch("/upload", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        // Hide loader after processing
        loader.classList.add("hidden");

        // Show result with modern styling
        const resultColor =
            data.result === "FAKE" ? "#ff4d4d" :
            data.result === "REAL" ? "#00cc66" :
            "#ffaa00";

        resultDiv.innerHTML = `
            <div class="result-box" style="border-color:${resultColor}">
                <h2 style="color:${resultColor}">
                    ${data.result}
                </h2>

                <p>
                    Confidence: <strong>${data.confidence}</strong>
                </p>
            </div>
        `;

    }
    catch (error) {

        loader.classList.add("hidden");

        resultDiv.innerHTML = `
            <div style="color:red;">
                Error analyzing video
            </div>
        `;

        console.error(error);
    }
}