async function uploadVideo() {

    const input = document.getElementById("videoInput");
    const file = input.files[0];

    if (!file) {
        alert("Please select a video");
        return;
    }

    const loader = document.getElementById("loader");
    const resultDiv = document.getElementById("result");

    loader.classList.remove("hidden");
    resultDiv.innerHTML = "";

    const formData = new FormData();
    formData.append("video", file);

    try {

        const response = await fetch("/upload", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        loader.classList.add("hidden");

        resultDiv.innerHTML = `
            <h2>Result: ${data.result}</h2>
            <h3>Confidence: ${data.confidence}</h3>
        `;

    }
    catch (error) {

        loader.classList.add("hidden");

        resultDiv.innerHTML = "Error analyzing video";

        console.error(error);
    }
}