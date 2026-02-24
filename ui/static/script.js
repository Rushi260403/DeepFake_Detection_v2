async function uploadVideo()
{
    const input = document.getElementById("videoInput");

    if (!input.files.length)
    {
        alert("Select video first");
        return;
    }

    const loader = document.getElementById("loader");
    const resultDiv = document.getElementById("result");

    loader.classList.remove("hidden");

    resultDiv.innerHTML = "";

    const formData = new FormData();

    formData.append("video", input.files[0]);

    try
    {
        const response = await fetch("/upload",
        {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        loader.classList.add("hidden");

        showResult(data);
    }
    catch(error)
    {
        loader.classList.add("hidden");

        alert("Error analyzing video");
    }
}


function showResult(data)
{
    const resultDiv = document.getElementById("result");

    let color = "#ffaa00";

    if(data.result == "FAKE") color = "#ff4444";

    if(data.result == "REAL") color = "#00ff88";


    resultDiv.innerHTML =
    `
    <div style="color:${color}">
    <h2>${data.result}</h2>
    <h3>${data.confidence}% confidence</h3>

    <canvas id="chart"></canvas>

    </div>
    `;

    drawChart(data);
}


function drawChart(data)
{
    let fake = data.result=="FAKE"?data.confidence:100-data.confidence;
    let real = data.result=="REAL"?data.confidence:100-data.confidence;

    new Chart(document.getElementById("chart"),
    {
        type: "doughnut",

        data:
        {
            labels:["Fake","Real"],

            datasets:
            [{
                data:[fake,real],

                backgroundColor:["red","green"]
            }]
        }
    });
}


const dropZone = document.getElementById("dropZone");

dropZone.ondragover = e => e.preventDefault();

dropZone.ondrop = e =>
{
    e.preventDefault();

    document.getElementById("videoInput").files = e.dataTransfer.files;
};