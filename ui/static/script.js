function uploadVideo() {

    let fileInput = document.getElementById("videoInput");

    if(fileInput.files.length === 0){
        alert("Select a video first");
        return;
    }

    let formData = new FormData();
    formData.append("video", fileInput.files[0]);

    document.getElementById("result").innerHTML = "Processing...";

    fetch("/upload", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {

        document.getElementById("result").innerHTML =
            "<h2>Result: " + data.result + "</h2>" +
            "<h3>Confidence: " + data.confidence + "</h3>";

    })
}