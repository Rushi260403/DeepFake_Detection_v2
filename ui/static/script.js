function uploadVideo(){

let file = document.getElementById("videoInput").files[0];

let formData = new FormData();

formData.append("video", file);

document.getElementById("loader").classList.remove("hidden");

fetch("/upload",{

method:"POST",
body:formData

})
.then(res=>res.json())
.then(data=>{

document.getElementById("loader").classList.add("hidden");

document.getElementById("result").innerHTML =
"<h2>"+data.result+"</h2>"+
"<h3>"+data.confidence+"</h3>";

let framesHTML="";

data.frames.forEach(f=>{

framesHTML += `<img src="${f}">`;

});

document.getElementById("frames").innerHTML=framesHTML;

});

}