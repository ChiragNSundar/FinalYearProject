document.getElementById("uploadForm").addEventListener("submit", async function (event) {
    event.preventDefault();

    const fileInput = document.getElementById("videoFile");
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {
        const response = await fetch("/upload-video/", {
            method: "POST",
            body: formData,
        });

        const result = await response.json();

        if (response.ok) {
            document.getElementById("uploadStatus").innerHTML = `<p class="text-green-600">Video processed successfully!</p>`;
        } else {
            document.getElementById("uploadStatus").innerHTML = `<p class="text-red-600">Error: ${result.detail}</p>`;
        }
    } catch (error) {
        console.error("Error uploading video:", error);
        document.getElementById("uploadStatus").innerHTML = `<p class="text-red-600">Error uploading video.</p>`;
    }
});

document.getElementById("violationsForm").addEventListener("submit", async function (event) {
    event.preventDefault();

    const vehicleNumber = document.getElementById("vehicleNumber").value;
    const formData = new FormData();
    formData.append("vehicle_number", vehicleNumber);

    try {
        const response = await fetch("/get-violations/", {
            method: "POST",
            body: formData
        });

        // Parse the HTML response
        const htmlText = await response.text();
        document.body.innerHTML = htmlText;
    } catch (error) {
        console.error("Error fetching violations:", error);
        alert("Error fetching violations.");
    }
});