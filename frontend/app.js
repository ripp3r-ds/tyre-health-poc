document.addEventListener("DOMContentLoaded", () => {
    // Get references to all the interactive elements on the page
    const conditionBtn = document.getElementById("condition-btn");
    const pressureBtn = document.getElementById("pressure-btn");
    const fileInput = document.getElementById("file-input");
    
    const resultArea = document.getElementById("result-area");
    const imagePreview = document.getElementById("image-preview");
    const spinner = document.getElementById("spinner");
    const resultsContent = document.getElementById("results-content");
    
    const predictionValue = document.getElementById("prediction-value");
    const confidenceValue = document.getElementById("confidence-value");

    // When the "Check Condition" button is clicked...
    conditionBtn.addEventListener("click", () => {
        // ...store which model we want to use and trigger the file input
        fileInput.dataset.modelType = "condition";
        fileInput.click();
    });

    // When the "Check Pressure" button is clicked...
    pressureBtn.addEventListener("click", () => {
        // ...store which model we want to use and trigger the file input
        fileInput.dataset.modelType = "pressure";
        fileInput.click();
    });

    // This event fires when the user selects a file
    fileInput.addEventListener("change", (event) => {
        const file = event.target.files[0];
        if (!file) return;

        // Get the model type we stored earlier
        const modelType = fileInput.dataset.modelType;

        // Show the result area and the uploaded image
        resultArea.style.display = "flex";
        imagePreview.src = URL.createObjectURL(file);

        // Show the spinner and hide any previous results
        spinner.style.display = "block";
        resultsContent.style.display = "none";
        
        // Call the main analysis function
        analyzeImage(file, modelType);
    });

    async function analyzeImage(file, modelType) {
        // Point to the local FastAPI backend
        // Change this URL if your backend runs on a different port or IP
        const apiBaseUrl = "http://localhost:8000";
        const apiEndpoint = `${apiBaseUrl}/predict/${modelType}`;

        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch(apiEndpoint, {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                // If the server returns an error, show it
                const errorData = await response.json();
                throw new Error(errorData.detail || "An unknown error occurred.");
            }

            const data = await response.json();

            // Display the results
            displayResults(data);

        } catch (error) {
            // Display any network or server errors
            displayError(error.message);
        }
    }

    function displayResults(data) {
        // Hide the spinner and show the results content
        spinner.style.display = "none";
        resultsContent.style.display = "block";

        const prediction = data.prediction;
        const confidence = (data.confidence * 100).toFixed(2); // Convert to percentage

        predictionValue.textContent = prediction.charAt(0).toUpperCase() + prediction.slice(1); // Capitalize
        confidenceValue.textContent = `${confidence}%`;

        // Change color based on prediction
        if (prediction === 'good' || prediction === 'full') {
            predictionValue.style.color = "var(--success-color)";
        } else {
            predictionValue.style.color = "var(--warning-color)";
        }
    }

    function displayError(errorMessage) {
        spinner.style.display = "none";
        resultsContent.style.display = "block";
        predictionValue.style.color = "var(--warning-color)";
        predictionValue.textContent = "Error";
        confidenceValue.textContent = errorMessage;
    }
});