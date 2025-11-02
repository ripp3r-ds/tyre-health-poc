// FOR LOCAL DEVELOPMENT: This is the address of your Python FastAPI server
const API_BASE_URL = "http://127.0.0.1:8000";

// --- Grab the HTML elements we need to work with ---
const imageInput = document.getElementById('tyreImageInput');
const imagePreviewContainer = document.getElementById('imagePreviewContainer');
const resultsContainer = document.getElementById('resultsContainer');
const resultsDiv = document.getElementById('results');

// --- Show a preview of the image when the user selects one ---
imageInput.addEventListener('change', () => {
    const file = imageInput.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (event) => {
            // Create an <img> element and show it in the preview container
            imagePreviewContainer.innerHTML = `<p>Image Preview:</p><img src="${event.target.result}" alt="Tyre Preview">`;
        };
        reader.readAsDataURL(file);
    }
});


// --- This is the main function that talks to our backend ---
async function predict(modelType) {
    const file = imageInput.files[0];
    if (!file) {
        alert("Please select an image first!");
        return;
    }

    // Show a loading message
    resultsContainer.style.display = 'block';
    resultsDiv.textContent = 'Analyzing image... 🧠';

    // FormData is the standard way to send files to a server
    const formData = new FormData();
    formData.append("file", file);

    try {
        // Use the 'fetch' function to send the data to our Python API
        // The URL is constructed based on which button was clicked ('condition' or 'pressure')
        const response = await fetch(`${API_BASE_URL}/predict/${modelType}`, {
            method: 'POST',
            body: formData,
        });

        // Check if the server responded with an error
        if (!response.ok) {
            const errorData = await response.json(); // Try to get error details from API
            throw new Error(`API Error: ${errorData.detail || response.statusText}`);
        }

        // If successful, get the JSON result from the response
        const data = await response.json();
        
        // Display the result in a nicely formatted way
        resultsDiv.textContent = JSON.stringify(data, null, 2);

    } catch (error) {
        console.error("Error:", error);
        resultsDiv.textContent = `An error occurred: \n${error.message}`;
    }
}