const generateButton = document.getElementById('generate-caption');
const statusText = document.getElementById('status');
const preview = document.getElementById('preview');
const imageInput = document.getElementById('image-upload');
const captionOutput = document.getElementById('caption-output');

// Show preview of selected image
imageInput.addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            preview.src = e.target.result;
            preview.style.display = 'block';
            statusText.textContent = 'Ready for caption generation';
            statusText.style.color = 'green';
        };
        reader.readAsDataURL(file);
    }
});

// Generate Caption button functionality
generateButton.addEventListener('click', () => {
    const imageInputFile = document.getElementById('image-upload').files[0];

    if (!imageInputFile) {
        alert('Please upload an image first!');
        return;
    }

    statusText.textContent = 'Generating...';
    statusText.style.color = 'orange';

    const formData = new FormData();
    formData.append('file', imageInputFile);

    fetch('http://127.0.0.1:5000/generate-caption/', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.captions && data.captions.length > 0) {
            captionOutput.textContent = data.captions.join('\n');
            statusText.textContent = 'Caption Generated';
            statusText.style.color = 'green';
        } else {
            captionOutput.textContent = 'No captions generated.';
            statusText.textContent = 'Error generating caption';
            statusText.style.color = 'red';
        }
    })
    .catch(error => {
        console.error('Error:', error);
        captionOutput.textContent = 'Failed to generate caption';
        statusText.textContent = 'Error generating caption';
        statusText.style.color = 'red';
    });
});

// Capture from camera
function captureFromCamera() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.capture = 'environment';
    input.onchange = function(event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                preview.src = e.target.result;
                preview.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }
    };
    input.click();
}

// Preview function when manually uploading
function previewImage(event) {
    const reader = new FileReader();
    reader.onload = function() {
        preview.src = reader.result;
        preview.style.display = 'block';
        statusText.textContent = 'Ready to generate caption';
        statusText.style.color = 'green';
    };
    reader.readAsDataURL(event.target.files[0]);
}
