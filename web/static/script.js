let currentSessionId = null;

async function uploadData() {
    const fileInput = document.getElementById('dataFile');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select a JSON file');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/upload-data', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        currentSessionId = data.session_id;
        document.getElementById('sessionId').innerHTML = `<p>✅ Session: ${currentSessionId}</p>`;
        document.getElementById('trainBtn').disabled = false;
        
        // Calculate file size
        const sizeKB = (file.size / 1024).toFixed(2);
        document.getElementById('dataSize').textContent = sizeKB;
        
    } catch (error) {
        console.error('Upload failed:', error);
        alert('Upload failed');
    }
}

async function startTraining() {
    if (!currentSessionId) {
        alert('Please upload data first');
        return;
    }
    
    const trainBtn = document.getElementById('trainBtn');
    const progressBar = document.getElementById('progressBar');
    const progressFill = document.querySelector('.progress-fill');
    
    trainBtn.disabled = true;
    progressBar.style.display = 'block';
    
    try {
        const response = await fetch(`/start-training/${currentSessionId}`, {
            method: 'POST'
        });
        
        // Simulate progress
        let progress = 0;
        const interval = setInterval(() => {
            progress += 10;
            progressFill.style.width = `${progress}%`;
            progressFill.textContent = `${progress}%`;
            if (progress >= 100) clearInterval(interval);
        }, 500);
        
        const result = await response.json();
        clearInterval(interval);
        progressFill.style.width = '100%';
        progressFill.textContent = 'Complete!';
        
        document.getElementById('trainingStatus').innerHTML = '<p>✅ Training completed! Model ready for inference.</p>';
        
    } catch (error) {
        console.error('Training failed:', error);
        alert('Training failed');
    }
}

async function generate() {
    const prompt = document.getElementById('prompt').value;
    if (!prompt) {
        alert('Please enter a prompt');
        return;
    }
    
    const formData = new FormData();
    formData.append('prompt', prompt);
    formData.append('use_personalized', 'true');
    
    try {
        const response = await fetch('/generate', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        document.getElementById('response').innerHTML = `<strong>Response:</strong><br>${data.response}`;
        
    } catch (error) {
        console.error('Generation failed:', error);
        alert('Generation failed');
    }
}