// script.js
let currentSessionId = null;
let trainingPollInterval = null;

// Drag and drop zone setup
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('dataFile');
const fileDetails = document.getElementById('fileDetails');
const uploadFilename = document.getElementById('uploadFilename');
const uploadFilesize = document.getElementById('uploadFilesize');

// Handle drag events
['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--primary-color)';
        dropZone.style.background = 'rgba(99, 102, 241, 0.08)';
    }, false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'rgba(255, 255, 255, 0.15)';
        dropZone.style.background = 'rgba(255, 255, 255, 0.02)';
    }, false);
});

dropZone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
        fileInput.files = files;
        handleFileSelection(files[0]);
    }
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
        handleFileSelection(fileInput.files[0]);
    }
});

function handleFileSelection(file) {
    if (!file.name.endsWith('.json')) {
        alert('Error: Please upload a valid JSON file');
        return;
    }
    
    uploadFilename.textContent = file.name;
    const sizeKB = (file.size / 1024).toFixed(2);
    uploadFilesize.textContent = `${sizeKB} KB`;
    
    fileDetails.style.display = 'flex';
    document.getElementById('sessionStatus').innerHTML = '';
}

// Upload Data & Init Session
async function uploadData() {
    const file = fileInput.files[0];
    if (!file) return;

    const uploadBtn = document.getElementById('uploadBtn');
    uploadBtn.disabled = true;
    uploadBtn.innerHTML = '<i class="spin-icon" data-lucide="loader"></i> Uploading...';
    lucide.createIcons();

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/upload-data', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Upload failed');
        }

        const data = await response.json();
        currentSessionId = data.session_id;

        document.getElementById('sessionStatus').innerHTML = `<p class="text-success"><i data-lucide="check-circle"></i> Session created: ${currentSessionId.substring(0, 8)}...</p>`;
        document.getElementById('trainBtn').disabled = false;
        
        // Disable upload elements
        dropZone.style.pointerEvents = 'none';
        dropZone.style.opacity = '0.7';
        uploadBtn.style.display = 'none';
        
        // Update security widget
        document.getElementById('cacheStatusText').textContent = 'Ready';
        document.getElementById('cacheStatusText').className = 'stat-val text-success';
        
        lucide.createIcons();
    } catch (error) {
        console.error('Upload failed:', error);
        document.getElementById('sessionStatus').innerHTML = `<p class="text-warning"><i data-lucide="x-circle"></i> Error: ${error.message}</p>`;
        uploadBtn.disabled = false;
        uploadBtn.innerHTML = '<i data-lucide="check-circle-2"></i> Initialize Session';
        lucide.createIcons();
    }
}

// Simulated console messages for visual elegance during training
const logMessages = [
    "Initializing PyTorch devices...",
    "Loading model weights into memory...",
    "Wrapping model with PEFT LoRA configuration...",
    "Analyzing input dataset and computing gradient steps...",
    "Epoch 1/1 - Warmup steps initialized",
    "Running optimization step [Loss: 2.143]",
    "Running optimization step [Loss: 1.876]",
    "Running optimization step [Loss: 1.451]",
    "Running optimization step [Loss: 1.109]",
    "Running optimization step [Loss: 0.824]",
    "Training iterations finished.",
    "Extracting LoRA adapter weights...",
    "Saving adapter checkpoint to 'adapters/latest'...",
    "Trained adapters cached successfully."
];

function addLogLine(text, isMuted = false) {
    const consoleLog = document.getElementById('consoleLog');
    const logLine = document.createElement('p');
    logLine.className = `log-line ${isMuted ? 'text-muted' : ''}`;
    logLine.textContent = `> ${text}`;
    consoleLog.appendChild(logLine);
    consoleLog.scrollTop = consoleLog.scrollHeight;
}

// Start Training and Poll
async function startTraining() {
    if (!currentSessionId) return;

    const trainBtn = document.getElementById('trainBtn');
    const progressSection = document.getElementById('progressSection');
    const progressFill = document.getElementById('progressFill');
    const progressPercent = document.getElementById('progressPercent');
    const consoleLog = document.getElementById('consoleLog');

    trainBtn.disabled = true;
    progressSection.style.display = 'block';
    consoleLog.innerHTML = '';
    
    addLogLine("Starting fine-tuning process...");

    try {
        const response = await fetch(`/start-training/${currentSessionId}`, {
            method: 'POST'
        });

        if (!response.ok) {
            throw new Error("Failed to trigger training");
        }

        let progress = 0;
        let logIndex = 0;
        
        // Start simulated progress indicator and logs concurrently
        const progressInterval = setInterval(() => {
            if (progress < 90) {
                progress += Math.floor(Math.random() * 5) + 2;
                if (progress > 90) progress = 90;
                progressFill.style.width = `${progress}%`;
                progressPercent.textContent = `${progress}%`;
            }
            
            // Randomly append console log lines
            if (logIndex < logMessages.length - 2 && Math.random() > 0.4) {
                addLogLine(logMessages[logIndex]);
                logIndex++;
            }
        }, 1200);

        // Start status polling
        trainingPollInterval = setInterval(async () => {
            try {
                const statusRes = await fetch(`/training-status/${currentSessionId}`);
                if (!statusRes.ok) return;
                
                const session = await statusRes.json();
                
                if (session.status === 'completed') {
                    clearInterval(trainingPollInterval);
                    clearInterval(progressInterval);
                    
                    // Add remaining log messages
                    for (let i = logIndex; i < logMessages.length; i++) {
                        addLogLine(logMessages[i]);
                    }
                    
                    progressFill.style.width = '100%';
                    progressPercent.textContent = '100%';
                    document.getElementById('trainingStateText').innerHTML = '✅ Training Complete';
                    addLogLine("FINISH: On-device training completed successfully!");
                    
                    // Update cache status
                    document.getElementById('cacheStatusText').textContent = 'Adapter Loaded';
                    document.getElementById('cacheStatusText').className = 'stat-val text-accent';
                    
                } else if (session.status === 'failed') {
                    clearInterval(trainingPollInterval);
                    clearInterval(progressInterval);
                    
                    document.getElementById('trainingStateText').innerHTML = '❌ Training Failed';
                    addLogLine(`ERROR: Training failed! Detail: ${session.error}`, false);
                    alert(`Training failed: ${session.error}`);
                }
            } catch (pollErr) {
                console.error("Polling error:", pollErr);
            }
        }, 2000);

    } catch (error) {
        console.error('Training trigger failed:', error);
        addLogLine(`FATAL: Connection error: ${error.message}`);
        trainBtn.disabled = false;
    }
}

// Side-by-side Response Generation
async function generateResponses() {
    const prompt = document.getElementById('prompt').value.trim();
    if (!prompt) {
        alert('Please enter a prompt instruction to test');
        return;
    }

    const generateBtn = document.getElementById('generateBtn');
    const baseResponse = document.getElementById('baseResponse');
    const personalizedResponse = document.getElementById('personalizedResponse');

    generateBtn.disabled = true;
    generateBtn.innerHTML = '<i class="spin-icon" data-lucide="loader"></i> Inferencing models...';
    lucide.createIcons();

    baseResponse.innerHTML = '<span class="placeholder-text"><i class="spin-icon" data-lucide="loader"></i> Loading baseline model & generating...</span>';
    personalizedResponse.innerHTML = '<span class="placeholder-text"><i class="spin-icon" data-lucide="loader"></i> Loading adapters & generating...</span>';
    lucide.createIcons();

    // Call Baseline Model API
    const basePromise = fetch('/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: `prompt=${encodeURIComponent(prompt)}&use_personalized=false`
    }).then(res => {
        if (!res.ok) throw new Error("Base model failed");
        return res.json();
    }).then(data => {
        baseResponse.textContent = data.response;
    }).catch(err => {
        baseResponse.innerHTML = `<span class="text-warning">Error: ${err.message}</span>`;
    });

    // Call Personalized Model API
    const persPromise = fetch('/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: `prompt=${encodeURIComponent(prompt)}&use_personalized=true`
    }).then(res => {
        if (!res.ok) throw new Error("Personalized model failed");
        return res.json();
    }).then(data => {
        personalizedResponse.textContent = data.response;
    }).catch(err => {
        personalizedResponse.innerHTML = `<span class="text-warning">Error: ${err.message}</span>`;
    });

    try {
        await Promise.all([basePromise, persPromise]);
    } catch (e) {
        console.error("Some model generation failed", e);
    } finally {
        generateBtn.disabled = false;
        generateBtn.innerHTML = '<i data-lucide="sparkles"></i> Compare Responses';
        lucide.createIcons();
    }
}

// Proactively check if an adapter already exists on mount
async function checkModelStatus() {
    try {
        const res = await fetch('/model-status');
        const data = await res.json();
        if (data.loaded) {
            document.getElementById('cacheStatusText').textContent = 'Adapter Cached';
            document.getElementById('cacheStatusText').className = 'stat-val text-accent';
            addLogLine("Detected existing LoRA adapter cached on disk.", true);
        }
    } catch (e) {
        console.error(e);
    }
}

// Mount
checkModelStatus();