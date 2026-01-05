// Matrix Rain Animation
const canvas = document.getElementById('matrixCanvas');
const ctx = canvas.getContext('2d');

// Set canvas size
function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}
resizeCanvas();
window.addEventListener('resize', resizeCanvas);

// Matrix characters
const chars = 'アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ';
const charArray = chars.split('');

// Rain configuration
const fontSize = 14;
const columns = Math.floor(canvas.width / fontSize);
const drops = [];

// Initialize drops
for (let i = 0; i < columns; i++) {
    drops[i] = Math.random() * -100;
}

// Draw function
function drawMatrix() {
    // Semi-transparent black background for trail effect
    ctx.fillStyle = 'rgba(10, 10, 10, 0.05)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Set text style
    ctx.fillStyle = '#00ff41';
    ctx.font = `${fontSize}px monospace`;

    // Draw each drop
    for (let i = 0; i < drops.length; i++) {
        // Select random character
        const char = charArray[Math.floor(Math.random() * charArray.length)];
        
        // Varying green shades
        const green = Math.floor(Math.random() * 155) + 100;
        ctx.fillStyle = `rgb(0, ${green}, 65)`;
        
        // Draw character
        ctx.fillText(char, i * fontSize, drops[i] * fontSize);

        // Reset drop to top randomly
        if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) {
            drops[i] = 0;
        }

        // Increment drop
        drops[i]++;
    }
}

// Animation loop
let animationId;
function animateMatrix() {
    drawMatrix();
    animationId = requestAnimationFrame(animateMatrix);
}

// Start matrix animation
animateMatrix();

// Application State
let analysisCount = 0;
const API_BASE = 'http://localhost:8000';

// DOM Elements
const newsInput = document.getElementById('newsInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const defaultState = document.getElementById('defaultState');
const loadingState = document.getElementById('loadingState');
const fakeResult = document.getElementById('fakeResult');
const realResult = document.getElementById('realResult');
const errorState = document.getElementById('errorState');
const modelStatus = document.getElementById('modelStatus');
const apiStatus = document.getElementById('apiStatus');
const analysisCountEl = document.getElementById('analysisCount');
const loadingStep = document.getElementById('loadingStep');
const fakeConfidence = document.getElementById('fakeConfidence');
const realConfidence = document.getElementById('realConfidence');
const fakeConfidenceBar = document.getElementById('fakeConfidenceBar');
const realConfidenceBar = document.getElementById('realConfidenceBar');
const errorMessage = document.getElementById('errorMessage');

// Loading steps for animation
const loadingSteps = [
    'Initializing neural network...',
    'Preprocessing input text...',
    'Extracting TF-IDF features...',
    'Analyzing linguistic patterns...',
    'Cross-referencing with trained data...',
    'Computing probability distribution...',
    'Finalizing detection result...'
];

// Show specific state
function showState(state) {
    defaultState.classList.add('hidden');
    loadingState.classList.add('hidden');
    fakeResult.classList.add('hidden');
    realResult.classList.add('hidden');
    errorState.classList.add('hidden');
    
    state.classList.remove('hidden');
}

// Update analysis count
function updateAnalysisCount() {
    analysisCount++;
    analysisCountEl.textContent = analysisCount;
}

// Simulate loading steps
async function simulateLoading() {
    const totalSteps = loadingSteps.length;
    for (let i = 0; i < totalSteps; i++) {
        loadingStep.textContent = loadingSteps[i];
        await new Promise(resolve => setTimeout(resolve, 400 + Math.random() * 300));
    }
}

// Check API status
async function checkApiStatus() {
    try {
        const response = await fetch(`${API_BASE}/status`);
        const data = await response.json();
        
        if (data.model_loaded) {
            modelStatus.textContent = 'READY';
            modelStatus.classList.remove('loading');
            modelStatus.classList.add('online');
        } else {
            modelStatus.textContent = 'NOT_LOADED';
            modelStatus.classList.remove('loading');
            modelStatus.classList.add('error');
        }
        
        apiStatus.textContent = 'CONNECTED';
        apiStatus.classList.remove('loading');
        apiStatus.classList.add('online');
        
        return data.model_loaded;
    } catch (error) {
        modelStatus.textContent = 'ERROR';
        modelStatus.classList.remove('loading');
        modelStatus.classList.add('error');
        
        apiStatus.textContent = 'OFFLINE';
        apiStatus.classList.remove('loading');
        apiStatus.classList.add('error');
        
        return false;
    }
}

// Make prediction
async function makePrediction(text) {
    try {
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text }),
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Prediction failed');
        }

        return await response.json();
    } catch (error) {
        throw error;
    }
}

// Animate confidence bar
function animateConfidence(element, bar, percentage) {
    setTimeout(() => {
        bar.style.width = `${percentage}%`;
    }, 100);
}

// Handle analyze button click
async function handleAnalyze() {
    const text = newsInput.value.trim();
    
    if (!text) {
        showState(errorState);
        errorMessage.textContent = 'Please enter some text to analyze.';
        return;
    }
    
    // Show loading state
    showState(loadingState);
    analyzeBtn.classList.add('loading');
    analyzeBtn.disabled = true;
    
    try {
        // Simulate loading steps
        await simulateLoading();
        
        // Make prediction
        const result = await makePrediction(text);
        
        // Update UI with result
        if (result.prediction === 'FAKE') {
            fakeConfidence.textContent = `${(result.confidence * 100).toFixed(1)}%`;
            animateConfidence(fakeConfidence, fakeConfidenceBar, result.confidence * 100);
            showState(fakeResult);
        } else {
            realConfidence.textContent = `${(result.confidence * 100).toFixed(1)}%`;
            animateConfidence(realConfidence, realConfidenceBar, result.confidence * 100);
            showState(realResult);
        }
        
        updateAnalysisCount();
    } catch (error) {
        showState(errorState);
        errorMessage.textContent = error.message || 'An error occurred during analysis. Make sure the API server is running.';
    } finally {
        analyzeBtn.classList.remove('loading');
        analyzeBtn.disabled = false;
    }
}

// Event listeners
analyzeBtn.addEventListener('click', handleAnalyze);

// Allow Enter key to submit (with Ctrl/Cmd)
newsInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
        handleAnalyze();
    }
});

// Initialize on page load
document.addEventListener('DOMContentLoaded', async () => {
    await checkApiStatus();
});

// Periodic API status check
setInterval(checkApiStatus, 10000);

// Add glitch effect on hover for title
const title = document.querySelector('.title');
if (title) {
    title.addEventListener('mouseenter', () => {
        title.style.animation = 'none';
        setTimeout(() => {
            title.style.animation = 'glitch 0.3s ease-in-out';
        }, 10);
    });
}

// Typing effect for placeholder
const placeholderText = 'Enter news article text to analyze...';
let placeholderIndex = 0;
let isDeleting = false;
let placeholderTimeout;

function typePlaceholder() {
    if (isDeleting) {
        newsInput.setAttribute('placeholder', placeholderText.substring(0, placeholderIndex - 1));
        placeholderIndex--;
    } else {
        newsInput.setAttribute('placeholder', placeholderText.substring(0, placeholderIndex + 1));
        placeholderIndex++;
    }
    
    if (!isDeleting && placeholderIndex === placeholderText.length) {
        isDeleting = true;
        placeholderTimeout = setTimeout(typePlaceholder, 2000);
    } else if (isDeleting && placeholderIndex === 0) {
        isDeleting = false;
        placeholderTimeout = setTimeout(typePlaceholder, 500);
    } else {
        placeholderTimeout = setTimeout(typePlaceholder, isDeleting ? 30 : 80);
    }
}

// Start typing effect after a delay
setTimeout(typePlaceholder, 2000);

// Matrix rain intensity based on mouse position
document.addEventListener('mousemove', (e) => {
    const intensity = 1 - (e.clientY / window.innerHeight);
    ctx.globalAlpha = Math.max(0.05, Math.min(0.2, intensity * 0.2));
});
