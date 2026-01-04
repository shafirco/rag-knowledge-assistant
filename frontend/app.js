/**
 * RAG Knowledge Assistant Frontend Application
 */

class RAGApp {
    constructor() {
        this.baseUrl = '/api/v1';
        this.init();
    }

    init() {
        this.bindEvents();
        this.loadStats();
        this.checkHealth();
    }

    bindEvents() {
        // Upload functionality
        document.getElementById('uploadBtn').addEventListener('click', () => this.uploadDocument());
        document.getElementById('fileInput').addEventListener('change', () => this.onFileSelected());

        // Query functionality
        document.getElementById('askBtn').addEventListener('click', () => this.askQuestion());
        document.getElementById('questionInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                this.askQuestion();
            }
        });

        // Stats functionality
        document.getElementById('refreshStatsBtn').addEventListener('click', () => this.loadStats());
        document.getElementById('clearKbBtn').addEventListener('click', () => this.clearKnowledgeBase());
    }

    onFileSelected() {
        const fileInput = document.getElementById('fileInput');
        const uploadBtn = document.getElementById('uploadBtn');
        
        if (fileInput.files.length > 0) {
            const file = fileInput.files[0];
            uploadBtn.textContent = `Upload "${file.name}"`;
        } else {
            uploadBtn.textContent = 'Upload Document';
        }
    }

    async uploadDocument() {
        const fileInput = document.getElementById('fileInput');
        const uploadBtn = document.getElementById('uploadBtn');
        const uploadStatus = document.getElementById('uploadStatus');

        if (!fileInput.files.length) {
            this.showStatus('uploadStatus', 'Please select a file to upload', 'error');
            return;
        }

        const file = fileInput.files[0];
        
        // Validate file type
        const validExtensions = ['.txt', '.md'];
        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
        
        if (!validExtensions.includes(fileExtension)) {
            this.showStatus('uploadStatus', 'Only .txt and .md files are supported', 'error');
            return;
        }

        // Show loading state
        const originalText = uploadBtn.textContent;
        uploadBtn.disabled = true;
        uploadBtn.innerHTML = '<span class="spinner"></span> Uploading...';
        this.showStatus('uploadStatus', 'Processing document...', 'loading');

        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch(`${this.baseUrl}/upload`, {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (response.ok) {
                this.showStatus('uploadStatus', 
                    `Success! Created ${result.chunks_created} chunks from "${result.filename}"`, 
                    'success'
                );
                
                // Reset form
                fileInput.value = '';
                uploadBtn.textContent = 'Upload Document';
                
                // Refresh stats
                this.loadStats();
            } else {
                throw new Error(result.detail || 'Upload failed');
            }

        } catch (error) {
            console.error('Upload error:', error);
            this.showStatus('uploadStatus', `Upload failed: ${error.message}`, 'error');
        } finally {
            uploadBtn.disabled = false;
            uploadBtn.textContent = originalText;
        }
    }

    async askQuestion() {
        const questionInput = document.getElementById('questionInput');
        const askBtn = document.getElementById('askBtn');
        const queryStatus = document.getElementById('queryStatus');
        const resultsSection = document.getElementById('resultsSection');

        const question = questionInput.value.trim();
        if (!question) {
            this.showStatus('queryStatus', 'Please enter a question', 'error');
            return;
        }

        // Get options
        const maxChunks = parseInt(document.getElementById('maxChunks').value) || 5;
        const similarityThreshold = parseFloat(document.getElementById('similarityThreshold').value) || 0.7;

        // Show loading state
        const originalText = askBtn.textContent;
        askBtn.disabled = true;
        askBtn.innerHTML = '<span class="spinner"></span> Processing...';
        this.showStatus('queryStatus', 'Searching knowledge base...', 'loading');
        resultsSection.style.display = 'none';

        try {
            const response = await fetch(`${this.baseUrl}/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question: question,
                    max_chunks: maxChunks,
                    similarity_threshold: similarityThreshold
                })
            });

            const result = await response.json();

            if (response.ok) {
                this.displayResults(result);
                this.showStatus('queryStatus', 'Query completed successfully', 'success');
                resultsSection.style.display = 'block';
                resultsSection.scrollIntoView({ behavior: 'smooth' });
            } else {
                throw new Error(result.detail || 'Query failed');
            }

        } catch (error) {
            console.error('Query error:', error);
            this.showStatus('queryStatus', `Query failed: ${error.message}`, 'error');
        } finally {
            askBtn.disabled = false;
            askBtn.textContent = originalText;
        }
    }

    displayResults(result) {
        const answerArea = document.getElementById('answerArea');
        const sourcesArea = document.getElementById('sourcesArea');
        const queryInfo = document.getElementById('queryInfo');

        // Display answer
        answerArea.textContent = result.answer;

        // Display sources
        sourcesArea.innerHTML = '';
        
        if (result.sources && result.sources.length > 0) {
            result.sources.forEach((source, index) => {
                const sourceDiv = document.createElement('div');
                sourceDiv.className = 'source-item';
                
                sourceDiv.innerHTML = `
                    <div class="source-header">
                        <span class="source-filename">${source.filename} (Chunk ${source.chunk_index + 1})</span>
                        <span class="source-score">${(source.similarity_score * 100).toFixed(1)}%</span>
                    </div>
                    <div class="source-preview">${source.content_preview}</div>
                `;
                
                sourcesArea.appendChild(sourceDiv);
            });
        } else {
            sourcesArea.innerHTML = '<p class="text-center" style="color: #718096;">No sources found</p>';
        }

        // Display query info
        queryInfo.textContent = `Processing time: ${result.processing_time_ms}ms | Retrieved chunks: ${result.retrieved_chunks}`;
    }

    async loadStats() {
        try {
            const response = await fetch(`${this.baseUrl}/stats`);
            const result = await response.json();

            if (response.ok && result.status === 'success') {
                const stats = result.data;
                document.getElementById('docCount').textContent = stats.total_documents || 0;
                document.getElementById('chunkCount').textContent = stats.total_chunks || 0;
                document.getElementById('vectorCount').textContent = stats.index_size || 0;
            } else {
                throw new Error('Failed to load stats');
            }
        } catch (error) {
            console.error('Stats loading error:', error);
            document.getElementById('docCount').textContent = '?';
            document.getElementById('chunkCount').textContent = '?';
            document.getElementById('vectorCount').textContent = '?';
        }
    }

    async clearKnowledgeBase() {
        if (!confirm('Are you sure you want to clear all documents? This action cannot be undone.')) {
            return;
        }

        try {
            const response = await fetch(`${this.baseUrl}/clear`, {
                method: 'DELETE'
            });

            const result = await response.json();

            if (response.ok && result.status === 'success') {
                alert('Knowledge base cleared successfully');
                this.loadStats();
                
                // Hide results if shown
                document.getElementById('resultsSection').style.display = 'none';
            } else {
                throw new Error(result.message || 'Clear operation failed');
            }
        } catch (error) {
            console.error('Clear error:', error);
            alert(`Failed to clear knowledge base: ${error.message}`);
        }
    }

    async checkHealth() {
        const healthStatus = document.getElementById('healthStatus');
        
        try {
            const response = await fetch(`${this.baseUrl}/health`);
            const result = await response.json();

            if (response.ok) {
                const status = result.status;
                healthStatus.textContent = `System Status: ${status.charAt(0).toUpperCase() + status.slice(1)}`;
                healthStatus.className = status === 'healthy' ? 'health-healthy' : 'health-unhealthy';
            } else {
                throw new Error('Health check failed');
            }
        } catch (error) {
            console.error('Health check error:', error);
            healthStatus.textContent = 'System Status: Unknown';
            healthStatus.className = 'health-unhealthy';
        }
    }

    showStatus(elementId, message, type) {
        const statusElement = document.getElementById(elementId);
        statusElement.textContent = message;
        statusElement.className = `status ${type}`;
        statusElement.style.display = 'block';

        // Auto-hide success messages after 5 seconds
        if (type === 'success') {
            setTimeout(() => {
                statusElement.style.display = 'none';
            }, 5000);
        }
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new RAGApp();
});

// Add some helpful keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl+U for upload focus
    if (e.ctrlKey && e.key === 'u') {
        e.preventDefault();
        document.getElementById('fileInput').focus();
    }
    
    // Ctrl+Q for question focus
    if (e.ctrlKey && e.key === 'q') {
        e.preventDefault();
        document.getElementById('questionInput').focus();
    }
});

// Add file drag and drop functionality
document.addEventListener('DOMContentLoaded', () => {
    const uploadArea = document.querySelector('.upload-area');
    const fileInput = document.getElementById('fileInput');

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        uploadArea.style.backgroundColor = '#edf2f7';
        uploadArea.style.border = '2px dashed #667eea';
    }

    function unhighlight(e) {
        uploadArea.style.backgroundColor = '';
        uploadArea.style.border = '';
    }

    uploadArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;

        if (files.length > 0) {
            fileInput.files = files;
            const event = new Event('change', { bubbles: true });
            fileInput.dispatchEvent(event);
        }
    }
});
