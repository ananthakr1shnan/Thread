document.addEventListener('DOMContentLoaded', function() {
    // Update matrix size display when slider changes
    const matrixSizeSlider = document.getElementById('matrixSize');
    const matrixSizeValue = document.getElementById('matrixSizeValue');
    
    matrixSizeSlider.addEventListener('input', function() {
        matrixSizeValue.textContent = this.value;
    });
    
    // Toggle advanced options
    const advancedToggle = document.getElementById('advancedToggle');
    const advancedOptions = document.getElementById('advancedOptions');
    
    advancedToggle.addEventListener('change', function() {
        advancedOptions.style.display = this.checked ? 'block' : 'none';
    });
    
    // Handle prediction button click
    const predictBtn = document.getElementById('predictBtn');
    predictBtn.addEventListener('click', async function() {
        // Show loading state
        predictBtn.disabled = true;
        predictBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Predicting...';
        
        // Gather input values
        const typeOp = document.getElementById('typeOp').value;
        const matrixSize = parseInt(document.getElementById('matrixSize').value);
        const varType = document.getElementById('varType').value;
        const matrixType = document.getElementById('matrixType').value;
        
        // Create request data
        const requestData = {
            type_op: typeOp,
            matrix_size: matrixSize,
            var_type: varType,
            matrix_type: matrixType
        };
        
        // Include advanced options if enabled
        if (advancedToggle.checked) {
            requestData.is_iterative = parseInt(document.querySelector('input[name="isIterative"]:checked').value);
            requestData.memory_pattern = parseInt(document.querySelector('input[name="memoryPattern"]:checked').value);
        }
        
        try {
            // Send API request
            const response = await fetch('/api', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });
            
            if (!response.ok) {
                throw new Error('API request failed');
            }
            
            const data = await response.json();
            
            // Update results
            document.getElementById('threadCount').textContent = data.optimal_threads;
            document.getElementById('exeTime').textContent = data.estimated_features.exeTime.toFixed(6) + ' s';
            document.getElementById('numOp').textContent = data.estimated_features.numOp.toLocaleString();
            document.getElementById('numVar').textContent = data.estimated_features.numVar;
            document.getElementById('complexity').textContent = data.estimated_features.complexity.toFixed(2);
            document.getElementById('isIterativeResult').textContent = data.estimated_features.isIterative ? 'Yes' : 'No';
            document.getElementById('memoryPatternResult').textContent = data.estimated_features.memoryPattern ? 'Yes' : 'No';
            
            // Show results card
            document.getElementById('resultsCard').style.display = 'block';
            
            // Scroll to results
            document.getElementById('resultsCard').scrollIntoView({ behavior: 'smooth' });
            
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred during prediction. Please try again.');
        } finally {
            // Reset button state
            predictBtn.disabled = false;
            predictBtn.textContent = 'Predict Optimal Thread Count';
        }
    });
});