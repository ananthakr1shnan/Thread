from flask import Flask, request, jsonify, send_from_directory
import os
import sys
from api.thread_predictor import ThreadPredictor

app = Flask(__name__, static_folder='public')

# Initialize the predictor with the model path
model_path = os.path.join(os.path.dirname(__file__), "api", "final_pipeline.pkl")
predictor = ThreadPredictor(model_path=model_path)

@app.route('/')
def index():
    return send_from_directory('public', 'index.html')

@app.route('/health')
def health_check():
    # Simple health check endpoint that Railway will use
    return jsonify({"status": "healthy"}), 200

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('public', path)

@app.route('/api', methods=['GET'])
def api_status():
    return jsonify({
        'status': 'Thread Predictor API is running. Use POST to make predictions.'
    })

@app.route('/api', methods=['POST'])
def predict():
    try:
        # Parse JSON request
        data = request.json
        
        # Extract parameters
        type_op = data.get('type_op')
        matrix_size = int(data.get('matrix_size'))
        var_type = data.get('var_type')
        matrix_type = data.get('matrix_type')
        
        # Optional parameters - set proper defaults
        is_iterative = data.get('is_iterative', False)
        
        # Use a numeric default for memory_pattern instead of "Unknown"
        memory_pattern = data.get('memory_pattern', 0)  # Use 0 or another appropriate numeric default
        
        # Make prediction
        optimal_threads, estimated_features = predictor.predict(
            type_op=type_op,
            matrix_size=matrix_size,
            var_type=var_type,
            matrix_type=matrix_type,
            is_iterative=is_iterative,
            memory_pattern=memory_pattern
        )
        
        # Prepare response
        response = {
            'optimal_threads': int(optimal_threads),
            'estimated_features': estimated_features
        }
        
        return jsonify(response)
        
    except Exception as e:
        # Print the error to logs
        print(f"Error in prediction: {str(e)}", file=sys.stderr)
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Ensure we're binding to 0.0.0.0 and using the PORT env variable
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting server on port {port}", file=sys.stderr)
    app.run(host='0.0.0.0', port=port, debug=False)