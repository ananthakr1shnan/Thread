import joblib
import pandas as pd
import os
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

class ThreadPredictor:
    def __init__(self, model_path="final_pipeline.pkl"):
        """Initialize the predictor by loading the trained pipeline and setting mean values per operation type."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.pipeline = joblib.load(model_path)  # Load the full pipeline
        
        # Define operation-specific mean values
        self.mean_values_by_op = {
            'Addition': {'numOp': 291944.5, 'numVar': 3.0, 'exeTime': 0.000948, 'complexity': 2.000},
            'Determinant': {'numOp': 130279000, 'numVar': 1.0, 'exeTime': 0.059061, 'complexity': 3.000},
            'Eigenvalue': {'numOp': 20844800, 'numVar': 2.0, 'exeTime': 0.003850, 'complexity': 2.000},
            'Exponential': {'numOp': 211493.1, 'numVar': 2.0, 'exeTime': 0.001734, 'complexity': 2.000},
            'Logarithm': {'numOp': 224100.6, 'numVar': 2.0, 'exeTime': 0.002242, 'complexity': 2.000},
            'LU Decomposition': {'numOp': 42477540, 'numVar': 3.0, 'exeTime': 0.067861, 'complexity': 2.997},
            'Multiplication': {'numOp': 181227500, 'numVar': 3.0, 'exeTime': 0.235311, 'complexity': 3.000},
            'Scaling': {'numOp': 271952.9, 'numVar': 2.0, 'exeTime': 0.000749, 'complexity': 2.000},
            'Square Root': {'numOp': 265174.2, 'numVar': 2.0, 'exeTime': 0.001466, 'complexity': 2.000},
            'Transposition': {'numOp': 296295.0, 'numVar': 2.0, 'exeTime': 0.001209, 'complexity': 2.000}
        }
    
    def predict(self, type_op, matrix_size, var_type, matrix_type, is_iterative, memory_pattern, num_op=None, num_var=None, exe_time=None, complexity=None):
        """Make a prediction using the loaded model, filling missing values with operation-specific averages."""
        try:
            # Ensure memory_pattern is numeric
            if isinstance(memory_pattern, str):
                memory_pattern = 0  # Default numeric value
            
            # Get operation-specific mean values
            op_defaults = self.mean_values_by_op.get(type_op, {'numOp': 0, 'numVar': 0, 'exeTime': 0.0, 'complexity': 1})
                    
            # Use mean values if parameters are missing
            num_op = num_op if num_op is not None else op_defaults['numOp']
            num_var = num_var if num_var is not None else op_defaults['numVar']
            exe_time = exe_time if exe_time is not None else op_defaults['exeTime']
            complexity = complexity if complexity is not None else op_defaults['complexity']
            
            # Prepare input data as a DataFrame to match training format
            input_data = pd.DataFrame([{
                'typeOp': type_op,
                'matrixSize': matrix_size,
                'varType': var_type,
                'matrixType': matrix_type,
                'isIterative': is_iterative,
                'memoryPattern': memory_pattern,
                'numOp': num_op,
                'numVar': num_var,
                'exeTime': exe_time,
                'complexity': complexity
            }])
            
            # Compute missing interaction features
            input_data['size_complexity_interaction'] = input_data['matrixSize'] * input_data['complexity']
            input_data['numop_numvar_interaction'] = input_data['numOp'] * input_data['numVar']
            
            # Ensure missing columns are handled
            input_data = input_data.fillna(0).infer_objects(copy=False)
            
            # Make prediction
            optimal_threads = self.pipeline.predict(input_data)  # Predict optimal thread count
            adjusted_optimal_threads = int(optimal_threads[0]) + 1
            # Return the estimated features along with the prediction
            estimated_features = {
                'exeTime': float(exe_time),
                'numOp': float(num_op),
                'numVar': float(num_var),
                'complexity': float(complexity),
                'isIterative': bool(is_iterative),
                'memoryPattern': int(memory_pattern)
            }
            
            return adjusted_optimal_threads, estimated_features
            
        except Exception as e:
            print(f"Prediction error: {e}")
            raise