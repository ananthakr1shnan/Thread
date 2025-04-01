import joblib
import pandas as pd
import os
pd.set_option('future.no_silent_downcasting', True)

class ThreadPredictor:
    def __init__(self, model_path="final_pipeline.pkl"):
        """Initialize the predictor by loading the trained pipeline and setting mean values per operation type."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.pipeline = joblib.load(model_path)  
        
        self.mean_values_by_op = {
            'Addition': {'numOp': 291944.5, 'numVar': 3.0, 'exeTime': 0.000948, 'complexity': 2.000},
            'Determinant': {'numOp': 130279000, 'numVar': 1.0, 'exeTime': 0.059061, 'complexity': 3.000},
            'Eigenvalue': {'numOp': 20844800, 'numVar': 2.0, 'exeTime': 0.003850, 'complexity': 2.000},
            'Exponential': {'numOp': 211493.1, 'numVar': 2.0, 'exeTime': 0.001734, 'complexity': 2.000},
            'Logarithm': {'numOp': 224100.6, 'numVar': 2.0, 'exeTime': 0.002242, 'complexity': 2.000},
            'LUDecomposition': {'numOp': 42477540, 'numVar': 3.0, 'exeTime': 0.067861, 'complexity': 2.997},
            'Multiplication': {'numOp': 181227500, 'numVar': 3.0, 'exeTime': 0.235311, 'complexity': 3.000},
            'Scaling': {'numOp': 271952.9, 'numVar': 2.0, 'exeTime': 0.000749, 'complexity': 2.000},
            'SquareRoot': {'numOp': 265174.2, 'numVar': 2.0, 'exeTime': 0.001466, 'complexity': 2.000},
            'Transposition': {'numOp': 296295.0, 'numVar': 2.0, 'exeTime': 0.001209, 'complexity': 2.000}
        }
        
        self.adjustment_factors = {
            'Addition': {'numOp': 0.1, 'numVar': 0.01, 'exeTime': 0.2, 'complexity': 0.0},
            'Determinant': {'numOp': 0.3, 'numVar': 0.0, 'exeTime': 0.4, 'complexity': 0.0},
            'Eigenvalue': {'numOp': 0.2, 'numVar': 0.01, 'exeTime': 0.3, 'complexity': 0.0},
            'Exponential': {'numOp': 0.1, 'numVar': 0.0, 'exeTime': 0.2, 'complexity': 0.0},
            'Logarithm': {'numOp': 0.1, 'numVar': 0.0, 'exeTime': 0.2, 'complexity': 0.0},
            'LUDecomposition': {'numOp': 0.3, 'numVar': 0.01, 'exeTime': 0.4, 'complexity': 0.0},
            'Multiplication': {'numOp': 0.3, 'numVar': 0.01, 'exeTime': 0.4, 'complexity': 0.0},
            'Scaling': {'numOp': 0.1, 'numVar': 0.0, 'exeTime': 0.1, 'complexity': 0.0},
            'SquareRoot': {'numOp': 0.1, 'numVar': 0.0, 'exeTime': 0.2, 'complexity': 0.0},
            'Transposition': {'numOp': 0.1, 'numVar': 0.0, 'exeTime': 0.1, 'complexity': 0.0}
        }
    
    def adjust_metrics(self, type_op, matrix_size, var_type, matrix_type, is_iterative, memory_pattern):
        """Adjust metrics based on input parameters while staying close to mean values."""
    
        base_values = self.mean_values_by_op.get(type_op, {
            'numOp': 10000, 'numVar': 2.0, 'exeTime': 0.001, 'complexity': 2.0
        })
        
        factors = self.adjustment_factors.get(type_op, {
            'numOp': 0.1, 'numVar': 0.0, 'exeTime': 0.1, 'complexity': 0.0
        })
        
        size_ratio = matrix_size / 1000.0
        
        num_op = base_values['numOp'] * (1 + (factors['numOp'] * (size_ratio - 1)))
        num_var = base_values['numVar'] * (1 + (factors['numVar'] * (size_ratio - 1)))
        exe_time = base_values['exeTime'] * (1 + (factors['exeTime'] * (size_ratio - 1)))
        complexity = base_values['complexity']  # Complexity generally stays the same
        
        # Handle matrix types as integers (0=RANDOM_DENSE, 1=SPARSE_50, 2=DIAGONAL_DOMINANT)
        if isinstance(matrix_type, int):
            if matrix_type == 1:  # SPARSE_50
                num_op *= 0.9
                exe_time *= 0.85
            elif matrix_type == 2:  # DIAGONAL_DOMINANT
                num_op *= 0.95
                exe_time *= 0.9
        # Also handle string matrix types for backward compatibility
        elif isinstance(matrix_type, str):
            if matrix_type.lower() == 'sparse':
                num_op *= 0.9
                exe_time *= 0.85
            elif matrix_type.lower() == 'banded':
                num_op *= 0.95
                exe_time *= 0.9
        
        # Handle variable types as strings from C code
        if isinstance(var_type, str):
            var_type_lower = var_type.lower()
            if var_type_lower == 'double' or var_type_lower == 'mixed':
                exe_time *= 1.1
            elif var_type_lower == 'int' or var_type_lower == 'integer':
                exe_time *= 0.9
        
        if is_iterative:
            exe_time *= 1.2
            num_op *= 1.1
        
        if memory_pattern > 0:
            exe_time *= (1 + (0.05 * memory_pattern))
        
        return num_op, num_var, exe_time, complexity
    
    def predict(self, type_op, matrix_size, var_type, matrix_type, is_iterative, memory_pattern, num_op=None, num_var=None, exe_time=None, complexity=None):
        """Make a prediction using the loaded model, filling missing values with adjusted operation-specific values."""
        try:
            if isinstance(memory_pattern, str):
                memory_pattern = 0 
            
            adjusted_num_op, adjusted_num_var, adjusted_exe_time, adjusted_complexity = self.adjust_metrics(
                type_op, matrix_size, var_type, matrix_type, is_iterative, memory_pattern
            )
            
            num_op = num_op if num_op is not None else adjusted_num_op
            num_var = num_var if num_var is not None else adjusted_num_var
            exe_time = exe_time if exe_time is not None else adjusted_exe_time
            complexity = complexity if complexity is not None else adjusted_complexity
            
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
            input_data['size_complexity_interaction'] = input_data['matrixSize'] * input_data['complexity']
            input_data['numop_numvar_interaction'] = input_data['numOp'] * input_data['numVar']
            
            input_data = input_data.fillna(0).infer_objects(copy=False)
            
            optimal_threads = self.pipeline.predict(input_data)  
            adjusted_optimal_threads = int(optimal_threads[0]) + 1
            
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