import os 
import sys
from flask import Flask, render_template, request, jsonify

# Add src directory to path
current_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(current_dir, 'src'))

try:
    from integrated_sanskrit_processor import IntegratedSanskritProcessor
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

app = Flask(__name__)

# Initialize the integrated processor
print("🔧 Loading Integrated Sanskrit Processor...")
try:
    processor = IntegratedSanskritProcessor(
        pos_model_path=os.path.join(current_dir, 'models', 'enhanced_crf_pos_model_v3.pkl'),
        bilstm_threshold=0.7,
        use_bilstm=True
    )
    print("✅ Processor loaded successfully")
except Exception as e:
    print(f"⚠️  Model loading failed: {e}")
    print("🔄 Loading without models...")
    processor = IntegratedSanskritProcessor(
        pos_model_path=None,
        bilstm_threshold=0.7,
        use_bilstm=False
    )
    print("✅ Processor loaded in basic mode")

@app.route('/')
def home():
    """Render the main analysis interface."""
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_text():
    """Process Sanskrit text and return results."""
    try:
        user_input = request.form.get('sanskrit_text', '').strip()
        
        if not user_input:
            return jsonify({
                'error': 'Please enter some Sanskrit text'
            })
        
        # Process the text using integrated processor
        results = processor.process_text(user_input)
        
        # Check for language validation error
        if 'error' in results and 'Sanskrit text only' in results['error']:
            return jsonify({
                'error': results['error'],
                'language_check': results.get('language_check', {}),
                'success': False
            })
        
        # Format results for the frontend
        pos_analysis = results.get('pos_analysis', {})
        tagged_tokens = pos_analysis.get('tagged_tokens', [])
        
        # Get sandhi operations
        sandhi_analysis = results.get('sandhi_analysis', {})
        sandhi_operations = sandhi_analysis.get('sandhi_operations', [])
        
        # Try alternative key if sandhi_operations is empty
        if not sandhi_operations:
            sandhi_operations = sandhi_analysis.get('operations', [])
        
        # Get confidence scores
        overall_confidence = results.get('overall_confidence', 0.95)
        
        # Enhance sandhi operations with method information
        enhanced_sandhi_ops = []
        for i, op in enumerate(sandhi_operations):
            enhanced_op = {
                'original': op.get('original', f'token_{i}'),
                'split': op.get('split', [op.get('original', f'token_{i}')]),
                'confidence': op.get('confidence', 0.95),
                'method': op.get('method', 'BILSTM' if op.get('split') and len(op.get('split', [])) > 1 else 'NONE')
            }
            enhanced_sandhi_ops.append(enhanced_op)
        
        # Ensure tokens have confidence scores
        enhanced_tokens = []
        for token in tagged_tokens:
            if len(token) == 2:
                enhanced_tokens.append([token[0], token[1], 0.95])  # Add default confidence
            else:
                enhanced_tokens.append(token)
        
        formatted_results = {
            'tokens': enhanced_tokens,
            'sandhi_operations': enhanced_sandhi_ops,
            'confidence': overall_confidence
        }
        
        return jsonify({
            'success': True,
            'results': formatted_results
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Processing error: {str(e)}'
        })

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'components': {
            'tokenizer': 'loaded',
            'sandhi_splitter': 'loaded',
            'crf_model': 'loaded' if processor.crf_model else 'not available',
            'bilstm_model': 'loaded' if processor.sandhi_splitter.bilstm_model else 'not available'
        }
    })

if __name__ == '__main__':
    print("🌐 Starting Simple Sanskrit NLP App...")
    print("🎯 Available at: http://localhost:8085")
    print("📊 API endpoint: http://localhost:8085/process")
    print("🏥 Health check: http://localhost:8085/health")
    
    app.run(debug=True, host='0.0.0.0', port=8085)
