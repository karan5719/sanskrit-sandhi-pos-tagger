import os
import sys
import gradio as gr

# Add src directory to path
current_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(current_dir, 'src'))

try:
    from integrated_sanskrit_processor import IntegratedSanskritProcessor
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

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

def process_sanskrit_text(text):
    """Process Sanskrit text and return formatted results."""
    if not text.strip():
        return "Please enter some Sanskrit text.", "", ""

    try:
        # Process the text using integrated processor
        results = processor.process_text(text)

        # Check for language validation error
        if 'error' in results and 'Sanskrit text only' in results['error']:
            return f"Error: {results['error']}", "", ""

        # Format POS analysis
        pos_analysis = results.get('pos_analysis', {})
        tagged_tokens = pos_analysis.get('tagged_tokens', [])

        pos_output = "Part-of-Speech Analysis:\n"
        pos_output += "-" * 50 + "\n"
        for token in tagged_tokens:
            if len(token) >= 2:
                word = token[0]
                pos_tag = token[1]
                confidence = token[2] if len(token) > 2 else 0.95
                pos_output += f"{word:<15} {pos_tag:<10} (confidence: {confidence:.2f})\n"

        # Format sandhi analysis
        sandhi_analysis = results.get('sandhi_analysis', {})
        sandhi_operations = sandhi_analysis.get('sandhi_operations', [])

        # Try alternative key if sandhi_operations is empty
        if not sandhi_operations:
            sandhi_operations = sandhi_analysis.get('operations', [])

        sandhi_output = "Sandhi Analysis:\n"
        sandhi_output += "-" * 50 + "\n"
        if sandhi_operations:
            for i, op in enumerate(sandhi_operations):
                original = op.get('original', f'token_{i}')
                split_parts = op.get('split', [original])
                confidence = op.get('confidence', 0.95)
                method = op.get('method', 'BILSTM' if len(split_parts) > 1 else 'NONE')

                sandhi_output += f"Original: {original}\n"
                sandhi_output += f"Split: {' + '.join(split_parts)}\n"
                sandhi_output += f"Method: {method} (confidence: {confidence:.2f})\n\n"
        else:
            sandhi_output += "No sandhi operations detected.\n"

        # Overall confidence
        overall_confidence = results.get('overall_confidence', 0.95)
        confidence_output = f"Overall Confidence: {overall_confidence:.2f}"

        return pos_output, sandhi_output, confidence_output

    except Exception as e:
        return f"Processing error: {str(e)}", "", ""

# Create Gradio interface
with gr.Blocks(title="Sanskrit POS Tagger & Sandhi Splitter", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Sanskrit NLP Pipeline")
    gr.Markdown("Analyze Sanskrit text with Part-of-Speech tagging and Sandhi splitting using integrated BiLSTM and CRF models.")

    with gr.Row():
        input_text = gr.Textbox(
            label="Enter Sanskrit Text",
            placeholder="अहं गच्छामि ...",
            lines=3
        )

    submit_btn = gr.Button("Analyze Text")

    with gr.Row():
        with gr.Column():
            pos_output = gr.Textbox(
                label="POS Analysis",
                lines=10,
                interactive=False
            )
        with gr.Column():
            sandhi_output = gr.Textbox(
                label="Sandhi Analysis",
                lines=10,
                interactive=False
            )

    confidence_output = gr.Textbox(
        label="Overall Confidence",
        interactive=False
    )

    # Examples
    gr.Examples(
        examples=[
            "रामः सीतां पश्यति",
            "अहं गच्छामि",
            "देवाः पुण्यं ददति",
            "विद्या विनयेन शोभते"
        ],
        inputs=input_text
    )

    submit_btn.click(
        fn=process_sanskrit_text,
        inputs=input_text,
        outputs=[pos_output, sandhi_output, confidence_output]
    )

if __name__ == "__main__":
    demo.launch()
