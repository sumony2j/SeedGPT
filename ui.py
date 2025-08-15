import gradio as gr
import subprocess
import os
import sys

# ✅ Function to run any command and return logs + optional file

def run_command(cmd, expected_file=None):
    try:
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=os.path.dirname(__file__))
        stdout, stderr = process.communicate()

        if process.returncode == 0:
            if expected_file and os.path.exists(expected_file):
                return f"✅ Completed successfully!\n\n{stdout}", expected_file
            return f"✅ Completed successfully!\n\n{stdout}"
        else:
            return f"❌ Error occurred!\n\n{stderr}", None
    except Exception as e:
        return f"⚠️ Exception: {str(e)}", None


# ✅ Generate Training Command
def generate_training_command(batch_size, iteration, dataset_file, context, emb_size, n_layers, lr, n_head, eval_itr):
    dataset_path = dataset_file.name if dataset_file else "./dataset.txt"
    script_path = os.path.join(os.path.dirname(__file__), "SeedGPT.py")

    cmd = (
        f"\"{sys.executable}\" \"{script_path}\" "
        f"--batch_size {batch_size} "
        f"--iteration {iteration} "
        f"--dataset \"{dataset_path}\" "
        f"--context {context} "
        f"--emb_size {emb_size} "
        f"--n_layers {n_layers} "
        f"--lr {lr} "
        f"--n_head {n_head} "
        f"--eval_itr {eval_itr}"
    )
    return cmd

# ✅ Generate Inference Command
def generate_inference_command(model_file, tokenizer_file, input_text, max_token):
    model_path = model_file.name if model_file else "SeedGPT.pt"
    tokenizer_path = tokenizer_file.name if tokenizer_file else "tokenizer.json"

    script_path = os.path.join(os.path.dirname(__file__), "Inference.py")  # Separate inference script
    output_file = os.path.join(os.path.dirname(__file__), "llm_output.txt")
    
    cmd = (
        f"\"{sys.executable}\" \"{script_path}\" "
        f"--model_path \"{model_path}\" "
        f"--tokenizer_path \"{tokenizer_path}\" "
        f"--input \"{input_text}\" "
        f"--max_token {max_token} " 
        f"--output_file \"{output_file}\""
    )
    return cmd

# ✅ Custom CSS for modern UI
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

* {
    font-family: 'Poppins', sans-serif;
}

body {
    background: #101820;
    color: #f5f5f5;
}

#title {
    text-align: center;
    font-size: 42px;
    font-weight: 600;
    background: linear-gradient(90deg, #6a11cb, #2575fc, #ff6a00);
    background-size: 200% auto;
    -webkit-background-clip: text;
    color: transparent;
    animation: gradientMove 4s linear infinite;
    margin-bottom: 10px;
}

@keyframes gradientMove {
    0% { background-position: 0% 50%; }
    100% { background-position: 200% 50%; }
}

.param-card {
    background: #1f2937;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    margin-bottom: 15px;
}

textarea, input, .gr-textbox {
    background-color: rgba(255,255,255,0.05) !important;
    color: white !important;
    border-radius: 12px !important;
    border: 1px solid #444 !important;
    padding: 10px !important;
}

.small-upload .wrap.svelte-1ipelgc {
    max-width: 50px !important;  /* adjust width */
    min-width: 50px !important;
    height: 15px !important;      /* match Input Prompt height */
    padding: 5px !important;
}

.small-upload .file-upload.svelte-1ipelgc {
    display: none !important; /* hide the big drop area */
}

.small-upload label {
    font-size: 12px !important;
    white-space: nowrap;
}

"""

# ✅ Build Gradio UI
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# 🌱 **SeedGPT-Basic Control Panel**", elem_id="title")
    
    gr.Markdown(
    """
    <div style="text-align:center; font-size:22px; font-weight:bold; color:#4CAF50;">
        🌟 Select Mode 🌟
    </div>
    <div style="text-align:center; font-size:18px; color:#555;">
        Train your model or run inference
    </div>
    <br>
    """
    )
    with gr.Accordion("📌 Instructions (Click to Expand)", open=False):
        gr.Markdown(
            """
            <ul style="margin-top:8px; line-height:1.6; font-size:15px; color:#444;">
                <li>✅ Use the model trained here during inference for best results.</li>
                <li>✅ Currently, a <b>demo model</b> is provided, trained on <code>shakespeare.txt</code>.</li>
                <li>✅ You can train the model on any language (English, Bengali, Hindi, etc.), but the dataset must be in <b>.txt format</b>.</li>
                <li>✅ This is a <b>decoder-only</b>, <b>character-based</b> LLM model.</li>
                <li>✅ It will generate <b>meaningful text</b> based on the provided <b>maximum number of characters (max tokens)</b>.</li>
                <li>⚠️ Currently running on the <b>free CPU tier of Hugging Face Spaces</b>, so <b>training/inference will be very slow</b>.</li>
                <li>💡 For better performance, <b><a href="https://github.com/sumony2j/SeedGPT.git" target="_blank" style="color:#4CAF50; text-decoration:none;">follow the GitHub instructions</a></b> to deploy locally.</li>
                <li>🔥 Supports <b>multi-GPU training</b> for faster execution when running on your hardware.</li>
            </ul>
            """
        )

    with gr.Tabs():
        # ✅ Training Tab
        with gr.Tab("🧑‍🏫 Training"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📂 Upload Dataset")
                    dataset_file = gr.File(file_types=[".txt"], label="Upload Dataset (.txt)", elem_classes="small-upload")
                
                with gr.Column(scale=2):
                    gr.Markdown("### ⚙️ Training Parameters")
                    
                    with gr.Accordion("📌 Core Settings", open=False):
                        batch_size = gr.Slider(1, 2048, value=32, step=1, label="Batch Size")
                        iteration = gr.Slider(1, 5000000, value=10000, step=100, label="Iterations")

                    with gr.Accordion("🧠 Model Configuration", open=False):
                        context = gr.Slider(1, 4096, value=128, step=1, label="Context Length")
                        emb_size = gr.Slider(1, 4096, value=64, step=1, label="Embedding Size")
                        n_layers = gr.Slider(1, 64, value=4, step=1, label="Number of Layers")
                        n_head = gr.Slider(1, 64, value=4, step=1, label="Number of Heads")
                    
                    with gr.Accordion("⚡ Optimization Settings", open=False):
                        lr = gr.Slider(1e-6, 1e-2, value=5e-4, step=1e-6, label="Learning Rate")
                        eval_itr = gr.Slider(1, 1000, value=1, step=10, label="Eval Iterations")

            train_cmd_preview = gr.Textbox(label="Generated Command", lines=3, visible=False)
            
            with gr.Row():
                start_train_btn = gr.Button("🚀 Start Training", variant="primary")
            
            # ✅ Allow download of BOTH files
            with gr.Row():
                with gr.Column(scale=3):
                    train_output = gr.Textbox(label="Log", lines=9, elem_classes="small-upload", autoscroll=False)
                with gr.Column(scale=1):
                    model_download = gr.File(label="⬇ Download Trained Model", file_types=[".pt"], elem_classes="small-upload")
                with gr.Column(scale=1):
                    tokenizer_download = gr.File(label="⬇ Download Tokenizer", file_types=[".json"], elem_classes="small-upload")


            def generate_and_run_training(batch_size, iteration, dataset_file, context, emb_size, n_layers, lr, n_head, eval_itr):
                cmd = generate_training_command(batch_size, iteration, dataset_file, context, emb_size, n_layers, lr, n_head, eval_itr)
                logs = run_command(cmd)
                model_file = os.path.join(os.path.dirname(__file__), "SeedGPT.pt")
                tokenizer_file = os.path.join(os.path.dirname(__file__), "tokenizer.json")
                return cmd, logs, model_file if os.path.exists(model_file) else None, tokenizer_file if os.path.exists(tokenizer_file) else None

            start_train_btn.click(
                generate_and_run_training,
                inputs=[batch_size, iteration, dataset_file, context, emb_size, n_layers, lr, n_head, eval_itr],
                outputs=[train_cmd_preview, train_output, model_download, tokenizer_download]
                )

        # ✅ Inference Tab
        with gr.Tab("🤖 Inference"):
            
            with gr.Row():
                with gr.Column(scale=1):
                    model_file = gr.File(file_types=[".pt"], label="Upload Model (SeedGPT.pt)", value="./SeedGPT_Demo.pt", elem_classes="small-upload")
                with gr.Column(scale=1):
                    tokenizer_file = gr.File(file_types=[".json"], label="Upload Tokenizer (tokenizer.json)", value="./tokenizer_demo.json", elem_classes="small-upload")
                with gr.Column(scale=2):
                    with gr.Row():
                        input_text = gr.Textbox(label="Input Prompt", placeholder="Enter your text...", lines=1, elem_classes="small-upload")
                    with gr.Row():
                        max_token = gr.Slider(1, 200000, value=10000, step=1, label="Max Tokens", elem_classes="small-upload")

            infer_cmd_preview = gr.Textbox(label="Generated Command", lines=3, visible=False)
            
            with gr.Row():
                start_infer_btn = gr.Button("🚀 Run Inference", variant="primary")
            
            with gr.Row():
                with gr.Column(scale=3):
                    infer_output = gr.Textbox(label="Generated Text", lines=9, autoscroll=False)
                with gr.Column(scale=1):
                    infer_download = gr.File(label="⬇ Download Generated Text", file_types=[".txt"], elem_classes="small-upload")
            

            def generate_and_run_inference(model_file, tokenizer_file, input_text, max_token):
                cmd = generate_inference_command(model_file, tokenizer_file, input_text, max_token)
                run_command(cmd)
                output_file = os.path.join(os.path.dirname(__file__), "llm_output.txt")
                with open(output_file, "r", encoding="utf-8") as f:
                    logs = f.read()
                return cmd, logs, output_file if os.path.exists(output_file) else None

            start_infer_btn.click(
                generate_and_run_inference,
                inputs=[model_file, tokenizer_file, input_text, max_token],
                outputs=[infer_cmd_preview, infer_output, infer_download]
                )
demo.launch()