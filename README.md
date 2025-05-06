# 🌱 SeedGPT

**SeedGPT** is a lightweight decoder-only Large Language Model (LLM) built from scratch using **PyTorch**. Inspired by architectures like GPT-2, It generates texts starting from a seed word, making it ideal for educational purposes and quick experimentation with LLM concepts.

---

## 🚀 Features

- 🧠 Minimal transformer-based LLM architecture in PyTorch  
- ✍️ Word-by-word text generation from a given seed  
- 🛠️ Simple and clean training & inference scripts  
- 📦 Modular codebase for learning and experimentation  
- 📜 Shell script support for quick execution (`train.sh` & `inference.sh`)

---

## 📦 Installation

1. Clone the repository:

```bash
git clone https://github.com/sumony2j/SeedGPT.git

cd SeedGPT
```
2. Install dependencies:

```bash
pip install torch numpy pandas
```

---

## 🚀 Usage

### 🏋️ Training

**Train the model using a plain text file:**

```bash
python3 SeedGPT.py --batch_size 64 --iteration 10000 --dataset "./Data.txt" --context 256 --emb_size 384 --n_layers 6 --lr 3e-4 --n_head 6 --eval_itr 100
```

**Alternatively, run the pre-defined shell script for training:**

```bash
bash train.sh
```

**Training Script Argument Descriptions ---**

| Argument     | Description                                                                             |
|--------------|-----------------------------------------------------------------------------------------|
| --batch_size | Number of training samples per batch. Controls memory usage and training speed.         |
| --iteration  | Total number of training steps/iterations. More iterations usually improve performance. |
| --dataset    | Path to the training text file. Only .txt format is currently supported.                |
| --context    | Context window size (i.e., number of tokens the model looks at in one input).           |
| --emb_size   | Dimensionality of token embeddings. Higher = more expressive power.                     |
| --n_layers   | Number of transformer layers (blocks) in the model.                                     |
| --lr         | Learning rate for the optimizer. Typical range: 1e-4 to 3e-4.                           |
| --n_head     | Number of attention heads in the multi-head attention mechanism.                        |
| --eval_itr   | How often (in training iterations) to print evaluation/loss metrics.                    |


---


### 🧠 Inference

**Generate a sequence of words from a seed word:**

```bash
python3 Inference.py --model_path "./SeedGPT.pt" --tokenizer_path "./tokenizer.json" --input "Hello"    --max_token 10000 --output_file "./llm_output.txt" --show false
```
**Or use the shell script:**

```bash
bash inference.sh
```

**Inference Script Argument Descriptions ---**

| Argument         | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| --model_path     | Path to the trained model file.                                             |
| --tokenizer_path | Path to the tokenizer file.                                                 |
| --input          | Seed text to begin generation with.                                         |
| --max_token      | Maximum number of tokens to generate.                                       |
| --output_file    | File path where the generated output will be saved.                         |
| --show           | Prints the generated text to the console. Else, only saves to output_file.  |

---

## 🌟 Inspiration

SeedGPT draws inspiration from the legendary [Andrej Karpathy](https://www.youtube.com/@karpathy) and his acclaimed tutorial series:

🎥 **Let’s Build GPT from Scratch**  
A step-by-step guide that demystifies the inner workings of transformer-based language models.

📺 **Watch the full series here**: [https://www.youtube.com/watch?v=kCc8FmEb1nY](https://www.youtube.com/watch?v=kCc8FmEb1nY)

> 💡 This project is built in the spirit of learning, simplicity, and hands-on experimentation—just like Karpathy intended!

---

## 🔮 Future Improvements

While SeedGPT is intentionally minimal to aid understanding, several enhancements are planned to expand its capabilities:

- 🧩 **Multi-node Training Support**  
  Integrate PyTorch Distributed or DeepSpeed for large-scale, multi-node training.
  *(Currently, multi-GPU on a single node is supported)*

- 📂 **Flexible Dataset Support**  
  Add support for diverse formats such as JSON, CSV, and code corpora.  
  *(Currently, only `.txt` files are supported.)*

- 🧠 **Positional Embeddings**  
  Incorporate proper positional encodings to enhance sequence awareness and context handling.

> 🚧 *These features are under active consideration and community contributions are welcome!*

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 🤝 Contributions

We welcome **contributions**, **issues**, and **feature requests**!  
Feel free to fork the repository, make improvements, and submit a pull request to help make SeedGPT even better.

Your contributions help make this project grow and evolve—thank you! 🙏

> 💡 **Note:** *SeedGPT is designed as an educational tool for learning the basics of generative modeling and large language models using PyTorch.*
