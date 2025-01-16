# Refusal-LLMs

Refusal behavior in large language models (LLMs) refers to the model’s ability to decline responding to harmful, unethical, or inappropriate prompts, and it is a crucial aspect of LLM alignment. This behavior ensures that models adhere to safety guidelines, ethical standards, and societal norms. Previous studies on refusal ablation and jailbreak attacks have demonstrated how refusal can be identified in the latent space and disabled, resulting in uncensored models. This paper investigates refusal behavior by comparing its representation across six models from three architectural families. 

[![How Llama mediates refusal](https://raw.githubusercontent.com/FabianHildebrandt/Refusal-LLMs/main/Llama-3.2-3B-Instruct/Scatter_PCA.png)](https://raw.githubusercontent.com/FabianHildebrandt/Refusal-LLMs/main/Llama-3.2-3B-Instruct/Scatter_Animation_PCA.mp4)

# How to get started

1. [Link to the paper]()
2. [All results]()
3. Try it yourself and do your own experiments...

# Reproduce the results 

1. Clone the repository.
```bash
git clone https://github.com/FabianHildebrandt/Refusal-LLMs.git
```

2. Navigate to the project directory
```bash
cd Refusal-LLMs/
```

3. Install the dependencies
```bash
pip install requirements.txt
```

4. Insert your Hugging Face token and the model you want to analyze in the [configuration](./config.yaml) in the section *extraction*.

5. Run the [extraction notebook](extract_refusal.ipynb) to extract the refusal activations. 

6. Repeat this for the models you want to analyze.

7. Add the analyzed models to the section *analysis->model_names* in the [configuration](./config.yaml)

8. Run the [analysis notebook](analyze_refusal.ipynb).



