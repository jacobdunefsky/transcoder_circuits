# Transcoder-circuits: reverse-engineering LLM circuits with transcoders

This repository contains tools for understanding what's going on inside large language models by using a tool called "transcoders". Transcoders decompose MLP sublayers in transformer models into a sparse linear combination of interpretable features. By using transcoders, we can reverse-engineer fine-grained circuits of features within the model.

To get started, we recommend working through the `walkthrough.ipynb` notebook. The full structure of the repository is as follows:

* Installation tools:
  * `setup.sh`: A shell script for installing dependencies and downloading transcoder weights.
  * `requirements.txt`: The standard Python dependencies list.
* Examples:
  * `walkthrough.ipynb`: A walkthrough notebook that demonstrates how to use the tools provided in this repository for reverse-engineering LLM circuits with transcoders.
  * `train_transcoder.py`: An example script for training a transcoder.
* Experimental results;
  * `sweep.ipynb`: An evaluation of transcoders and SAEs trained on Pythia-410M (weights not provided).
  * `interp-comparison.ipynb`: Code for performing a blind comparison of SAE and transcoder feature interpretability
* Case studies:
  * `case_study_citations.ipynb`: An example of a reverse-engineering case study that we carried out, in which we investigated a transcoder feature that activates on semicolons in parenthetical citations.
  * `case_study_caught.ipynb`: An example of a reverse-engineering case study that we carried out, in which we investigated a transcoder feature that activates on the verb "caught".
  * `case_study_local_context.ipynb`: An example of a reverse-engineering case study that we carried out, in which we attempted to reverse-engineer a circuit that computes a harder-to-interpret transcoder feature. (We were less successful in this case study, but are including it in the interest of transparency.)
  * `restricted blind case studies.ipynb`: A notebook containing a set of "restricted blind case studies" that reverse-engineer random GPT2-small transcoder features *without looking at MLP0 transcoder feature activations*.
* Libraries:
  * `sae_training/`: Code for training and using transcoders. The code is largely based on an older version of [Joseph Bloom's excellent SAE repository](https://github.com/jbloomAus/SAELens) -- **shoutouts to him!**. (The misnomer `sae_training` is a vestige of this origin of the code.)
  * `transcoder_circuits/`: Code for reverse-engineering and analyzing circuits with transcoders. These are the tools that we use in the walkthrough notebook and in the case studies.
