# CA3
PyTorch implementation of the article Attention-Emotion Assessment of ASD Children via Representation Learning based on Cross-Modal Disentanglement and Attention Alignment

# Introduction
This project aims to establish an open-source GitHub repository named CA3 (Cognitive-Affective Ability Assessment), focusing on providing a comprehensive solution for assessing cognitive and emotional abilities in children with Autism Spectrum Disorder (ASD). Our research introduces a novel assessment framework that combines attention control and emotion perception abilities, overcoming the limitations of existing methods in processing multimodal data such as EEG, eye-tracking, and facial expressions.
The core innovation of the CA3 framework lies in aligning and integrating multiple neural recordings and behavioral cues to generate personalized cognitive-affective profiles. This framework not only reveals the heterogeneity in attention and emotion processing among children with ASD but also provides essential insights for clinical decision-making.
In this GitHub repository, users will have access to:
- The source code of the CA3 framework, offering a complete model implementation and training procedures.
- The multimodal datasets used, including EEG, eye-tracking, and facial expression recordings, enabling users to test and validate the model in practical applications.
- Detailed guidance on the preprocessing and data alignment methods used in the experiments, allowing users to replicate our research findings.
We hope that this open-source project will support personalized assessments and intervention strategies for children with ASD, promoting the advancement of research and applications in the relevant fields.

# Framework
 <picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/51cloud/CA3/framework.pdf">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/51cloud/CA3/framework.pdf">
  <img alt="Shows an illustrated sun in light mode and a moon with stars in dark mode." src="https://github.com/51cloud/CA3/framework.pdf">
</picture>

## Requirements
We provide a large set of baseline results and trained models available for download in the [Model Zoo](MODEL_ZOO.md).

Please find installation instructions for CA3 in [INSTALL.md](INSTALL.md). 

## How to download the datasets
The full datasets can be downloaded via: 
['Google Drive']()

## Quick Start

Follow the example in [GETTING_STARTED.md](GETTING_STARTED.md) to start playing video models with CA3.

## Visualization Tools

We offer a range of visualization tools for the train/eval/test processes, model analysis, and for running inference with trained model.
More information at [Visualization Tools](VISUALIZATION_TOOLS.md).

## Citing PySlowFast
If you find CA3 useful in your research, please use the following BibTeX entry for citation.
```BibTeX
