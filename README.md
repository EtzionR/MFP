#  MFP: Multi View Feature Propagation

Multi View Graph Feature Propagation for Feature Sparsity and Privacy Preservation.

You can learn more on MFP with the following links:

| Title | Link |
| :---------- | :--- |
| Arxiv pre-print MFP paper :books: | [https://arxiv.org/abs/2510.11347](https://arxiv.org/abs/2510.11347) |
| Google colab notebook :rocket: | [MFP_Tutorial](https://colab.research.google.com/drive/1taG0704lVq50dBUq1NThndP2elL9J5In#scrollTo=6xE3qXNvmUcr) |
| NotebookLM Podcast on MFP :notes: | [Media/MFP_Podcast_English.m4a](https://github.com/EtzionR/MFP/blob/main/Media/MFP_Podcast_English.m4a) |
| NotebookLM Video on MFP :video_camera: | [MFP_Video_English.mp4](https://www.youtube.com/watch?v=elvaDyqWC-c)|

To install proper conda env to run MFP, you can use this following commands:
[build_conda_env.txt](https://github.com/EtzionR/MFP/blob/main/build_conda_env.txt)

## MFP Framework:

Multi-View Feature Propagation (MFP) is a graph-learning framework designed to improve representation quality when node features are missing, sparse, or privacy-sensitive. MFP extends classical Feature Propagation (FP) by introducing multiple noisy partial views of the feature space, each propagated independently across the graph. The final representation is obtained by aggregating (e.g., concatenating) all propagated views.

![picture](https://github.com/EtzionR/MFP/raw/main/figures/pipeline.png)

## BibTeX
If you find our code useful, please consider citing our work using the following BibTeX entry:

``` sh
@misc{harari2025multiviewgraphfeaturepropagation,
      title={Multi-View Graph Feature Propagation for Privacy Preservation and Feature Sparsity}, 
      author={Etzion Harari and Moshe Unger},
      year={2025},
      eprint={2510.11347},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.11347}, 
}
```

## License
MIT © [Etzion Harari](https://github.com/EtzionR) and [Moshe Unger](https://en-coller.tau.ac.il/profile/mosheunger_62) | TAU
