# ACSESS: Automatic Combination of Sample Selection Strategies

This repistory contains the code for the evaluation of sample selection strategies for the few-shot learning and the implementation of the ACSESS strategy for the paper with title `Comparison and Automatic Combination of Sample Selection Strategies for Few-Shot Learning: Impact of Sample Quality and Quantity`.

The experiments utilise 4 different code repositories, which are modified in order to work with few-shot learning and sample selection. This includes following repositories:
- Active Learning and Core-Set selection strategies using the [AL_vs_SubsetSelection](https://github.com/dongmean/AL_vs_SubsetSelection/) and [DeepCore](https://github.com/PatrickZH/DeepCore) (both contained in the `AL_vs_SubsetSelection` folder)
- Dataset Cartography and [cartography](https://github.com/allenai/cartography) (contained in the `cartography` folder) -- here we train the model separately and use the repository only for evaluation based on training dynamics (training script contained in the `DataCartography` folder)
- Running and Evaluation of the few-shot learning using [meta-album](https://github.com/ihsaan-ullah/meta-album/) (contained in the `meta-album` folder). This folder also contains the implementation of the Forward, Backward and Datamodels methods for identifying the relevant strategies.
- LENS strategy for finding supporting samples for the in-context learning using [ICL_Support_Example](https://github.com/LeeSureman/ICL_Support_Example/) (contained in `ICL_Support_Example` folder)

The processing of results from different sample selection strategies, the implementation of `Similarity`, `Diversity` and `Random` selection strategies, and the processing of evaluation results are provided in the `dataset_creation` folder. This folder also contains the implementation for the weighted combination and the Datamodels method for identifying the relevant strategies (using the results from the meta-album evaluation).
