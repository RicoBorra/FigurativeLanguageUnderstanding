# Figurative Language Understanding

This repository contains all the work made on Deep Natural Language Processing project on Figurative Language Understanding. The project was made by the Decepticons team.

All the models trained for this project have been posted on HuggingFace and are retrieved from there on the execution of the notebooks.

## FLUTE

As the FLUTE dataset creation involved a high amount of human checking and modifications to generate figurative sentences, we couldn't afford to reproduce the creation process.

However, the FLUTE paper experiments have been reproduced in the notebook [flute.ipynb](flute.ipynb). These experiments consisted in fine-tuning a T5 model on the e-SNLI dataset and on the FLUTE dataset separately, to show that using FLUTE improves understanding of figurative language. The T5-small model was used to reproduce this experiment on our devices.

## DREAM Model

The only available checkpoint of DREAM was a fine-tuned version of T5-11B, which was too heavy to be used on our computers. Therefore, the DREAM model had to be reproduced at a lower scale, using T5-small.

The first step to reproduce the model was to create the dataset, using the code given in the DREAM paper's repository. This code is available in the notebook [compile_DREAM_dataset.ipynb](compile_DREAM_dataset.ipynb).

The fine-tuning of the T5-small model on this dataset is done in the file [dream.py](dream.py).

## DREAM-FLUTE Model

Using the FLUTE dataset and the DREAM model, we were able to reproduce the ensemble described in the DREAM-FLUTE paper.

Once again, we used T5-small models instead of T5-3B models as basis for all of the 9 systems involved in the ensemble. The creation of the dataset as well as the training of all models is achieved in the notebook [flute-dream-training.ipynb](flute-dream-training.ipynb).

The [flute_dream.py](flute_dream.py) script implements classes to handle both the Pipeline for the System 4 ("Classifiy then Explain") and the ensembling algorithm, that takes a majority vote on the 5 most accurate models for the labels, and picks the explanation of the first model that agrees with this label within a predetermined sequence.

## Extension 1 : Synthetic Data Generation

As the creation process for the FLUTE dataset involves a lot of human intervention, it cannot be automated and thus can only produce a limited amount of figurative data. However, some additional data might be needed to further enhance a model's understanding of figurative language. Therefore, we explored ways to generate automatically synthetic figurative data.

Inspired by the FLUTE creation process, we used a Large Language Model to try obtaining figurative sentences from non-figurative ones. After some tests, it appeared that the [quantized Mistral-7B](https://huggingface.co/TheBloke/CapybaraHermes-2.5-Mistral-7B-GPTQ) model seemed to be the more promising one.

In [synthetic_data_stages.ipynb](synthetic_data_stages.ipynb), we tested generating sarcastic sentences by stages, firstly prompting the model to generate a paraphrase of the sentence, then asking it to mdifiy it so it becomes a contradiction of the original one. We used datasets focusing on a specific type of figurative language ([Empathetic Dialogues](https://huggingface.co/datasets/empathetic_dialogues) for Sarcasm, [metaeval/figurative-nli](https://huggingface.co/datasets/metaeval/figurative-nli) for Similes) to have the model generate new data.

In [synthetic_data_fewshot.ipynb](synthetic_data_fewshot.ipynb), in the contrary, data is generated using only one prompt, which seemed to be more effective. The FLUTE dataset sentences were used to generate synthetic data.

A new T5-small model was fine-tuned on this synthetic data in [t5_with_synthetic.ipynb](t5_with_synthetic.ipynb) to test the quality of the new data.

## Extension 2 : Statistical Tests

As the increasing of performance given by the ensemble is quite low with respect to the performance of some of its models taken alone, we performed statistical tests on each model to evaluate their impact on the final performance. These tests are available in [paired_statistics.ipynb](paired_statistics.ipynb). In this file, System 5 is the ensemble approach. Two additional models were tested : System 6 is the model fine-tuned with our previously generated synthetic data, and System 7 is a model that first generates an explanation given a pair of sentences, and then generates an entailment label. This system aims at reproducing a Chain-of-Thought approach.
