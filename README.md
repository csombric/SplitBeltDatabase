# What is a split-belt treadmill and what is it used for?

A split-belt treadmill is a treadmill that has two belts, one for each foot. The belts can be controlled independently, such that one belt moves faster than the other (i.e., split-belt).  The belts can also be run at the same speed (i.e., tied-belt), which is the same as walking on a normal treadmill with one belt. 

When you walk with split-belts for 10 minutes, you learn a new walking pattern that manifests as a limp when the belts are tied. Importantly, if you already have a limp due to a brain lesion (e.g., stroke) the new walking pattern learned from split-belt walking can cancel out your existing limp such that you walk symmetrically.

# What is the goal of this repository?

The goal of this repository is to classify whether individual subjects will successfully learn* a new walking pattern given a specific split-belt training protocol. 

*More details about the characterization of learning can be found in the "Characterization of Learning" section.
*More details about the relevance of the goal of this repository can be found in the "Background of Problem & Goals" section.

# Who are the stakeholders?

It is currently unclear what split-belt training responses we can expect even from young healthy subjects, much less those with neural or biomechanical pathologies (aging, braining lesions, cognitive decline, amputation, etc.). Thus, it is essential to successfully classify which patients could benefit from what kinds of gait training interventions to optimize rehabilitation.  Thus clinicians and patients are important stakeholders, as are insurance companies who want to pay for efficacious treatments.

# Overview of Repository and Analysis

This repository consists of Jupyter Notebooks and Python scripts:

1. **SplitBeltAnalysis_DataClearning.ipynd**: This notebook takes an existing dataset (described in more detail below) for cleaning and restructuring for the current classification problem.
2. **SplitBeltAnalysis_TraingingTestingBatching.py**: This script trains and tests many different models.  The models are run on different splits of the data.  Performance metrics for each model and each split are stored in an output file.
3. **SplitBeltAnalysis_MethodSelection.ipynd**: This script takes the output of SplitBeltAnalysis_TraingingTestingBatching.py and visualizes the performance of these models to identify robustly performing models.

# Characterization of Learning

Gait is characterized by a measure of stepping symmetry called **Step Length Asymmetry**.  Consider that if each step is the same size that Step Length Asymmetry is zero, whereas if one step is longer than the other that Step Length Asymmetry is non-zero. Step Length Asymmetry will be used to characterize changes in gait due to split-belt walking.  

![\Large Step Length Asymmetry=\frac{stepLengthFast-stepLengthSlow}{stepLengthFast+stepLengthSlow}](https://latex.codecogs.com/svg.latex?x%3D%5Cfrac%7B-b%5Cpm%5Csqrt%7Bb%5E2-4ac%7D%7D%7B2a%7D)

Recall that the goal is to classify whether subjects successfully learned a new walking pattern.  Here, learning a new walking pattern was assessed with TMAfter.  **TMAfter** is the Step Length Asymmetry during the tied-belt walking that directly follows split-belt walking relative to the subject's baseline Step Length Asymmetry. 

# Background of Problem & Goals

It is currently unclear what split-belt training responses we could even expect from young healthy subjects, much less those with neural or biomechanical pathologies (aging, braining lesions, cognitive decline, amputation, etc.). Split-belt results are usually published as group averages as it is often difficult to explain fluctuations within groups.  **The primary goal of this analysis is to be able to classify responses to split-belt training for individual subjects given (1) subject demographics, (2) protocol details, and (3) baseline movement features.**

**A secondary goal is to make good use of the data we have already collected.**  There are many questions that the scientific split-belt community may have that can be answered more completely using the large amount of data that has already been collected as opposed to collecting a small group of new subjects. ** I hope that this analysis and creation of the database will serve split-belt researchers and the human motor control and motor adaptation communities well.**

The current plan is to train a series of machine learning algorithms on training data, which will consist of a random sampling of the population, and access the model on a training set.

# Split-Belt Dataset

This repository consists of a Jupyter Notebook of the analysis of a split-belt dataset.  The split-belt dataset is currently compiled from only the Sensorimotor Learning Laboratory data (700+ samples).  Ongoing efforts are being made to increase the size of the dataset in collaboration with other labs.

Each row in the dataset is an individual split-belt session.  The columns are different variables, such as **subject demographics**, **experimental protocol details**, and **motor outcomes**.

1. **Subject Demographics**
  -	Age at time of testing
  -	Weight
  -	Height
  -	Gender
  -	Clinical Diagnosis
  -	Naive indicates if the subject has previous split-belt experience influences motor learning. 
  
2. **Experimental Procotol Details**
  -	Abrupt vs. Gradual
  -	Mean walking speed during adaptation
  -	Adaptation speed ratio (e.g., 2:1)
  -	Adaptation speed delta (e.g., fast belt speed – slow belt speed)
  -	Duration of Adaptation (strides, steps, or time)

3. **Motor Outcomes** (all outcomes are presented for step length asymmetry)
  - Baseline Step Length Asymmetry 
  - Baseline Step Length Asymmetry Standard Deviation
  - Treadmill Step Length Asymmetry After-Effects (TMAfter) with Baseline Step Length Asymmetry removed
