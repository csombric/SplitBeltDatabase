# What is a split-belt treadmill and what is it used for?

A split-belt treadmill is a treadmill that has two belts, one for each foot. The belts can be controled independently, such that one belt moves faster than the other (i.e., split-belt).  The belts can also be run at the same speed (i.e., tied-belt), which is the same as walking on a normal treadmill with one belt. 

When you walk with split-belts for 10 minutes, you learn a new walking pattern that manifests as a limp when the belts are tied. Importantly, if you already have a limp due to a brain lesions (e.g., stroke) the new walking pattern learned from split-belt walking can cancel out your exisiting limp such that you walk symemtrically.

# What is the goal of this repository?

The goal of this repository is to classify whether individual subject will sucessfully learn* a new walking pattern given a specific split-belt training protocol. 

*More details about the characterization of learning can be found in the "Characterization of Learning" section.
*More details about the relavence of the goal of this repository can be found in the "Background of Problem & Goals" section.

# Who are the stakeholders?

It is currently unclear what split-belt training responses we could even expect from young healthy subjects, much less those with neural or biomechanical pathologies (aging, braining lesions, cogntive decline, amputaiton, etc.). Thus, it is essential to successfully classify which patients could benefit from what kinds of gait training interventions to optimize rehabilitation.  Thus clinicians and patients are important stakeholders, as are insurance companies who want to pay for efficisious treatments.

# Overview of Repository

This repository consists of Jupyter Notebooks of the analysis of a split-belt dataset. 

1. SplitBeltAnalysis_DataClearning.ipynd
2. Learning: TMSteady
   - SplitBeltAnalysis_SteadyState.ipynd
3. Learning: TMAfter
   - SplitBeltAnalysis_AfterEffects.ipynd


# Characterization of Learning

Gait is characterized by a measure of stepping symmetry called Step Length Asymmetry.  Consider that if each step is the same size that Step Length Asymmetry is zero, whereas if one step is longter than the other that Step Length Asymmetry is non-zero. Step Length Asymmetry will be used to charactize changes in gait due to split-belt walking.  

Recall that the goal is to classify wether subjects sucessfully learned a new walking pattern.  Here, learning a new walking patter was assessed with at two time points, TMSteady and TMAfter.  TMSteady is the Step Length Asymmetry at the end the very end of split-belt walking. TMAfter is the Step Length Asymmetry during the tied-belt walking that directly follows split-belt walking.  Any Baseline gait asymmetry is accounted for when presenting TMSteady and TMAfter.

# Analysis

Currently, Random Forests are being used for the selection of individual features. Two and three-way interactions
feature selection are performed. 

Once features are selected, several classifiers, including Logistic Regression, are run and the best models are selected based on the precision and recall measures.

The data analysis is ongoing. Preliminary characterizations of the data are ongoing as more data is added to the lab repositories.


# Background of Problem & Goals

It is currently unclear what split-belt training responses we could even expect from young healthy subjects, much less those with neural or biomechanical pathologies (aging, braining lesions, cogntive decline, amputaiton, etc.). Split-belt results are usually published as group averages as it is often difficult to explain fluctuations within groups.  **The primary goal of this analysis is to be able to predict responses to split-belt training for individual subjects given (1) subject demographics, (2) protocol details, and (3) potentially baseline movement features.**

**A secondary goal is to make good use of the data we have already collected.**  There are many questions that the community may have that can be answered more completely using the large amount of data that has already been collected as opposed to collecting a small group of new subjects.  **I hope that this analysis and creation of the database will serve split-belt researchers and the human motor control and motor adaptation communities well.**

The current plan is to train a series of machine learning algorithms on training data, which will consist of a random sampling fo the population, and access the model on a training set.


# Split-Belt Database

This repository consists of a Jupyter Notebook of the analysis of a split-belt dataset.  The split-belt dataset is currrently compiled from only the Sensorimotor Learning Laboratory data (500+ samples).  Onging efforts are being made to increase the size of the dataset to 5,000 samples by collaborating with other labs.

Current Contributors:
- Sensorimotor Learning Laboratory at the University of Pittsburgh (PI: Dr. Gelsy Torres-Oviedo)

# Database

Each row in the dataset is an indvidual split-belt session.  The columns are different variables, such as **subject demographics**, **experimental protocol details**, and **motor outcomes**.

1. **Subject Demographics**
  -	Age at time of testing
  -	Weight
  -	Height
  -	Gender
  -	Dominant Leg
  -	Clinical Diagnosis
    -Control
    -	Stroke
        -	Cerebellar
        - Cerebral
    - Parkinson’s
    - Amputee
    - Etc.
  -	Clinically Impaired Leg: If applicable, please indicate which was the paretic/impaired/amputated leg.
  -	Slow/Fast Belt to which Leg: Please indicate which leg (right or left) was placed on the slow/fast belt.
  -	If known, how many prior split-belt walking protocols has the subject experienced?  If the exact number is unknown, it is still
  helpful to know if subject are naïve or not.  It is encouraged to include multiple sessions for a single subject.
  
2. **Experimental Procotol Details**
  -	Abrupt vs. Gradual
  -	Mean walking speed during adaptation
  -	Adaptation speed ratio (e.g., 2:1)
  -	Adaptation speed delta (e.g., fast belt speed – slow belt speed)
  -	Short Exposure prior to adaptation?
  -	Duration of Adaptation (strides, steps or time)
  -	Were subjects supposed to be adjusting their movement in response to biofeedback while adapting? 
  -	Event Detection used (kinetic, kinematic, foot switches, etc.)
  -	Handrail used?

3. **Motor Outcomes** (all outcomes are presented for step length asymmetry  $$=\frac{stepLengthFast-stepLengthSlow}{stepLengthFast+stepLengthSlow}$$)
  - Baseline Measures
    - Treadmill (TM):
      - Baseline used for bias removal (the measure will be redundant with one of the following measures)
      - Slow Speed baseline (if collected)
      - Mid Speed baseline (if collected)
      - Fast Speed Baseline (if collected)
    - Overground (OG):
      - Overground baseline used for bias removal
  - Early Adaptation adjusted for bias
  - Steady State adjusted for bias
  - After-Effects adjusted for bias
    - Treadmill (TM):
    - Catch/TM After-Effects adjusted for bias
    - Overground (OG):
    - Overground After-effects adjusted for bias
