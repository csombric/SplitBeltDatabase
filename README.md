# SplitBeltDatabase

This repository consists of a Jupyter Notebook of the analysis of a split-belt dataset.  The split-belt dataset is currrently compiled from only the Sensorimotor Learning Laboratory data (500+ samples).  I am seeking to increase the data contained in the dataset to 5,000 by collaborating with other labs.

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


In the future a table better describing the each variables content's might be good

Also need to lay out what analysis is in the Jyputer Notebook
