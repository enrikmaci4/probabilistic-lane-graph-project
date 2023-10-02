# Contents

_About_ : About this code, about the author.

_Results_ : Some early results to keep you interested.

_How to Use_ : How to use this code to generate your own results.

_Some More Results_ : As the title suggests.

# About

**_What is this?_**

Here is the code for my research project: _Maci, Enrik and Howard, Rhys Peter Matthew and Kunze, Lars (2023) ”Generating and Explaining Corner Cases Using Learnt Probabilistic Lane Graphs”_. This paper was published and presented at the IEEE Intelligent Transportation Systems (ITSC) Conference, 2023 in Bilbao, Spain. This research project aims to generate and explain decision based corner case scenarios for Autonomous Vehicle (AV) training. This codebase can be used to generate spatiotemporal timeline data with multiple agents and can produce visualisations/animations of this artificially generated data.

**_Who am I?_**

I graduated with an MEng from the University of Oxford in 2022 and I have been working as a full-time Software Engineer at Microsoft since graduation. This research publication is a version of my final year Masters Project which I refined over the course of the year outside of my working hours.

**_Who are you?_**

_Someone Who Simply Enjoys Corner Cases:_ Please enjoy this code :)

_An Academic Researcher:_ Please feel free to play with this code and discover its advantages and limitations. A description of what this code can do is written below and contributions of this work are given in the paper. Please contact me with your findings if you decide to investigate this code in more detail. I am very interested in hearing possible advantages/limitations of this work and I am always looking to improve it.

_Recruiter/Hiring Manager:_ I am happy to answer any specific questions about this code, from high level design to specific implementations.

**_Notes_**

I am actively maintaining this, and other publicly available codebases on my GitHub. If you have any queries or find a bug in my code, please do not hesiteate to contact me at: _enrik1@hotmail.co.uk_.

# Results

This code uses _Probabilistic Lane Graphs (PLGs)_ to generate artificial traffic scenarios. The PLG is used to efficiently solve the vehicle path planning problem for large maps. Given the vehicle paths generated, a vehicle action model is then used to modify the path and kinematic state of the vehicles.

The original action model is designed to generate standard traffic scenarios with no collisions. This action model is then modified in order to generate realistic crash events. An example result of this work is shown below. The animation on the left is the generated scenario which does not contain a crash event. This scenario is then optimised in order to generate a realistic crash event which could occurr in this situation.

![anim](https://github.com/enrikmaci4/probabilistic-lane-graph-project/assets/102254720/3e04f71d-2918-4925-9b35-962f93772b27)

Both scenarios in the above example begin from an identical initial state. More details describing this process can be found in the paper.

# How to Use

**_Data required_**

In order for this code to work, sufficient data must be provided as input. There are 7 different columns/vectors of data required, these are: _Vehicle ID_, _Frame ID/Time_, _X Position_, _Y Position_, _Speed_, _Acceleration_, _Lane ID_. An example dataset is already provided within this codebase. This example dataset is the NGSIM, Lankershem dataset which can be found at the following location: https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm. The example dataset is given as 7 different text files in the _data\lankershim\original_ folder. If you would like to add an additional dataset other than the one provided, you must copy the directory structure given by _data\lankershim\*_. I.e., you must create the following directories _data\NewDatasetName\*_, where the * folders should look the same as the example dataset given in the _lankershim_ folder.

**_Codebase structure_**

The code is split up into four distinct sections: _Dataset_, _PLG Generation_, _PLG Visulation/Animation Scripts_, _Simulation Scripts/Models_.

**_What can I change?_**

I like to think that the code is fairly well commented so if you feel like it then jump right in and mess around with the code to see what you can do. However, if you would just like to generate some results then you're looking for _inputs.py_. This defines and explains all variables that the user can adjust in order to modify the results generated. I.e., this file can control the length of the simulations, the max number of background vehicles in the simulation, the node spacing in the PLG, and so on.

Note that since the PLG is saved in the _data\<DatasetName>\structs_ directory, if you make any changes which affect the PLG, i.e., the minimum node spacing, _R_, you would need to re-run _plg_generation.py_ in order to generate a new PLG which will pick up these changes.

**_What scripts should I run and in what order?_**:

_data_cleaner.py_ : Cleans the dataset by removing super long points anomalous points.

_plg_generation.py_ : Generates the PLG structure and saves it.

_plg_visualisation.py_ [Optional] : Plots the PLG for visualisation purposes.

_test_simulation_*.py_ : Runs simulations and saves the output as a Python pickle in the output directory.

_animation*.py_ : Takes the saved Python pickle simulation data and generates GIFs of the form shown above.

# Wow! This is cool, show me some more results

Here are some more interesting simulations I've generated. Please let me know what you think/provide me with any feedback on the method, results, code etc.

![anim_2](https://github.com/enrikmaci4/probabilistic-lane-graph-project/assets/102254720/fa349769-8c02-4c7a-acf1-14d90f29bdcd)

![anim_5](https://github.com/enrikmaci4/probabilistic-lane-graph-project/assets/102254720/7323fb97-8332-41d0-966c-cffa846b5ee0)

![anim_9](https://github.com/enrikmaci4/probabilistic-lane-graph-project/assets/102254720/c2ba24bc-90ea-43c7-8d9a-3adad12f8907)

For more content including an example PLG for some roundabout data please see: https://github.com/cognitive-robots/probabilistic-lane-graph-paper-resources.

