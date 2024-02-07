# Pathway Feature Identification

This script, "Pathway_Feature_Identification.py", is designed to analyze microbial genomic data, particularly focusing on identifying pathway features associated with antimicrobial resistance (AMR). It utilizes KEGG pathway data and applies logistic regression for feature identification.

## Overview

The script preprocesses input genomic data and KEGG pathway information to identify pathways associated with AMR. It performs feature selection using logistic regression to determine the significance of each pathway in predicting AMR.

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Usage

```bash
python Pathway_Feature_Identification.py

Input Files:
Susceptibility Groups File: input/susceptibility_groups.txt
KEGG Pathway Data File: input/KEGG_pathways.tab

Primary Output:
output/module_association_plot_corrected_legend.pdf: A PDF plot showing the association of KEGG modules with carbapenem resistance, with a corrected legend.
output/module_association_plot_corrected_legend_coefficients.txt: A text file containing the coefficients of the logistic regression model for each KEGG module.

## Example Input Files

Susceptibility Groups File (input/susceptibility_groups.txt):

Organism    Category
Organism1   Carbapenem-Resistant
Organism2   Carbapenem-Susceptible
...

KEGG Pathway Data File (input/KEGG_pathways.tab):

module  name        pathway group    Organism1   Organism2   ...
M00001  Glycolysis  Carbohydrate Metabolism 1 0 ...
M00002  TCA cycle   Energy Metabolism   1 1 ...
...


## Acknowledgements

This script is designed for analyzing microbial genomic data, identifying antimicrobial resistance-associated pathways using KEGG data, and applying logistic regression for feature selection. We acknowledge the developers of KEGG for their valuable resource in pathway analysis. KEGG documentation is available at KEGG website.

## Citation

If you use this script for your research, please consider citing it as follows:

Sharma, V. (2024). Pathway_Feature_Identification.py [Python script]. Retrieved from https://github.com/vsmicrogenomics/PanGenomeAnalysisTool
