# Research Methods: Python Data Analysis Labs

Course materials for NSCI 202 Experimental Design and Data Analysis, Spring 2026. All labs run in Google Colab -- no installation required.

---

## Lab 1: Python vs. JASP

Introduction to statistical analysis in Python, contrasted with the JASP GUI. Covers t-tests, one-way ANOVA, two-way mixed ANOVA, Tukey HSD, and bootstrap confidence intervals.

| Notebook | Open |
|----------|------|
| Python vs. JASP Lab | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cmahlen/python-stats-demo/blob/main/Python_vs_JASP_Lab_v2.ipynb) |

---

## Lab 2: fMRI Lab

Hands-on lab using simulated neuroimaging data. Students test brain-behavior relationships across thousands of functional connectivity edges and discover how analysis choices affect reproducibility.

**Exploratory group** (anxiety, pain, or depression topic assigned by instructor):

| Notebook | Open |
|----------|------|
| Day 1 -- Exploratory Analysis | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cmahlen/python-stats-demo/blob/main/fMRI_Lab_Exploratory.ipynb) |
| Day 2 -- Validation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cmahlen/python-stats-demo/blob/main/fMRI_Lab_Exploratory_Day2.ipynb) |

**Hypothesis-driven group** (pain or depression topic assigned by instructor):

| Notebook | Open |
|----------|------|
| Day 1 -- Pain (NAc-vmPFC) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cmahlen/python-stats-demo/blob/main/fMRI_Lab_Hypothesis_Pain.ipynb) |
| Day 1 -- Depression (Salience Network) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cmahlen/python-stats-demo/blob/main/fMRI_Lab_Hypothesis_Depression.ipynb) |
| Day 2 -- Validation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cmahlen/python-stats-demo/blob/main/fMRI_Lab_Hypothesis_Day2.ipynb) |

---

## Data

The `data/` folder contains simulated fMRI functional connectivity datasets (.npz files). Each dataset includes 200 subjects, 216 brain regions, and 23,220 connectivity edges.
