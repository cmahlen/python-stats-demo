The point of this lab is to introduce students to programming-based statistics (in this case, Python). Right before this, they will have done some basic analyses (various t-tests and ANOVAs) using JASP, a GUI-based statistical software. 

This lab will be 45-60 mins (potentially more). These students are assumed to have little to no programming knowledge. 

The learning objectives:
1. Gain basic familiarity with Python programming and some of its libraries. Be able to:
   - Load Excel (.xlsx) and CSV files using pandas
   - Inspect dataframes with .head(), .tail(), and quality checks
   - Make simple graphs (histograms, swarm plots, bar plots)
   - Run statistical tests (t-test, ANOVA, Tukey HSD, mixed ANOVA)
   - Apply the "peek-then-analyze" workflow
2. Understand the tradeoffs between GUI- and code-based statistics
3. Build confidence experimenting with code (changing colors, parameters, etc.)
4. Recognize and interpret basic Python errors 


The source data is found at: 

- class_data_undergrad.xlsx (for ANOVAs)
- class_data_longitudinal.xlsx (two-way ANOVAs)
- Old and New Data Set t-tests.xlsx (for t-tests)

The summary of the JASP output for the ANOVAs is at `Locomotion data.html`. 

The notebook that we are using for now is `Python_vs_JASP_Lab.ipynb`.

Currently, the notebook matches the JASP output exactly.

## Git version control and GitHub

We are uploading the notebook and files to GitHub so that the students can access it via Google Colab on their own browser without downloading anything. Here are some useful commands

Steps to upload $file to GitHub
git add $file
git commit -m "Commite message here..."
git push

How to get rid of outputs from cells (so that exercises are not spoiled)
jupyter nbconvert --clear-output --inplace Python_vs_JASP_Lab_v2.ipynb 

## NEW FEEDBACK

...


## OLD FEEDBACK 
Here are some of the points we need to improve: 

1. The table at the beginning comparing and contrasting JASP and python is a good idea, but it can be improved. `for roi in rois` doesn't make any sense to the student, that is from a later lab. The principle is correct but they may need a different example. Also it is not true that you see "every step" in python, you just see more of them. 
2. The main issue is that since these students have not programmed before, we should consider breaking up some of the longer cells and explaining to them what exactly is going on. For example, the cell that is importing data could be broken up. We don't need to import all the data at the same time up front, let's import it as needed so that they can get practice importing it. They should get used to importing -> using df.head() -> VERY minor cleaning (if necessary). ONLY THEN they should do the basic descriptive stats (mean, SD). 
- After teaching them how to do something once, ask them to do it on their own (e.g., for importing data). Since they are new to this, we still need to provide scaffolding. After asking them to do something, we can give them the answer, hidden, in the form of white text that they cannot see until they highlight and copy/paste. Like this: 
```
The color of this text is <span style="color: red">red</span> and this text is <span style="color: blue">blue</span>.
```
3. They also should be doing "quality checks" on the data. Ideally, these would be using simple functions like df.hist(x) or df.plot(x) that they can easily use themselves. Or similar matplotlib functions.
4. Relatedly, there is a heavy use of text-based stats as the output. Humans (mostly) prefer to see plots. whenever stats are being shown, let's consider asking them to plot it as well. This is especially important for things like equal variances, which will be understood far more quickly with a simple graph. 
5. Motivation for bootstrapping missing. Why should they care? 
6. Repeated measures ANOVA is incomplete; code is in markdown instead of code block for some reason. 
7. "AUTOMATED PAIRWISE T-TESTS" will be confusing to students. How is this different from post-hoc comparisons that JASP does? Why should they care about this? 


Additional feedback:

1. Tone: The “GUIs are training wheels” line and “black box” framing may feel dismissive. A softer framing (“GUI is great for quick checks; code is better for traceability/scale”) keeps students engaged without implying their prior work was lesser.
2. “Runs anywhere” and “you see every step” are overstated. Code still needs a Python environment, and you see more steps, not all steps. A short note about environments (or a preconfigured class environment) would keep this honest.
3. Data import: `header=None` + `iloc[2:, 0]` + `dropna()` is a lot for novices. Consider `skiprows`, `nrows`, or a “peek then clean” flow with `df.head()` and `df.tail()` to show the summary rows they should exclude.
4. Paired t-test demo: Truncating arrays to force pairing can mislead. It’s worth explicitly stating that paired tests require true subject-level pairing and that this is just a “what happens if we misuse the test” example.
5. Output accuracy: In the ANOVA cell, the printed `p < .001` is hard-coded; it should use the computed `p_value`. Same for “key pharmacological findings” p-values which are currently hard-coded. Accuracy matters if this is meant to teach reproducibility.
6. Assumptions and QC: Add a short “sanity check” step before each analysis: `df.info()`, `df.isna().sum()`, `df['Group'].value_counts()`, and a simple box/violin/hist plot. This builds habits for data vetting.
7. Visuals earlier: You already plan more plots; I’d suggest a simple plot immediately after the first import so they see a benefit before the stats.
8. Pairwise tests vs. post‑hoc: The “automated pairwise t-tests” section needs a stronger warning about multiple comparisons and why Tukey/Bonferroni exists. Otherwise it looks like “28 t‑tests is the right thing to do.”
9. Version fragility: `scipy.stats.tukey_hsd` is relatively new. If students have older SciPy, this will fail. You might want a fallback to `statsmodels.stats.multicomp.pairwise_tukeyhsd` or pin versions in a setup cell.
10. Repeated measures section: It’s in markdown and uses `pingouin` with a doc link. If you keep it, either include a minimal runnable example and install note or clearly label it as “optional; not run in class.”
11. Exercises: Add 2–3 “now you try” checkpoints where students fill in one line (e.g., “load Data Set 2”, “compute descriptives”, “plot histogram for Group X”). This builds confidence without overwhelming them.
12. Error literacy: A short “if you see this error, it means…” box (e.g., KeyError for column names, FileNotFoundError for paths) helps first‑timers recover.

Even more feedback: 
1. every time we do something new in code, we need to break it down and explain what we are doing. We do a good job of this at the beginning, but get lazy and skip ahead later on. 
2. Box plots are good, but it might be better to use something like a swarm plot so we are looking the ACTUAL data. Box plots add needless abstraction
3. The "Your Turn" exercises are a good start, but they should only be done when the student has seen an example of something close enough in the past. For example, the first your turn is loading t-test data, but they haven't learned to use arguments like sheet_name or skiprows yet. 
4. The ttest data was annoying and needed cleaning, which isn't the focus here. I made a new file `ttest_data.xlsx` which should be easier to load without adding confusion. 
5. Encourage them to explore different options/arguemnts wherever possible (bins/alpha values, labels, colors, everything. Our goal is to help them build confidence in using python so they can perform better in subsequent labs.). Related: NO hexcodes--encourage them to experiment with colors on their own.
6. No emojis.
7. Let's add to the comparison that learning Python is a general skill: learning to do stats in Python means that you will better be able to learn all of the following: machine learning, neuroimaging, bioinformatics, finance, computer science, web design, and more!
Any other feedback? 