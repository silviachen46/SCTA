# SCAagent
update: moved to Kaggle for memory requirement for large dataset:
can be accessed here: https://www.kaggle.com/code/silvia46/sca-analysis

currently on T-cell and B-cell annotation:
in preprocessing some R function are used and are not available in python, how do we substitute it?
the paper mentioned annotation based on canonical genes, but no specific detail about which ones to look at for specific types in T-cells and B-cells

## features design

#### input: a folder of single cell data

#### read initial gene data file, column info

#### identify main steps (action units)

#### split tasks into specific agents, agents decide what tools to use and test with paramters

#### coding and debugging

#### evaluate on file by file basis
current need more info on this step, how do we evaluate each single cell analysis result

#### cumulate the result
