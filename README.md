# Paralleled String Similarity Matching Computing Extension
## Install
```
pip install pybind11
pip install git+https://github.com/GoHomeToMacDonal/pstringmatching.git
```

## Usage
```
import pstringmatching
import numpy as np

X = np.array(['hello', 'world'], dtype=np.str)
Y = np.array(['hallo', 'world'], dtype=np.str)
print(pstringmatching.pairwise_bigram_jaccard(X, Y))
```