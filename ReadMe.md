# Paralleled String Similarity Matching Computing Extension
## Build
```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

## Usage
```
import pstringmatching
import numpy as np

X = np.array(['hello', 'world'], dtype=np.str)
Y = np.array(['hallo', 'world'], dtype=np.str)
print(pstringmatching.pairwise_bigram_jaccard(X, Y))
```