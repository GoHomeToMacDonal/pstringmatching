from pstringmatching import jaccard, unigram_jaccard, bigram_jaccard, trigram_jaccard, pairwise_jaccard, pairwise_unigram_jaccard, pairwise_bigram_jaccard, pairwise_trigram_jaccard
import time

for i in range(0, 1):
    a, b = ["nihao", "nihaoa", "nibuhao"] * 1, ["nihaoa", "nihao", "nibuhao"] * 1
    start_time = time.time()
    print(
        jaccard(a, b),
        unigram_jaccard(a, b),
        bigram_jaccard(a, b),
        trigram_jaccard(a, b)
    )
    print("--- %s seconds ---" % (time.time() - start_time))
    # print(x.shape)

print(pairwise_unigram_jaccard(a * 1000000, b * 100))
