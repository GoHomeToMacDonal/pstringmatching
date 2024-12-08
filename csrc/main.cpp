#include <iostream>

#include <omp.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "similarity_measure/jaccard.hpp"
#include "similarity_measure/dice.hpp"
#include "similarity_measure/cosine.hpp"
#include "similarity_measure/overlap_coefficient.hpp"
#include "similarity_measure/bag_distance.hpp"
#include "similarity_measure/jaro.hpp"
#include "similarity_measure/jaro_winkler.hpp"
#include "similarity_measure/levenshtein.hpp"
#include "similarity_measure/needleman_wunsch.hpp"
#include "similarity_measure/smith_waterman.hpp"
#include "similarity_measure/tversky_index.hpp"

#include "tokenizer/qgram_tokenizer.hpp"
#include "tokenizer/whitespace_tokenizer.hpp"
#include "tokenizer/alphabetic_tokenizer.hpp"
#include "tokenizer/alphanumeric_tokenizer.hpp"
#include "tokenizer/token_counter.hpp"

namespace py = pybind11;

template <class T>
using QgramTokenizer1 = tokenizer::QgramTokenizer<1, T>;
template <class T>
using QgramTokenizer2 = tokenizer::QgramTokenizer<2, T>;
template <class T>
using QgramTokenizer3 = tokenizer::QgramTokenizer<3, T>;

template <template <class> class Measure, template <typename> class Tokenizer, class... Args>
struct PyOjbectSimilarityFunction
{
    inline float operator()(PyObject *x, PyObject *y, Args... args)
    {
#ifdef _WIN32
        typename Measure<typename Tokenizer<int>::token_type>::container_type x_tokens, y_tokens;
        tokenizer_4(PyUnicode_4BYTE_DATA(x), x_tokens);
        tokenizer_4(PyUnicode_4BYTE_DATA(y), y_tokens);
        return measure_4.get_sim_score(x_tokens, y_tokens);
#else
        int char_width = std::max(PyUnicode_KIND(x), PyUnicode_KIND(y));

        if (char_width == 1)
        {
            typename Measure<typename Tokenizer<char>::token_type>::container_type x_tokens, y_tokens;
            tokenizer_1(PyUnicode_1BYTE_DATA(x), x_tokens);
            tokenizer_1(PyUnicode_1BYTE_DATA(y), y_tokens);
            return measure_1.get_sim_score(x_tokens, y_tokens, args...);
        }
        else if (char_width == 2)
        {
            typename Measure<typename Tokenizer<wchar_t>::token_type>::container_type x_tokens, y_tokens;
            tokenizer_2(PyUnicode_2BYTE_DATA(x), x_tokens);
            tokenizer_2(PyUnicode_2BYTE_DATA(y), y_tokens);
            return measure_2.get_sim_score(x_tokens, y_tokens, args...);
        }
        else if (char_width == 4)
        {
            typename Measure<typename Tokenizer<int>::token_type>::container_type x_tokens, y_tokens;
            tokenizer_4(PyUnicode_4BYTE_DATA(x), x_tokens);
            tokenizer_4(PyUnicode_4BYTE_DATA(y), y_tokens);
            return measure_4.get_sim_score(x_tokens, y_tokens, args...);
        }
        else
        {
            throw std::runtime_error("unexpected string format");
        }
#endif
    }

private:
    Tokenizer<char> tokenizer_1;
    Tokenizer<wchar_t> tokenizer_2;
    Tokenizer<int> tokenizer_4;
    Measure<typename Tokenizer<char>::token_type> measure_1;
    Measure<typename Tokenizer<wchar_t>::token_type> measure_2;
    Measure<typename Tokenizer<int>::token_type> measure_4;
};

template <class SimFunctor, class ...Args>
py::array_t<double> compute_list_similarity(py::list a, py::list b, Args... args)
{
    SimFunctor func;

    py::array_t<double> result(std::min(a.size(), b.size()));
    py::buffer_info result_buf = result.request();
    double *output = static_cast<double *>(result_buf.ptr);
    int M = std::min(a.size(), b.size());

    PyObject **a_ptr = ((PyListObject *)a.ptr())->ob_item;
    PyObject **b_ptr = ((PyListObject *)b.ptr())->ob_item;

#pragma omp parallel for
    for (int i = 0; i < M; i++)
    {
        output[i] = func(a_ptr[i], b_ptr[i], args...);
    }

    return result;
}

template <template <class> class Measure, template <typename> class Tokenizer, class ...Args>
py::array_t<double> compute_pairwise_list_similarity(py::list X, py::list Y, Args... args)
{
    using container_type = typename Measure<typename Tokenizer<int>::token_type>::container_type;
    Tokenizer<int> tokenizer;
    Measure<typename Tokenizer<int>::token_type> measure;

    unsigned long M = X.size(), N = Y.size();
    py::array_t<double> result(M * N);
    py::buffer_info result_buf = result.request();
    double *sims = static_cast<double *>(result_buf.ptr);

    std::vector<container_type> _X(M), _Y(N);
    PyObject **x_ptr = ((PyListObject *)X.ptr())->ob_item;
    PyObject **y_ptr = ((PyListObject *)Y.ptr())->ob_item;

#pragma omp parallel for
    for (int i = 0; i < M; i++)
    {
        switch (PyUnicode_KIND(x_ptr[i]))
        {
        case 1:
            tokenizer(PyUnicode_1BYTE_DATA(x_ptr[i]), _X[i]);
            break;
        case 2:
            tokenizer(PyUnicode_2BYTE_DATA(x_ptr[i]), _X[i]);
            break;
        case 4:
            tokenizer(PyUnicode_4BYTE_DATA(x_ptr[i]), _X[i]);
            break;
        }
    }

#pragma omp parallel for
    for (int j = 0; j < N; j++)
    {
        switch (PyUnicode_KIND(y_ptr[j]))
        {
        case 1:
            tokenizer(PyUnicode_1BYTE_DATA(y_ptr[j]), _Y[j]);
            break;
        case 2:
            tokenizer(PyUnicode_2BYTE_DATA(y_ptr[j]), _Y[j]);
            break;
        case 4:
            tokenizer(PyUnicode_4BYTE_DATA(y_ptr[j]), _Y[j]);
            break;
        }
    }

#pragma omp parallel for
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            sims[i * N + j] = measure.get_sim_score(_X[i], _Y[j], args...);
        }
    }

    return result.reshape({M, N});
}

PYBIND11_MODULE(pstringmatching, m)
{
    m.doc() = "Similarity measures"; // optional module docstring

    // bag distance
    m.def("bag_distance", &compute_list_similarity<PyOjbectSimilarityFunction<similarity_measure::BagDistance, tokenizer::TokenCounter>>, "bag distance");
    m.def("pairwise_bag_distance", &compute_pairwise_list_similarity<similarity_measure::BagDistance, tokenizer::TokenCounter>, "bag distance");

    // jaro
    m.def("jaro", &compute_list_similarity<PyOjbectSimilarityFunction<similarity_measure::Jaro, tokenizer::UnigramTokenizer>>, "jaro");
    m.def("pairwise_jaro", &compute_pairwise_list_similarity<similarity_measure::Jaro, tokenizer::UnigramTokenizer>, "jaro");

    // jaro winkler
    m.def("jaro_winkler", &compute_list_similarity<PyOjbectSimilarityFunction<similarity_measure::JaroWinkler, tokenizer::UnigramTokenizer, float>, float>, "jaro winkler");
    m.def("pairwise_jaro_winkler", &compute_pairwise_list_similarity<similarity_measure::JaroWinkler, tokenizer::UnigramTokenizer, float>, "jaro winkler");

    // levenshtein
    m.def("levenshtein", &compute_list_similarity<PyOjbectSimilarityFunction<similarity_measure::Levenshtein, tokenizer::UnigramTokenizer>>, "levenshtein");
    m.def("pairwise_levenshtein", &compute_pairwise_list_similarity<similarity_measure::Levenshtein, tokenizer::UnigramTokenizer>, "levenshtein");

    // needleman wunsch
    m.def("needleman_wunsch", &compute_list_similarity<PyOjbectSimilarityFunction<similarity_measure::NeedlemanWunsch, tokenizer::UnigramTokenizer>>, "needleman wunsch");
    m.def("pairwise_needleman_wunsch", &compute_pairwise_list_similarity<similarity_measure::NeedlemanWunsch, tokenizer::UnigramTokenizer>, "needleman wunsch");

    // needleman wunsch
    m.def("smith_waterman", &compute_list_similarity<PyOjbectSimilarityFunction<similarity_measure::SmithWaterman, tokenizer::UnigramTokenizer>>, "smith waterman");
    m.def("pairwise_smith_waterman", &compute_pairwise_list_similarity<similarity_measure::SmithWaterman, tokenizer::UnigramTokenizer>, "smith waterman");

    // jaccard similarity measures
    m.def("jaccard", &compute_list_similarity<PyOjbectSimilarityFunction<similarity_measure::Jaccard, tokenizer::WhitespaceTokenizer>>, "jaccard similarity measure with whitespace tokenizer");
    m.def("alphabetic_jaccard", &compute_list_similarity<PyOjbectSimilarityFunction<similarity_measure::Jaccard, tokenizer::AlphabeticTokenizer>>, "jaccard similarity measure with alphabetic tokenizer");
    m.def("alphanumeric_jaccard", &compute_list_similarity<PyOjbectSimilarityFunction<similarity_measure::Jaccard, tokenizer::AlphanumericTokenizer>>, "jaccard similarity measure with alphanumeric tokenizer");

    m.def("unigram_jaccard", &compute_list_similarity<PyOjbectSimilarityFunction<similarity_measure::Jaccard, QgramTokenizer1>>, "unigram jaccard similarity measure");
    m.def("bigram_jaccard", &compute_list_similarity<PyOjbectSimilarityFunction<similarity_measure::Jaccard, QgramTokenizer2>>, "bigram jaccard similarity measure");
    m.def("trigram_jaccard", &compute_list_similarity<PyOjbectSimilarityFunction<similarity_measure::Jaccard, QgramTokenizer3>>, "trigram jaccard similarity measures");

    m.def("pairwise_jaccard", &compute_pairwise_list_similarity<similarity_measure::Jaccard, tokenizer::WhitespaceTokenizer>, "jaccard similarity measure with whitespace tokenizer");
    m.def("pairwise_alphabetic_jaccard", &compute_pairwise_list_similarity<similarity_measure::Jaccard, tokenizer::AlphabeticTokenizer>, "jaccard similarity measure with alphabetic tokenizer");
    m.def("pairwise_alphanumeric_jaccard", &compute_pairwise_list_similarity<similarity_measure::Jaccard, tokenizer::AlphanumericTokenizer>, "jaccard similarity measure with alphanumeric tokenizer");

    m.def("pairwise_unigram_jaccard", &compute_pairwise_list_similarity<similarity_measure::Jaccard, QgramTokenizer1>, "unigram jaccard similarity measure");
    m.def("pairwise_bigram_jaccard", &compute_pairwise_list_similarity<similarity_measure::Jaccard, QgramTokenizer2>, "unigram jaccard similarity measure");
    m.def("pairwise_trigram_jaccard", &compute_pairwise_list_similarity<similarity_measure::Jaccard, QgramTokenizer3>, "unigram jaccard similarity measure");

    // dice similarity measures
    m.def("dice", &compute_list_similarity<PyOjbectSimilarityFunction<similarity_measure::Dice, tokenizer::WhitespaceTokenizer>>, "dice similarity measure with whitespace tokenizer");
    m.def("alphabetic_dice", &compute_list_similarity<PyOjbectSimilarityFunction<similarity_measure::Dice, tokenizer::AlphabeticTokenizer>>, "dice similarity measure with alphabetic tokenizer");
    m.def("alphanumeric_dice", &compute_list_similarity<PyOjbectSimilarityFunction<similarity_measure::Dice, tokenizer::AlphanumericTokenizer>>, "dice similarity measure with alphanumeric tokenizer");

    m.def("unigram_dice", &compute_list_similarity<PyOjbectSimilarityFunction<similarity_measure::Dice, QgramTokenizer1>>, "unigram dice similarity measure");
    m.def("bigram_dice", &compute_list_similarity<PyOjbectSimilarityFunction<similarity_measure::Dice, QgramTokenizer2>>, "bigram dice similarity measure");
    m.def("trigram_dice", &compute_list_similarity<PyOjbectSimilarityFunction<similarity_measure::Dice, QgramTokenizer3>>, "trigram dice similarity measures");

    m.def("pairwise_dice", &compute_pairwise_list_similarity<similarity_measure::Dice, tokenizer::WhitespaceTokenizer>, "dice similarity measure with whitespace tokenizer");
    m.def("pairwise_alphabetic_dice", &compute_pairwise_list_similarity<similarity_measure::Dice, tokenizer::AlphabeticTokenizer>, "dice similarity measure with alphabetic tokenizer");
    m.def("pairwise_alphanumeric_dice", &compute_pairwise_list_similarity<similarity_measure::Dice, tokenizer::AlphanumericTokenizer>, "dice similarity measure with alphanumeric tokenizer");

    m.def("pairwise_unigram_dice", &compute_pairwise_list_similarity<similarity_measure::Dice, QgramTokenizer1>, "unigram dice similarity measure");
    m.def("pairwise_bigram_dice", &compute_pairwise_list_similarity<similarity_measure::Dice, QgramTokenizer2>, "unigram dice similarity measure");
    m.def("pairwise_trigram_dice", &compute_pairwise_list_similarity<similarity_measure::Dice, QgramTokenizer3>, "unigram dice similarity measure");
    
    // cosine similarity measures
    m.def("cosine", &compute_list_similarity<PyOjbectSimilarityFunction<similarity_measure::Cosine, tokenizer::WhitespaceTokenizer>>, "cosine similarity measure with whitespace tokenizer");
    m.def("alphabetic_cosine", &compute_list_similarity<PyOjbectSimilarityFunction<similarity_measure::Cosine, tokenizer::AlphabeticTokenizer>>, "cosine similarity measure with alphabetic tokenizer");
    m.def("alphanumeric_cosine", &compute_list_similarity<PyOjbectSimilarityFunction<similarity_measure::Cosine, tokenizer::AlphanumericTokenizer>>, "cosine similarity measure with alphanumeric tokenizer");

    m.def("unigram_cosine", &compute_list_similarity<PyOjbectSimilarityFunction<similarity_measure::Cosine, QgramTokenizer1>>, "unigram cosine similarity measure");
    m.def("bigram_cosine", &compute_list_similarity<PyOjbectSimilarityFunction<similarity_measure::Cosine, QgramTokenizer2>>, "bigram cosine similarity measure");
    m.def("trigram_cosine", &compute_list_similarity<PyOjbectSimilarityFunction<similarity_measure::Cosine, QgramTokenizer3>>, "trigram cosine similarity measures");

    m.def("pairwise_cosine", &compute_pairwise_list_similarity<similarity_measure::Cosine, tokenizer::WhitespaceTokenizer>, "cosine similarity measure with whitespace tokenizer");
    m.def("pairwise_alphabetic_cosine", &compute_pairwise_list_similarity<similarity_measure::Cosine, tokenizer::AlphabeticTokenizer>, "cosine similarity measure with alphabetic tokenizer");
    m.def("pairwise_alphanumeric_cosine", &compute_pairwise_list_similarity<similarity_measure::Cosine, tokenizer::AlphanumericTokenizer>, "cosine similarity measure with alphanumeric tokenizer");

    m.def("pairwise_unigram_cosine", &compute_pairwise_list_similarity<similarity_measure::Cosine, QgramTokenizer1>, "unigram cosine similarity measure");
    m.def("pairwise_bigram_cosine", &compute_pairwise_list_similarity<similarity_measure::Cosine, QgramTokenizer2>, "unigram cosine similarity measure");
    m.def("pairwise_trigram_cosine", &compute_pairwise_list_similarity<similarity_measure::Cosine, QgramTokenizer3>, "unigram cosine similarity measure");

    // overlap coefficient similarity measures
    m.def("overlap_coefficient", &compute_list_similarity<PyOjbectSimilarityFunction<similarity_measure::OverlapCoefficient, tokenizer::WhitespaceTokenizer>>, "overlap_coefficient similarity measure with whitespace tokenizer");
    m.def("alphabetic_overlap_coefficient", &compute_list_similarity<PyOjbectSimilarityFunction<similarity_measure::OverlapCoefficient, tokenizer::AlphabeticTokenizer>>, "overlap_coefficient similarity measure with alphabetic tokenizer");
    m.def("alphanumeric_overlap_coefficient", &compute_list_similarity<PyOjbectSimilarityFunction<similarity_measure::OverlapCoefficient, tokenizer::AlphanumericTokenizer>>, "overlap_coefficient similarity measure with alphanumeric tokenizer");

    m.def("unigram_overlap_coefficient", &compute_list_similarity<PyOjbectSimilarityFunction<similarity_measure::OverlapCoefficient, QgramTokenizer1>>, "unigram overlap coefficient similarity measure");
    m.def("bigram_overlap_coefficient", &compute_list_similarity<PyOjbectSimilarityFunction<similarity_measure::OverlapCoefficient, QgramTokenizer2>>, "bigram overlap coefficient similarity measure");
    m.def("trigram_overlap_coefficient", &compute_list_similarity<PyOjbectSimilarityFunction<similarity_measure::OverlapCoefficient, QgramTokenizer3>>, "trigram overlap coefficient similarity measures");

    m.def("pairwise_overlap_coefficient", &compute_pairwise_list_similarity<similarity_measure::OverlapCoefficient, tokenizer::WhitespaceTokenizer>, "overlap_coefficient similarity measure with whitespace tokenizer");
    m.def("pairwise_alphabetic_overlap_coefficient", &compute_pairwise_list_similarity<similarity_measure::OverlapCoefficient, tokenizer::AlphabeticTokenizer>, "overlap_coefficient similarity measure with alphabetic tokenizer");
    m.def("pairwise_alphanumeric_overlap_coefficient", &compute_pairwise_list_similarity<similarity_measure::OverlapCoefficient, tokenizer::AlphanumericTokenizer>, "overlap_coefficient similarity measure with alphanumeric tokenizer");

    m.def("pairwise_unigram_overlap_coefficient", &compute_pairwise_list_similarity<similarity_measure::OverlapCoefficient, QgramTokenizer1>, "unigram overlap coefficient similarity measure");
    m.def("pairwise_bigram_overlap_coefficient", &compute_pairwise_list_similarity<similarity_measure::OverlapCoefficient, QgramTokenizer2>, "unigram overlap coefficient similarity measure");
    m.def("pairwise_trigram_overlap_coefficient", &compute_pairwise_list_similarity<similarity_measure::OverlapCoefficient, QgramTokenizer3>, "unigram overlap coefficient similarity measure");
    
    // tversky index similarity measures
    m.def("tversky_index", &compute_list_similarity<PyOjbectSimilarityFunction<similarity_measure::TverskyIndex, tokenizer::WhitespaceTokenizer, float, float>, float, float>, "tversky index similarity measure with whitespace tokenizer");
    m.def("alphabetic_tversky_index", &compute_list_similarity<PyOjbectSimilarityFunction<similarity_measure::TverskyIndex, tokenizer::AlphabeticTokenizer, float, float>, float, float>, "tversky index similarity measure with alphabetic tokenizer");
    m.def("alphanumeric_tversky_index", &compute_list_similarity<PyOjbectSimilarityFunction<similarity_measure::TverskyIndex, tokenizer::AlphanumericTokenizer, float, float>, float, float>, "tversky index similarity measure with alphanumeric tokenizer");

    m.def("unigram_tversky_index", &compute_list_similarity<PyOjbectSimilarityFunction<similarity_measure::TverskyIndex, QgramTokenizer1, float, float>, float, float>, "unigram tversky index similarity measure");
    m.def("bigram_tversky_index", &compute_list_similarity<PyOjbectSimilarityFunction<similarity_measure::TverskyIndex, QgramTokenizer2, float, float>, float, float>, "bigram tversky index similarity measure");
    m.def("trigram_tversky_index", &compute_list_similarity<PyOjbectSimilarityFunction<similarity_measure::TverskyIndex, QgramTokenizer3, float, float>, float, float>, "trigram tversky index similarity measures");

    m.def("pairwise_tversky_index", &compute_pairwise_list_similarity<similarity_measure::TverskyIndex, tokenizer::WhitespaceTokenizer, float, float>, "tversky index similarity measure with whitespace tokenizer");
    m.def("pairwise_alphabetic_tversky_index", &compute_pairwise_list_similarity<similarity_measure::TverskyIndex, tokenizer::AlphabeticTokenizer, float, float>, "tversky index similarity measure with alphabetic tokenizer");
    m.def("pairwise_alphanumeric_tversky_index", &compute_pairwise_list_similarity<similarity_measure::TverskyIndex, tokenizer::AlphanumericTokenizer, float, float>, "tversky index similarity measure with alphanumeric tokenizer");

    m.def("pairwise_unigram_tversky_index", &compute_pairwise_list_similarity<similarity_measure::TverskyIndex, QgramTokenizer1, float, float>, "unigram tversky index similarity measure");
    m.def("pairwise_bigram_tversky_index", &compute_pairwise_list_similarity<similarity_measure::TverskyIndex, QgramTokenizer2, float, float>, "unigram tversky index similarity measure");
    m.def("pairwise_trigram_tversky_index", &compute_pairwise_list_similarity<similarity_measure::TverskyIndex, QgramTokenizer3, float, float>, "unigram tversky index similarity measure");
}