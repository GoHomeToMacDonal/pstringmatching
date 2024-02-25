#include <iostream>

#include <omp.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "similarity_measure/jaccard.hpp"
#include "tokenizer/qgram_tokenizer.hpp"
#include "tokenizer/whitespace_tokenizer.hpp"

namespace py = pybind11;

template <class T>
using QgramTokenizer1 = tokenizer::QgramTokenizer<1, T>;
template <class T>
using QgramTokenizer2 = tokenizer::QgramTokenizer<2, T>;
template <class T>
using QgramTokenizer3 = tokenizer::QgramTokenizer<3, T>;

template <template <class> class Measure, template <typename> class Tokenizer>
struct PyOjbectSimilarityFunction
{
    inline float operator()(PyObject *x, PyObject *y)
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
            return measure_1.get_sim_score(x_tokens, y_tokens);
        }
        else if (char_width == 2)
        {
            typename Measure<typename Tokenizer<wchar_t>::token_type>::container_type x_tokens, y_tokens;
            tokenizer_2(PyUnicode_2BYTE_DATA(x), x_tokens);
            tokenizer_2(PyUnicode_2BYTE_DATA(y), y_tokens);
            return measure_2.get_sim_score(x_tokens, y_tokens);
        }
        else if (char_width == 4)
        {
            typename Measure<typename Tokenizer<int>::token_type>::container_type x_tokens, y_tokens;
            tokenizer_4(PyUnicode_4BYTE_DATA(x), x_tokens);
            tokenizer_4(PyUnicode_4BYTE_DATA(y), y_tokens);
            return measure_4.get_sim_score(x_tokens, y_tokens);
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

template <class SimFunctor>
py::array_t<double> compute_list_similarity(py::list a, py::list b)
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
        output[i] = func(a_ptr[i], b_ptr[i]);
    }

    return result;
}

template <template <class> class Measure, template <typename> class Tokenizer>
py::array_t<double> compute_pairwise_list_similarity(py::list X, py::list Y)
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
        tokenizer(PyUnicode_4BYTE_DATA(x_ptr[i]), _X[i]);
    }

    #pragma omp parallel for
    for (int j = 0; j < N; j++)
    {
        tokenizer(PyUnicode_4BYTE_DATA(y_ptr[j]), _Y[j]);
    }

    #pragma omp parallel for
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            sims[i * N + j] = measure.get_sim_score(_X[i], _Y[j]);
        }
    }

    return result.reshape({M, N});
}

PYBIND11_MODULE(pstringmatching, m)
{
    m.doc() = "Similarity measures"; // optional module docstring

    m.def("jaccard", &compute_list_similarity<PyOjbectSimilarityFunction<similarity_measure::Jaccard, tokenizer::WhitespaceTokenizer>>, "unigram jaccard similarity measure");
    m.def("unigram_jaccard", &compute_list_similarity<PyOjbectSimilarityFunction<similarity_measure::Jaccard, QgramTokenizer1>>, "unigram jaccard similarity measure");
    m.def("bigram_jaccard", &compute_list_similarity<PyOjbectSimilarityFunction<similarity_measure::Jaccard, QgramTokenizer2>>, "bigram jaccard similarity measure");
    m.def("trigram_jaccard", &compute_list_similarity<PyOjbectSimilarityFunction<similarity_measure::Jaccard, QgramTokenizer3>>, "trigram jaccard similarity measures");

    m.def("pairwise_jaccard", &compute_pairwise_list_similarity<similarity_measure::Jaccard, tokenizer::WhitespaceTokenizer>, "unigram jaccard similarity measure");
    m.def("pairwise_unigram_jaccard", &compute_pairwise_list_similarity<similarity_measure::Jaccard, QgramTokenizer1>, "unigram jaccard similarity measure");
    m.def("pairwise_bigram_jaccard", &compute_pairwise_list_similarity<similarity_measure::Jaccard, QgramTokenizer2>, "unigram jaccard similarity measure");
    m.def("pairwise_trigram_jaccard", &compute_pairwise_list_similarity<similarity_measure::Jaccard, QgramTokenizer3>, "unigram jaccard similarity measure");
}