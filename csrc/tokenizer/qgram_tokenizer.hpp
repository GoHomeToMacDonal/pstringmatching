#ifndef PSTRINGMATCHING_TOKENIZER_QGRAM_TOKENIZER_HPP
#define PSTRINGMATCHING_TOKENIZER_QGRAM_TOKENIZER_HPP

#pragma once

#include <array>
#include <set>
#include <algorithm>

namespace tokenizer
{

  template <int qval, typename char_type>
  class QgramTokenizer
  {
  public:
    using token_type = typename std::array<wchar_t, qval>;
    template <class input_type, class container_type>
    inline bool operator()(const input_type *begin, container_type &grams)
    {
      token_type gram;
      std::array<char_type, 2 * qval - 2> buf;
      buf.fill(0);

      auto output_it = std::begin(buf) + qval - 1, output_end = std::end(buf), input_end = std::end(buf);
      auto i = begin;

      for (; *i != 0 && output_it != output_end; ++i, ++output_it)
      {
        *output_it = *i;
        std::copy(output_it - qval + 1, output_it + 1, std::begin(gram));
        grams.insert(gram);
      }

      for (; output_it < output_end; ++output_it)
      {
        std::copy(output_it - qval + 1, output_it + 1, std::begin(gram));
        grams.insert(gram);
      }

      if (*i != 0)
      {
        // length of string >= N
        for (i = begin; i[qval] != 0; ++i)
        {
          std::copy(i, i + qval, std::begin(gram));
          grams.insert(gram);
        }
        std::copy(i, i + qval, std::begin(gram));
        grams.insert(gram);
        buf.fill(0);
        std::copy(i + 1, i + qval, std::begin(buf));
        input_end = std::begin(buf) + qval - 1;
      }
      else
      {
        buf.fill(0);
        std::copy(begin, i, std::begin(buf));
        input_end = std::begin(buf) + (i - begin);
      }

      for (auto input_it = std::begin(buf); input_it != input_end; ++input_it)
      {
        std::copy(input_it, input_it + qval, std::begin(gram));
        grams.insert(gram);
      }

      return true;
    }
  };

  template <typename char_type>
  class UnigramTokenizer
  {
  public:
    using token_type = int;
    template <class input_type, class container_type>
    inline bool operator()(const input_type *begin, container_type &grams)
    {
      for (; *begin != 0; ++begin)
      {
        grams.push_back((token_type)*begin);
      }
      return true;
    }
  };
}

#endif