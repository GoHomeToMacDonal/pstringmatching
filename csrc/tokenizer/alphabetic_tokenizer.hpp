#ifndef PSTRINGMATCHING_TOKENIZER_ALPHABETIC_TOKENIZER_HPP
#define PSTRINGMATCHING_TOKENIZER_ALPHABETIC_TOKENIZER_HPP

#pragma once

#include <array>
#include <set>
#include <algorithm>
#include <string>

namespace tokenizer
{
  template <typename char_type>
  inline bool is_alphabetic(char_type ch)
  {
    return ((ch >= (char_type)'a') && (ch <= (char_type)'z')) || ((ch >= (char_type)'A') && (ch <= (char_type)'Z'));
  }

  template <typename char_type>
  struct AlphabeticTokenizer
  {
    using token_type = typename std::basic_string<char_type>;

    template <class input_type, class container_type>
    inline bool operator()(input_type *c, container_type &grams) const
    {
      for (auto i = c; *i != 0; ++i)
      {
        if (is_alphabetic(*i))
        {
          auto s = i;
          while ((*i != 0) && is_alphabetic(*i))
            ++i;
          ;
          grams.insert(token_type{s, i});
          if (*i == 0)
            break;
        }
      }

      return true;
    }
  };
}

#endif