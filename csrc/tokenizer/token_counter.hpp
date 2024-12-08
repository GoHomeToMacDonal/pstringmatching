#ifndef REMP_EXT_TOKENIZER_TOKEN_COUNTER_HPP
#define REMP_EXT_TOKENIZER_TOKEN_COUNTER_HPP

#pragma once

#include <array>
#include <map>
#include <algorithm>

namespace tokenizer
{
  template<typename char_type>
  class TokenCounter
  {
  public:
    using token_type = char_type;
    template <class input_type>
    inline bool operator()(const input_type *begin, std::map<token_type, unsigned int> &counter)
    {
      for (; *begin != 0; ++begin)
      {
        counter[(token_type)*begin]++;
      }

      return true;
    }
  };
}

#endif