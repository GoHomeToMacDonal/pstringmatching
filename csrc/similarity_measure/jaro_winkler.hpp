#ifndef PSTRINGMATCHING_SIMILARITY_MEASURE_JARO_WINKLER_HPP
#define PSTRINGMATCHING_SIMILARITY_MEASURE_JARO_WINKLER_HPP

#pragma once

#include "jaro.hpp"

namespace similarity_measure
{
  template <class token_type>
  struct JaroWinkler
  {
    using container_type = std::vector<token_type>;

    template <class container_type>
    inline float get_raw_score(const container_type &x, const container_type &y, float prefix_weight = 0.1)
    {
      auto lx = x.size(), ly = y.size();
      if (lx == 0 || ly == 0)
      {
        return 0.0f;
      }

      auto min_len = lx < ly ? lx : ly;
      auto j = min_len < 4 ? min_len : 4;
      auto i = j;
      auto jw_score = jaro.get_raw_score(x, y);

      for (i = 0; i < j && x[i] == y[i]; i++)
        ;
      if (i != 0)
      {
        jw_score += i * prefix_weight * (1.0f - jw_score);
      }

      return jw_score;
    }

    template <class container_type>
    inline float get_sim_score(const container_type &x, const container_type &y, float prefix_weight = 0.1)
    {
      return get_raw_score<container_type>(x, y, prefix_weight);
    }

  protected:
    Jaro<token_type> jaro;
  };
}

#endif