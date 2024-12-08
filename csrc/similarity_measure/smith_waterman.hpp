#ifndef PSTRINGMATCHING_SIMILARITY_MEASURE_SMITH_WATERMAN_HPP
#define PSTRINGMATCHING_SIMILARITY_MEASURE_SMITH_WATERMAN_HPP

#pragma once

#include <vector>

namespace similarity_measure
{
  template <class token_type, int GAP=1>
  struct SmithWaterman
  {
    using container_type = std::vector<token_type>;

    template <class container_type>
    inline float get_raw_score(const container_type &x, const container_type &y)
    {
      auto lx = x.size(), ly = y.size();
      auto W = ly + 1;
      int match = 0, remove = 0, insert = 0, max_value = 0;

      std::vector<int> dp((lx + 1) * (ly + 1));

      for (int i = 1; i <= lx; i++)
      {
        for (int j = 1; j <= ly; j++)
        {
          match = dp[(i - 1) * W + j - 1] + (x[i - 1] == y[j - 1] ? 1 : 0);
          remove = dp[(i - 1) * W + j] - GAP;
          insert = dp[i * W + j - 1] - GAP;
          dp[i * W + j] = std::max({match, remove, insert, 0});
          max_value = std::max(max_value, dp[i * W + j]);
        }
      }

      return max_value;
    }

    template <class container_type>
    inline float get_sim_score(const container_type &x, const container_type &y)
    {
      return get_raw_score<container_type>(x, y);
    }
  };
}

#endif