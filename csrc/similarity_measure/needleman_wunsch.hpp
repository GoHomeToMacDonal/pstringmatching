#ifndef PSTRINGMATCHING_SIMILARITY_MEASURE_NEEDLEMAN_WUNSCH_HPP
#define PSTRINGMATCHING_SIMILARITY_MEASURE_NEEDLEMAN_WUNSCH_HPP

#pragma once

#include <vector>

namespace similarity_measure
{
  template <class token_type, int GAP=1>
  struct NeedlemanWunsch
  {
    using container_type = std::vector<token_type>;

    template <class container_type>
    inline float get_raw_score(const container_type &x, const container_type &y)
    {
      auto lx = x.size(), ly = y.size();
      auto W = ly + 1;
      float match = 0, remove = 0, insert = 0;

      std::vector<int> dp((lx + 1) * (ly + 1));

      for (int i = 0; i <= lx; i++)
      {
        dp[i * W] = -i * GAP;
      }

      for (int j = 0; j <= ly; j++)
      {
        dp[j] = -j * GAP;
      }

      for (int i = 1; i <= lx; i++)
      {
        for (int j = 1; j <= ly; j++)
        {
          match = dp[(i - 1) * W + j - 1] + (x[i - 1] == y[j - 1] ? 1 : 0);
          remove = dp[(i - 1) * W + j] - GAP;
          insert = dp[i * W + j - 1] - GAP;
          dp[i * W + j] = std::max({match, remove, insert});
        }
      }

      return dp[lx * W + ly];
    }

    template <class container_type>
    inline float get_sim_score(const container_type &x, const container_type &y)
    {
      return get_raw_score<container_type>(x, y);
    }
  };
}

#endif