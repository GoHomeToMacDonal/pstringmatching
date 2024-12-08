#ifndef PSTRINGMATCHING_SIMILARITY_MEASURE_LEVENSHTEIN_HPP
#define PSTRINGMATCHING_SIMILARITY_MEASURE_LEVENSHTEIN_HPP

#pragma once

#include <vector>

namespace similarity_measure
{
  template <class token_type, int I=1, int D=1, int S=1>
  struct Levenshtein
  {
    using container_type = std::vector<token_type>;

    template <class container_type>
    inline float get_raw_score(const container_type &x, const container_type &y)
    {
      auto lx = x.size(), ly = y.size();
      if (lx == 0)
      {
        return ly * I;
      }

      if (ly == 0) {
        return lx * D;
      }

      auto W = ly + 1;

      std::vector<int> dp((lx + 1) * (ly + 1));

      for (int i = 0; i <= lx; i++) {
        dp[i * W] = i * D;
      }

      for (int j = 0; j <= ly; j++) {
        dp[j] = j * I;
      }

      for (int i = 1; i <= lx; i++) {
        for (int j = 1; j <= ly; j++) {
          int cost = x[i - 1] == y[j - 1] ? 0 : S;
          dp[i * W + j] = std::min({dp[(i - 1) * W + j] + D, dp[i * W + j - 1] + I, dp[(i - 1) * W + j - 1] + cost});
        }
      }

      return dp[lx * W + ly];
    }

    template <class container_type>
    inline float get_sim_score(const container_type &x, const container_type &y)
    {
      if (x.size() == 0 || y.size() == 0) {
        return 1.0f;
      }
      return 1 - 1.0f * get_raw_score<container_type>(x, y) / std::max(x.size(), y.size());
    }
  };
}

#endif