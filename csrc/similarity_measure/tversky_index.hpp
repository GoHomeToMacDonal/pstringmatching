#ifndef REMP_EXT_SIMILARITY_MEASURE_TVERSKY_INDEX_HPP
#define REMP_EXT_SIMILARITY_MEASURE_TVERSKY_INDEX_HPP

#pragma once

#include <algorithm>
#include "../util/counter_iterator.hpp"

namespace similarity_measure
{
  template <class token_type>
  struct TverskyIndex
  {
    using container_type = std::set<token_type>;

    template <class _container_type>
    inline float get_raw_score(const _container_type &x, const _container_type &y, float alpha, float beta)
    {
      if (x.size() == 0 || y.size() == 0)
      {
        return 0.0f;
      }

      auto xi = x.begin(), yi = y.begin();
      auto xe = x.end(), ye = y.end();

      int intersect = 0, x_minus_y = 0, y_minus_x = 0;

      while (xi != xe && yi != ye)
      {
        if (*xi == *yi)
        {
          intersect++;
          xi++;
          yi++;
        }
        else if (*xi < *yi)
        {
          x_minus_y++;
          xi++;
        }
        else
        {
          y_minus_x++;
          yi++;
        }
      }

      for (; xi != xe; ++xi)
      {
        x_minus_y++;
      }

      for (; yi != ye; ++yi)
      {
        y_minus_x++;
      }

      return intersect * 1.0f / (intersect + alpha * x_minus_y + beta * y_minus_x);
    }

    template <class _container_type>
    inline float get_sim_score(const _container_type &x, const _container_type &y, float alpha, float beta)
    {
      return get_raw_score(x, y, alpha, beta);
    }
  };
}

#endif