#ifndef PSTRINGMATCHING_SIMILARITY_MEASURE_OVERLAP_COEFFICIENT_HPP
#define PSTRINGMATCHING_SIMILARITY_MEASURE_OVERLAP_COEFFICIENT_HPP

#pragma once

#include <algorithm>
#include <set>
#include <utility>
#include "../util/counter_iterator.hpp"

namespace similarity_measure
{
  template <class token_type>
  struct OverlapCoefficient
  {
    using container_type = std::set<token_type>;

    template <class _container_type>
    inline float get_raw_score(const _container_type &x, const _container_type &y)
    {
      if (x.size() == 0 || y.size() == 0)
      {
        return 0.0f;
      }

      __pstringmatching_impl::counter_iterator<container_type> i, u;
      i = std::set_intersection(std::begin(x), std::end(x), std::begin(y), std::end(y), i);

      auto i_cnt = i.count();

      return i_cnt * 1.0 / std::min(x.size(), y.size());
    }

    template <class _container_type>
    inline float get_sim_score(const _container_type &x, const _container_type &y)
    {
      return get_raw_score(x, y);
    }
  };
}

#endif