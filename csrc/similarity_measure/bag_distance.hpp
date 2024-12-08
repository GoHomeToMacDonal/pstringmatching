#ifndef PSTRINGMATCHING_SIMILARITY_MEASURE_BAG_DISTANCE_HPP
#define PSTRINGMATCHING_SIMILARITY_MEASURE_BAG_DISTANCE_HPP

#pragma once

#include <algorithm>
#include <set>
#include <map>

namespace similarity_measure
{
  template <class token_type>
  struct BagDistance
  {
    using container_type = std::map<token_type, unsigned int>;

    template <class _container_type>
    inline float get_raw_score(const _container_type &x, const _container_type &y)
    {
      auto xi = x.begin(), yi = y.begin();
      auto xe = x.end(), ye = y.end();
      unsigned int bag1 = 0;
      unsigned int bag2 = 0;

      while (xi != xe && yi != ye)
      {
        if (xi->first == yi->first)
        {
          if (xi->second > yi->second)
          {
            bag1 += xi->second - yi->second;
          }
          else
          {
            bag2 += yi->second - xi->second;
          }
          xi++;
          yi++;
        }
        else if (xi->first < yi->first)
        {
          bag1 += xi->second;
          xi++;
        }
        else
        {
          bag2 += yi->second;
          yi++;
        }
      }

      for (; xi != xe; ++xi)
      {
        bag1 += xi->second;
      }

      for (; yi != ye; ++yi)
      {
        bag2 += yi->second;
      }

      return std::max(bag1, bag2);
    }

    template <class _container_type>
    inline float get_sim_score(const _container_type &x, const _container_type &y)
    {
      auto xi = x.begin(), yi = y.begin();
      auto xe = x.end(), ye = y.end();
      unsigned int bag1 = 0;
      unsigned int bag2 = 0;
      unsigned int xl = 0;
      unsigned int yl = 0;

      while (xi != xe && yi != ye)
      {
        if (xi->first == yi->first)
        {
          if (xi->second > yi->second)
          {
            bag1 += xi->second - yi->second;
          }
          else
          {
            bag2 += yi->second - xi->second;
          }
          xl += xi->second;
          yl += yi->second;
          xi++;
          yi++;
        }
        else if (xi->first < yi->first)
        {
          bag1 += xi->second;
          xl += xi->second;
          xi++;
        }
        else
        {
          bag2 += yi->second;
          yl += yi->second;
          yi++;
        }
      }

      for (; xi != xe; ++xi)
      {
        bag1 += xi->second;
        xl += xi->second;
      }

      for (; yi != ye; ++yi)
      {
        bag2 += yi->second;
        yl += yi->second;
      }

      if (xl == 0 || yl == 0)
      {
        return 1.0f;
      }

      return 1.0f - 1.0f * std::max(bag1, bag2) / std::max(xl, yl);
    }
  };
}

#endif