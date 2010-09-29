/* 
    Copyright (C) 2008 Wei Dong <wdong@princeton.edu>. All Rights Reserved.
  
    This file is part of LSHKIT.
  
    LSHKIT is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    LSHKIT is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with LSHKIT.  If not, see <http://www.gnu.org/licenses/>.
*/

/**
 * \file topk.h
 * \brief Top-K data structures.
 */

#ifndef __LSHKIT_TOPK__
#define __LSHKIT_TOPK__

#include <vector>
#include <limits>
#include <algorithm>
#include <fstream>

/**
 * \file topk.h
 * \brief Top-K data structure for K-NN search.
 *
 * Usage:
 * \code
 *
 * Topk<unsigned> knn(100); // search for 100-NNs.
 * knn.reset();
 *
 * BOOST_FOREACH(data_point, DATABASE) {
 *      Topk<unsigned>::Element e;
 *      e.key = key of data_point
 *      e.dist = distance(query, data_point);
 *      knn << e;
 * }
 * 
 * for (unsigned i = 0; i < knn.size(); ++i) {
 *      cout << knn[i].key << ':' << knn[i].dist << endl;
 * }
 *
 * \endcode
 */

namespace lshkit {

/// Top-K entry.
/**
  * The entry stored in the top-k data structure.  The class Topk is implemented
  * as a heap of TopkEntry.
  */
template <typename KEY>
struct TopkEntry
{
    KEY key;
    float dist;   
    bool match (const TopkEntry &e) const { return key == e.key; }
    bool match (KEY e) const { return key == e; }

    TopkEntry (KEY key_, float dist_) : key(key_), dist(dist_) {}
    TopkEntry () : dist(std::numeric_limits<float>::max()) { }
    void reset () { dist = std::numeric_limits<float>::max(); }

    friend bool operator < (const TopkEntry &e1, const TopkEntry &e2)
    {
        return e1.dist < e2.dist;
    }
};

/// Top-K heap.
/**
  * Following is an example of using the Topk class:
  *
  * Topk<Key> topk;
  * topk.reset(k);
  * 
  * for each candidate key {
  *     topk << key;
  * }
  *
  * At this point topk should contain the best k keys.
  */
template <class KEY>
class Topk: public std::vector<TopkEntry<KEY> >
{
    unsigned K;
    float R;
    float th;
public:
    typedef TopkEntry<KEY> Element;
    typedef typename std::vector<TopkEntry<KEY> > Base;

    Topk () {}

    ~Topk () {}

    /// Reset the heap.
    void reset (unsigned k, float r = std::numeric_limits<float>::max()) {
        if (k == 0) throw std::invalid_argument("K MUST BE POSITIVE");
        R = th = r;
        K = k;
        this->resize(k);
        for (typename Base::iterator it = this->begin(); it != this->end(); ++it) it->reset();
    }

    void reset (unsigned k, KEY key, float r = std::numeric_limits<float>::max()) {
        if (k == 0) throw std::invalid_argument("K MUST BE POSITIVE");
        R = th = r;
        K = k;
        this->resize(k); for (typename
            Base::iterator it = this->begin(); it != this->end(); ++it) {
        it->reset(); it->key = key; }
    }

    void reset (float r) {
        K = 0;
        R = th = r;
        this->clear();
    }

    float threshold () const {
        return th;
    }

    /// Insert a new element, update the heap.
    Topk &operator << (Element t)
    {
        if (!(t.dist < th)) return *this;
        if (K == 0) { // R-NN
            this->push_back(t);
            return *this;
        }
        // K-NN
        unsigned i = this->size() - 1;
        unsigned j;
        for (;;)
        {
            if (i == 0) break;
            j = i - 1;
            if (this->at(j).match(t)) return *this;
            if (this->at(j) < t) break;
            i = j;
        }
        /* i is the place to insert to */

        j = this->size() - 1;
        for (;;)
        {
            if (j == i) break;
            this->at(j) = this->at(j-1);
            --j;
        }
        this->at(i) = t;
        th = this->back().dist;
        return *this;
    }

    /// Calculate recall.
    /** Recall = size(this /\ topk) / size(this). */
    float recall (const Topk<KEY> &topk /* to be evaluated */) const
    {
        unsigned matched = 0;
        for (typename Base::const_iterator ii = this->begin(); ii != this->end(); ++ii)
        {
            for (typename Base::const_iterator jj = topk.begin(); jj != topk.end(); ++jj)
            {
                if (ii->match(*jj))
                {
                    matched++;
                    break;
                }
            }
        }
        return float(matched)/float(this->size());
    }

    unsigned getK () const {
        return K;
    }
};

}

#endif

