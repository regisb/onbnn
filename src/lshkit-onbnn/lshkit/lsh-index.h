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
 * \file lsh-index.h
 * \brief Flat (non-multi-probe) LSH index.
 */

#ifndef __LSHKIT_FLAT__
#define __LSHKIT_FLAT__

#include <stdexcept>
#include <algorithm>
#include "common.h"
#include "topk.h"
#include "archive.h"

namespace lshkit {

/// Flat LSH index.
/** Flat LSH index is implemented as L hash tables using mutually independent
  * LSH functions.  Given a query point q, the points in the bins to which q is
  * hashed to are scanned for the nearest neighbors of q.
  */
template <typename LSH, typename ACCESSOR, typename METRIC>
class LshIndex
{

    BOOST_CONCEPT_ASSERT((LshConcept<LSH>));
public:
    typedef typename LSH::Parameter Parameter;
    typedef typename LSH::Domain Domain;
    typedef typename ACCESSOR::Key Key;

protected:
    typedef std::vector<Key> Bin;

    std::vector<LSH> lshs_;
    std::vector<std::vector<Bin> > tables_;
    ACCESSOR accessor_;
    METRIC metric_;

public:
    /// Constructor.
    /**
      * @param accessor object accessor.
      * @param metric distance metric.
      *
      * (ACCESSOR)accessor is used as a function.  Given a key, accessor(key)
      * returns an object (or reference) of the type LSH::Domain. That object is
      * used to calculate a hash value.  The key is saved in the hash table.
      * When the associated object is needed, e.g. when scanning a bin,
      * accessor(key) is called to access the object.
      *
      * Metric metric is also used as a function.  It accepts two parameters of
      * the type LSH::Domain and returns the distance between the two.
      */
    LshIndex(ACCESSOR &accessor, const METRIC metric)
        : accessor_(accessor), metric_(metric) {
    }

    /// Initialize the hash tables.
    /**
      * @param param parameter of LSH function.
      * @param engine random number generator.
      * @param L number of hash table maintained.
      *
      */
    template <typename Engine>
    void init (const Parameter &param, Engine &engine, unsigned L)
    {
        verify(lshs_.size() == 0);
        verify(tables_.size() == 0);
        lshs_.resize(L);
        tables_.resize(L);

        for (unsigned i = 0; i < L; ++i)
        {
            lshs_[i].reset(param, engine);
            if (lshs_[i].getRange() == 0) {
                throw std::logic_error("LSH with unlimited range should not be used to construct an LSH index.  Use lshkit::Tail<> to wrapp the LSH.");
            }
            tables_[i].resize(lshs_[i].getRange());
        }
    }

    /// Constructor
    /** Load the LSH index from a stream. */
    void load (std::istream &ar)
    {
        unsigned L;
        ar & L;
        lshs_.resize(L);
        tables_.resize(L);
        for (unsigned i = 0; i < L; ++i) {
            lshs_[i].serialize(ar, 0);
            unsigned l;
            ar & l;
            std::vector<Bin> &table = tables_[i];
            table.resize(l);
            for (;;) {
                unsigned idx, ll;
                ar & idx;
                ar & ll;
                if (ll == 0) break;
                table[idx].resize(ll);
                ar.read((char *)&table[idx][0], ll * sizeof(Key));
            }
        }
    }

    /// Save the LSH index to a stream.
    void save (std::ostream &ar)
    {
        unsigned L;
        L = lshs_.size();
        ar & L;
        for (unsigned i = 0; i < L; ++i) {
            lshs_[i].serialize(ar, 0);
            std::vector<Bin> &table = tables_[i];
            unsigned l = table.size();
            ar & l;
            unsigned idx, ll;
            for (unsigned j = 0; j < l; ++j) {
                if (table[j].empty()) continue;
                idx = j;
                ll = table[j].size();
                ar & idx;
                ar & ll;
                ar.write((char *)&table[j][0], ll * sizeof(Key));
            }
            idx = ll = 0;
            ar & idx;
            ar & ll;
        }
    }

    /// Insert an item to the index.
    /**
      * @param key the key to the item.
      *
      * The inserted object is not explicitly given, but is obtained by
      * accessor(key).
      */
    void insert (Key key)
    {
        for (unsigned i = 0; i < lshs_.size(); ++i)
        {
            unsigned index = lshs_[i](accessor_(key));
            tables_[i][index].push_back(key);
        }
    }

    /// Query for K-NNs.
    /**
      * @param obj the query object.
      * @param topk the returned values.  Should be initialized to the required
      * size.
      * @param pcnt if not 0, the number of scanned items will be stored in it.
      */
    void query (const Domain &obj, Topk<Key> *topk, unsigned *pcnt = (unsigned *)0)
    {
//        if (L == 0) L = lshs_.size();
 //       assert(L <= lshs_.size());
        accessor_.reset();
        unsigned L = lshs_.size();
        unsigned cnt = 0;
        for (unsigned i = 0; i < L; ++i)
        {
            unsigned index = lshs_[i](obj);
            Bin &bin = tables_[i][index];
            for (typename Bin::const_iterator it = bin.begin();
                    it != bin.end(); ++it)
            {
                ++cnt;
                (*topk) << typename Topk<Key>::Element(*it, metric_(obj, accessor_(*it)));
            }
        }
        if (pcnt != 0) *pcnt = cnt;
    }
};


}

#endif

