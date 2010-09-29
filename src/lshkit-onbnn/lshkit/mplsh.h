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
  * \file mplsh.h
  * \brief Multi-Probe LSH indexing.
  *
  * Multi-Probe LSH (MPLSH) uses the same data structure as LshIndex, except that it
  * probes more than one buckets in each hash table to generate more accurate
  * results.  Equivalently, less hash tables are needed to achieve the same
  * accuracy.  The limitation is that the current implementation only works for
  * L2 distance.
  * 
  * Follow the following 4 steps to use the MPLSH API.
  * 
  * \section mplsh-1 1. Implement an Accessor class which maps from keys to feature vectors.
  * 
  * The MPLSH data structure doesn't manage the feature vectors, but only keeps
  * the keys to retrieve them.  You need to provide an Accessor class to MPLSH
  * so that it can retrieve the corresponding feature vector of a key.
  *
  * If you use lshkit::Matrix<> to manage the feature vectors, then simply use
  * lshkit::Matrix<>::Accessor.
  * 
  * \code
  * class Accessor
  * {
  *      // The key type for MPLSH to use.  Key should be copyable.
  *      typedef ... Key; 
  *
  *      //Mark that key has been accessed.  If key has already been marked, return false,
  *      //otherwise return true.  MPLSH will use this to avoid scanning the data more than
  *      //one time per query.
  *      bool mark (unsigned key);
  *
  *      //To clear all the marks.  Reset() is invoked when every query begins.
  *      void reset (); 
  *   
  *      //Given a key, operator () return the pointer to a feature vector.
  *      const float *operator () (unsigned key)
  * };
  * \endcode
  *
  * \section mplsh-2 2. Construct the MPLSH data structure.
  *
  * Assume we have the class ACCESSOR defined.
  *
  * \code
  *
  * typedef MultiProbeLshIndex<ACCESSOR> Index;
  *
  * ACCESSOR accessor (...);  //you need to define and construct the ACCESSOR cluster yourself.
  *
  * Index index(accessor);
  * \endcode
  *
  * The index can not be used yet.
  *
  * \section mplsh-3 3. Populate the index / Load the index from a previously saved file.
  *
  * When the index is initially built, use the following to populate the index:
  * \code
  * Index::Parameter param;
  *
  * //Setup the parameters.  Note that L is not provided here.
  * param.W = W;
  * param.H = H; // See H in the program parameters.  You can just use the default value.
  * param.M = M;
  * param.dim = DIMENSION_OF_THE_DATA
  * DefaultRng rng; // random number generator.
  * 
  * index.init(param, rng, L);
  * 
  * for (each possible key) {
  *     index.insert(key);
  * }
  * 
  * // You can now save the index for future use.
  * ofstream os(index_file.c_str(), std::ios::binary);
  * index.save(os);
  * \endcode
  *  
  * Or you can load from a previously saved file
  *  
  * \code
  * ifstream is(index_file.c_str(), std::ios::binary);
  * index.load(is);
  * \endcode
  * 
  * \section mplsh-4 4. Query the MPLSH. 
  * The K-NNs are stored in the Topk<> class.  Topk<Key> is a vector of <key,
  * distance> pairs and the the MPLSH index will have the vector sorted in
  * ascending order of distance.
  *  
  * \code
  *   
  * Topk<ACCESSOR::Key> topk;
  * float *query;
  * unsigned cnt;
  * ...
  * topk.reset(K);
  * index.query(query, &topk, T, &cnt);   cnt is the number of points actually scanned.
  * 
  * \endcode
  *
  * See the source file lshkit/tools/mplsh-run.cpp for a full example of using MPLSH.
  *
  * For adaptive probing, I hard coded the sensitive range of KNN distance to
  * [0.0001W, 100W] and logarithmically quantized the range to 200 levels.
  * If you find that your KNN distances fall outside this range, or want more refined
  * quantization, you'll have to modify the code in lshkit::MultiProbeLshIndex::init().
  *
  * \section ref Reference
  *
  * Wei Dong, Zhe Wang, William Josephson, Moses Charikar, Kai Li. Modeling LSH
  * for Performance Tuning.. To appear in In Proceedings of ACM 17th Conference
  * on Information and Knowledge Management (CIKM). Napa Valley, CA, USA.
  * October 2008.
  *
  * Qin Lv, William Josephson, Zhe Wang, Moses Charikar, Kai Li. Multi-Probe LSH:
  * Efficient Indexing for High-Dimensional Similarity Search. Proceedings of the
  * 33rd International Conference on Very Large Data Bases (VLDB). Vienna,
  * Austria. September 2007.
  *
  */

#ifndef __LSHKIT_PROBE__
#define __LSHKIT_PROBE__

#include "common.h"
#include "lsh.h"
#include "composite.h"
#include "metric.h"
#include "lsh-index.h"
#include "mplsh-model.h"
#include "topk.h"

namespace lshkit
{

/// Probe vector.
struct Probe
{
    unsigned mask;
    unsigned shift;
    float score;
    unsigned reserve;
    bool operator < (const Probe &p) const { return score < p.score; }
    Probe operator + (const Probe &m) const
    {
        Probe ret;
        ret.mask = mask | m.mask;
        ret.shift = shift | m.shift;
        ret.score = score + m.score;
        return ret;
    }
    bool conflict (const Probe &m)
    {
        return (mask & m.mask) != 0;
    }
    static const unsigned MAX_M = 20;
    static const unsigned MAX_T = 200;
}; 

/// Probe sequence.
typedef std::vector<Probe> ProbeSequence;

/// Generate a template probe sequence.
void GenProbeSequenceTemplate (ProbeSequence &seq, unsigned M, unsigned T);

/// Probe sequence template.
class ProbeSequenceTemplates: public std::vector<ProbeSequence>
{
public:
    ProbeSequenceTemplates(unsigned max_M, unsigned max_T)
        : std::vector<ProbeSequence>(max_M + 1)
    {
        for (unsigned i = 1; i <= max_M; ++i)
        {
            GenProbeSequenceTemplate(at(i), i, max_T);
        }
    }
};

extern ProbeSequenceTemplates __probeSequenceTemplates;

/// Multi-Probe LSH class.
class MultiProbeLsh: public RepeatHash<GaussianLsh> 
{
    unsigned H_;
public:
    typedef RepeatHash<GaussianLsh> Super;
    typedef Super::Domain Domain;

    /**
     * Parameter to MPLSH. 
     *
     * The following parameters are inherited from the ancestors
     * \code
     *   unsigned repeat; // the same as M in the paper
     *   unsigned dim;
     *   float W;
     * \endcode
     */
    struct Parameter : public Super::Parameter {

        unsigned range;

        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & range;
            ar & repeat;
            ar & dim;
            ar & W;
        }
    };

    MultiProbeLsh () {}

    template <typename RNG>
    void reset(const Parameter &param, RNG &rng)
    {
        H_ = param.range;
        Super::reset(param, rng);
    }

    template <typename RNG>
    MultiProbeLsh(const Parameter &param, RNG &rng)
    {
        H_ = param.range;
        Super::reset(param, rng);
    }

    unsigned getRange () const
    {
        return H_;
    }

    unsigned operator () (Domain obj) const
    {
        return Super::operator ()(obj) % H_;
    }

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        Super::serialize(ar, version);
        ar & H_;
    }

    void genProbeSequence (Domain obj, std::vector<unsigned> &seq, unsigned T);
};


/// Multi-Probe LSH index.
template <typename ACCESSOR>
class MultiProbeLshIndex: public LshIndex<MultiProbeLsh, ACCESSOR, metric::l2<float> >
{
public:
    typedef LshIndex<MultiProbeLsh, ACCESSOR, metric::l2<float> > Super;
    /**
     * Super::Parameter is the same as MultiProbeLsh::Parameter
     */
    typedef typename Super::Parameter Parameter;

private:

    Parameter param_;
    MultiProbeLshRecallTable recall_;

public: 
    typedef typename Super::Domain Domain;
    typedef typename Super::Key Key;

    /// Constructor.
    MultiProbeLshIndex(ACCESSOR &accessor)
        : Super(accessor, metric::l2<float>(0) /* placeholder*/) {
    } 

    /// Initialize MPLSH.
    /**
      * @param param parameters.
      * @param engine random number generator (if you are not sure about what to
      * use, then pass DefaultRng.
      * @param accessor object accessor (same as in LshIndex).
      * @param L number of hash tables maintained.
      */
    template <typename Engine>
    void init (const Parameter &param, Engine &engine, unsigned L) {
        Super::init(param, engine, L);
        param_ = param;
        Super::metric_ = metric::l2<float>(param.dim);
        // we are going to normalize the distance by window size, so here we pass W = 1.0.
        // We tune adaptive probing for KNN distance range [0.0001W, 20W].
        recall_.reset(MultiProbeLshModel(Super::lshs_.size(), 1.0, param_.repeat, Probe::MAX_T), 200, 0.0001, 20.0);
    }

    /// Load the index from stream.
    void load (std::istream &ar) 
    {
        Super::load(ar);
        param_.serialize(ar, 0);
        Super::metric_ = metric::l2<float>(param_.dim);
        recall_.load(ar);
        verify(ar);
    }

    /// Save to the index to stream.
    void save (std::ostream &ar)
    {
        Super::save(ar);
        param_.serialize(ar, 0);
        recall_.save(ar);
        verify(ar);
    }

    /// Query for K-NNs.
    /**
      * @param obj the query object.
      * @param topk the returned values.  Should be initialized to the required
      * size.
      * @param pcnt if not 0, the number of scanned items will be stored in it.
      */
    void query (const Domain &obj, Topk<Key> *topk, unsigned T, unsigned *pcnt = (unsigned *)0)
    {
        std::vector<unsigned> seq;
        unsigned L = Super::lshs_.size();
        unsigned cnt = 0;
        Super::accessor_.reset();
        for (unsigned i = 0; i < L; ++i)
        {
            Super::lshs_[i].genProbeSequence(obj, seq, T);
            for (unsigned j = 0; j < seq.size(); ++j)
            {
                typename Super::Bin &bin = Super::tables_[i][seq[j]];
                for (typename Super::Bin::const_iterator it = bin.begin();
                    it != bin.end(); ++it)
                    if (Super::accessor_.mark(*it))
                    {
                        ++cnt;
                        (*topk) << typename Topk<Key>::Element(*it, Super::metric_(obj,
                                    Super::accessor_(*it)));
                    }
            }
        }
        if (pcnt != 0) *pcnt = cnt;
    }

    /// Query for K-NNs, try to achieve the given recall by adaptive probing.
    void query (const Domain &obj, Topk<Key> *topk, float recall, unsigned *pcnt = (unsigned *)0)
    {
        if (topk->getK() == 0) throw std::logic_error("CANNOT ACCEPT R-NN QUERY");
        if (topk->getK() != topk->size()) throw std::logic_error("TOPK SIZE != K");
        unsigned L = Super::lshs_.size();
        std::vector<std::vector<unsigned> > seqs(L);
        for (unsigned i = 0; i < L; ++i) Super::lshs_[i].genProbeSequence(obj,
                seqs[i], Probe::MAX_T);

        unsigned cnt = 0;
        Super::accessor_.reset();
        for (unsigned j = 0; j < Probe::MAX_T; ++j)
        {
            if (j >= seqs[0].size()) break;
            for (unsigned i = 0; i < L; ++i)
            {
                typename Super::Bin &bin = Super::tables_[i][seqs[i][j]];
                for (typename Super::Bin::const_iterator it = bin.begin();
                    it != bin.end(); ++it)
                    if (Super::accessor_.mark(*it))
                    {
                        ++cnt;
                        (*topk) << typename Topk<Key>::Element(*it, Super::metric_(obj,
                                    Super::accessor_(*it)));
                    }
            }
            float r = 0.0;
            for (unsigned i = 0; i < topk->size(); ++i)
            {
                r += recall_.lookup(topk->at(i).dist / param_.W, j + 1);
            }
            r /= topk->size();
            if (r >= recall) break;
        }
        if (pcnt != 0) *pcnt = cnt;
    }
};

}


#endif

