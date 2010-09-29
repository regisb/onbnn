#ifndef __LSHKIT_PROBE_CUSTOM__
#define __LSHKIT_PROBE_CUSTOM__

#include "lshkit/common.h"
#include "lshkit/lsh.h"
#include "lshkit/composite.h"
#include "lshkit/metric.h"
#include "lshkit/lsh-index.h"
#include "lshkit/mplsh-model.h"
#include "lshkit/topk.h"
#include "lshkit/archive.h"

#include "mplsh-fit-tune.h"

using namespace lshkit;

namespace mplsh_custom
{

/***************************************************************/
/*********** Data structures for vector access *****************/
/***************************************************************/

template<class T> class Data{
		
		public:
			Data(){}
		
			Data( const Data<T>& src ){
				this->_data = src._data;
			}
		
			Data( const std::vector<T>& src ){
				this->_data = src;
			}
			
			Data<T>& operator=( const Data<T>& src ){
				this->_data = src._data;
				return *this;
			}
			
			virtual T operator[]( int i ) const {
				return this->_data[i];
			}
			
			inline const std::vector<T>& data() const { return this->_data; }
      inline std::vector<T>& data() { return this->_data; }
			
			virtual inline T dist( const Data<T>& x ) const {
        // Return L2 distance (squared)
        typename std::vector<T>::const_iterator it = this->_data.begin(), itx = x._data.begin();
        T distance = 0, sub;
        for(int d = 0; d < this->_data.size(); d++, ++itx, ++it)
        {
          sub = *itx - *it;
          distance += sub*sub;
        }
        return distance;
      }
		
			int getDim() const { return this->_data.size(); }
			int dim() const { return this->_data.size(); }
			
		protected:
			std::vector<T> _data;
	};

	template<class T> class DataAccessor{
		public:
			typedef unsigned Key;
		
			DataAccessor(){}
		
			DataAccessor( const std::vector<T>& src ) : _flags( src.size() ){
				this->_data = &src;
			}
		
			void reset(){ this->_flags.reset(); }
			
			bool mark (unsigned key) {
				if ( this->_flags[key] )
					return false;
				this->_flags.set(key);
				return true;
			}
			
			const T* operator()( unsigned i ) const {
				return &( (*this->_data)[i] );
			}
			
		protected:
			const std::vector<T>* _data;
			boost::dynamic_bitset<> _flags;
	};

	class DataMetric{
		public:
			DataMetric(){};
			DataMetric( int dummy ){};
			template<class T> T operator()( const Data<T>* x, const Data<T>* y ) const {
				return x->dist(*y);
			}
			template<class T> T operator()( const Data<T>& x, const Data<T>& y ) const {
				return x.dist(y);
			}
	};

	template<class T> class ThresholdingLsh
	{
		unsigned dim_;
		float threshold_;
		public:
			struct Parameter
			{
			  // Dimension of domain.
				unsigned dim;
        // Lower bound of each dimension.
				float min;
        // Upper bound of each dimension.
				float max;
			};
			typedef const Data<T>* Domain;

			ThresholdingLsh ()
			{
			}

			template <typename RNG>
			void reset(const Parameter &param, RNG &rng)
			{
				dim_ = boost::variate_generator<RNG &, lshkit::UniformUnsigned>(rng, UniformUnsigned(0,param.dim - 1))();
				threshold_ = boost::variate_generator<RNG &, lshkit::Uniform>(rng, Uniform(param.min,param.max))();
			}

			template <typename RNG>
			ThresholdingLsh(const Parameter &param, RNG &rng)
			{
				reset(param, rng);
			}

			unsigned getRange () const
			{
				return 2;
			}

			unsigned operator () (Domain obj) const
			{
				return (*obj)[dim_] >= threshold_ ? 1 : 0;
			}

			unsigned operator () (Domain obj, float *delta) const
			{
				float ret = (*obj)[dim_] - threshold_;
				if (ret >= 0)
				{
					*delta = ret;
					return 1;
				}
				else
				{
					*delta = -ret;
					return 0;
				}
			}

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & dim_;
				ar & threshold_;
			}
	};

	template<class DATA, class DIST> class StableDistLsh
	{
		std::vector<float> a_;
		float b_;
		float W_;
		unsigned dim_;
		public:
      /*
       * Parameter to StableDistLsh.
       *
       */
			struct Parameter
			{
        // Dimension of domain.
				unsigned dim;   
        // Window size.
				float W;
			};

			typedef const DATA* Domain;

			StableDistLsh (){}

			template <typename RNG>	void reset(const Parameter &param, RNG &rng)
			{
				a_.resize(param.dim);
				W_ = param.W;
				dim_ = param.dim;

				boost::variate_generator<RNG &, DIST> gen(rng, DIST());

				for (unsigned i = 0; i < dim_; ++i) a_[i] = gen();

				b_ = boost::variate_generator<RNG &, lshkit::Uniform>(rng, lshkit::Uniform(0,W_))();
			}

			template <typename RNG> StableDistLsh(const Parameter &param, RNG &rng){
				reset(param, rng);
			}


			unsigned getRange () const{
				return 0;
			}

			unsigned operator () (Domain obj) const
			{
				float ret = b_;
				for (unsigned i = 0; i < dim_; ++i)
				{
					ret += a_[i] * (*obj)[i];
				}
				return unsigned(int(std::floor(ret / W_)));
			}

			unsigned operator () (Domain obj, float *delta) const
			{
				float ret = b_;
				for (unsigned i = 0; i < dim_; ++i)
				{
					ret += a_[i] * (*obj)[i];
				}
				ret /= W_;

				float flr =  std::floor(ret);
				*delta = ret - flr;
				return unsigned(int(flr));
			}

			template<class Archive> void serialize(Archive & ar, const unsigned int version)
			{
				ar & a_;
				ar & b_;
				ar & W_;
				ar & dim_;
				assert(a_.size() == dim_);
			}
	};


  /***************************************************/
  /***************************************************/
  /***************************************************/

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

  // Multi-Probe LSH class.
  template<class DATA> class MultiProbeLsh: public lshkit::RepeatHash< mplsh_custom::StableDistLsh<DATA, lshkit::Gaussian> >
  {
    unsigned H_;
    public:
      typedef typename lshkit::RepeatHash< mplsh_custom::StableDistLsh<DATA, lshkit::Gaussian> > Super;
      typedef typename Super::Domain Domain;

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

        template<class Archive>	void serialize(Archive & ar, const unsigned int version)
        {
          ar & this->range;
          ar & this->repeat;
          ar & this->dim;
          ar & this->W;
        }
      };

      MultiProbeLsh () {}

      template <typename RNG> void reset(const Parameter &param, RNG &rng){
        H_ = param.range;
        Super::reset(param, rng);
      }

      template <typename RNG> MultiProbeLsh(const Parameter &param, RNG &rng){
        H_ = param.range;
        Super::reset(param, rng);
      }
    
      unsigned getRange () const{ return H_; }

      unsigned operator () (Domain obj) const{ return Super::operator ()(obj) % H_;}

      template<class Archive> void serialize(Archive & ar, const unsigned int version){
        Super::serialize(ar, version);
        ar & H_;
      }

      void genProbeSequence (Domain obj, std::vector<unsigned> &seq, unsigned T){
        ProbeSequence scores;
        std::vector<unsigned> base;
        scores.resize(2 * this->lsh_.size());
        base.resize(this->lsh_.size());
        for (unsigned i = 0; i < this->lsh_.size(); ++i)
        {
          float delta;
          base[i] = Super::lsh_[i](obj, &delta);
          scores[2*i].mask = i;
          scores[2*i].reserve = 1;    // direction
          scores[2*i].score = delta;
          scores[2*i+1].mask = i;
          scores[2*i+1].reserve = unsigned(-1);
          scores[2*i+1].score = 1.0 - delta;
        }
        std::sort(scores.begin(), scores.end());

        ProbeSequence &tmpl = __probeSequenceTemplates[this->lsh_.size()];

        seq.clear();
        for (ProbeSequence::const_iterator it = tmpl.begin();
              it != tmpl.end(); ++it)
        {
          if (seq.size() == T) break;
          const Probe &probe = *it;
          unsigned hash = 0;
          for (unsigned i = 0; i < this->lsh_.size(); ++i)
          {
            unsigned h = base[scores[i].mask];
            if (probe.mask & (1 << i))
            {
              if (probe.shift & (1 << i))
              {
              h += scores[i].reserve;
              }
              else
              {
              h += unsigned(-1) * scores[i].reserve;
              }
            }
            hash += h * this->a_[scores[i].mask];
          }
          seq.push_back(hash % H_);
        }
      }
  };


  /// Multi-Probe LSH index.
  template <class DATA, class ACCESSOR, class METRIC>
  class MultiProbeLshIndex: public lshkit::LshIndex<MultiProbeLsh<DATA>, ACCESSOR, METRIC >
  {
  public:
      typedef lshkit::LshIndex<MultiProbeLsh<DATA>, ACCESSOR, METRIC > Super;
      /**
       * Super::Parameter is the same as MultiProbeLsh::Parameter
       */
      typedef typename Super::Parameter Parameter;

  private:

      Parameter param_;
      lshkit::MultiProbeLshRecallTable recall_;

  public: 
      typedef typename Super::Domain Domain;
      typedef typename Super::Key Key;

      /// Constructor.
      MultiProbeLshIndex(ACCESSOR &accessor)
          : Super(accessor, METRIC(0) /* placeholder*/) {
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
          Super::metric_ = METRIC(param.dim);
          // we are going to normalize the distance by window size, so here we pass W = 1.0.
          // We tune adaptive probing for KNN distance range [0.0001W, 20W].
          recall_.reset(lshkit::MultiProbeLshModel(Super::lshs_.size(), 1.0, param_.repeat, Probe::MAX_T), 200, 0.0001, 20.0);
      }

      /// Load the index from stream.
      void load (std::istream &ar) 
      {
          Super::load(ar);
          param_.serialize(ar, 0);
          Super::metric_ = METRIC(param_.dim);
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
      void query (const Domain &obj, lshkit::Topk<Key> *topk, unsigned T, unsigned *pcnt = (unsigned *)0)
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
        (*topk) << typename lshkit::Topk<Key>::Element(*it, Super::metric_(obj, Super::accessor_(*it)));
                      }
              }
          }
          if (pcnt != 0) *pcnt = cnt;
      }

      /// Query for K-NNs, try to achieve the given recall by adaptive probing.
      void query (const Domain &obj, lshkit::Topk<Key> *topk, float recall, unsigned *pcnt = (unsigned *)0)
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
                          (*topk) << typename lshkit::Topk<Key>::Element(*it, Super::metric_(obj, Super::accessor_(*it)));
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



  /*
   * Function to quickly build an index for mplsh_custom data with automated fit-tune. 
   */
  template<class Data, class Metric, class Index, class Accessor> 
  double build_lsh_index( const std::vector<Data>& data, const Metric& metric, int NN, int L, int T, float recall, 
                          int& default_M, float& default_W, Index& index, Accessor& dummy )
  {
    
    int pointN = data.size();
    int dim = data.front().dim();
      
    // Fit-Tune MPLSH model
    int M = -1;
    float W = -1;
    double cost = -1;
    try
    {
      
      cost = mplsh_fit_tune( data, metric, L, T, M, W, recall, NN );
      default_M = M;
      default_W = W;
    }
    catch(...)
    {
      std::cout << "Fit-Tune Crash !!" << std::endl;
      M = default_M;
      W = default_W;
    }
    
    // Build LSH index
    typename mplsh_custom::MultiProbeLshIndex<Data,Accessor,Metric>::Parameter param;
    param.W = W;
    param.repeat = M;
    param.dim = dim;
    param.range = 1017881;
    lshkit::DefaultRng rng;
    index.init( param, rng, L );
    for( unsigned i = 0; i < pointN; i++ )
      index.insert(i);
    
    return cost	;
  }

/***************************************************************/
/************************ NnIndex class ************************/
/***************************************************************/

  /* 
   * Function for computation of distance to nearest neighbor.
   */
  template<class D, class A, class M> 
  float mplsh_nn_dist(const D& query, mplsh_custom::MultiProbeLshIndex<D,A,M>& index, int T)
  {
    float dist = -1;
    int n = 1;
    while(dist < 0)
    {
      lshkit::Topk<unsigned> topk;
      topk.reset((unsigned)(n));
      index.query( &query, &topk, (unsigned)T );
      if( topk[n-1].dist < 3.4e38 )
        dist = topk[n-1].dist;
      n++;
    }
    return dist;
  }

  /*
   * Index for easy-to-use nearest neighbor search.
   */
  template<class T> class NnIndex
  {
    typedef mplsh_custom::Data<T> LshData;
    typedef mplsh_custom::DataMetric LshMetric;
    typedef mplsh_custom::DataAccessor<LshData> LshAccessor;
    typedef mplsh_custom::MultiProbeLshIndex<LshData, LshAccessor, LshMetric> LshIndex;

    public:
      NnIndex(int dim) : _dim(dim){
        if(dim < 0)
          throw "Cannot build index with negative data dimensionality.";
        this->_accessor = NULL;
        this->_index = NULL;
      }
  ;
      ~NnIndex(){
        if(this->_accessor != NULL)
        {
          delete this->_accessor;
          delete this->_index;
        }
      }

      /*
       * Build an index using some training points.
       */
      void build(const std::vector< std::vector<T> >& points)
      {
        // Check training data is not empty
        unsigned int num_points = points.size();
        unsigned int dim = this->_dim;
        if(num_points == 0)
          throw "Cannot build index without data points.";
        // Default values. Don't ask. (TODO)
        int L_param = 3, T_param = 5;
        float recall = 0.8;
        int default_M = 15;
        float default_W = 11;
        
        // Copy training data (TODO not good)
        this->_data.resize(num_points);
        for(unsigned int p = 0; p < num_points; p++)
            this->_data[p].data() = points[p];

        // Define accessor and index    
        // TODO delete accessor and index prior to that (define clear() function)
        this->_accessor = new LshAccessor(this->_data);
        this->_index = new LshIndex(*this->_accessor);

        // Build index
        build_lsh_index(this->_data, LshMetric(), 1, L_param, T_param, recall, default_M, default_W, *this->_index, *this->_accessor);
      }

      /*
       * Find the nearest neighbor of a point x and return 
       * the square L2 distance of its nearest neighbor.
       */
      T nn_dist(const std::vector<T>& x){
        // Build query TODO this is slow
        LshData query(x);

        // T parameter (don't ask) TODO
        int T_param = 5;

        // Return distance
        return mplsh_nn_dist(query, *this->_index, T_param);
      }

    private:
      std::vector<LshData> _data;  // TODO Currently, the training data is copied here. This is not really satisfying.
      LshAccessor* _accessor;
      LshIndex* _index;
      int _dim;
  };
  typedef NnIndex<float> FloatNnIndex;

}

#endif

