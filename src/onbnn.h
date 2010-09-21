#ifndef _ONBNN_INCLUDED_
#define _ONBNN_INCLUDED_

#include <vector>

namespace onbnn
{

  /*
   * An Object is basically a point cloud, or multiple point clouds. 
   * Our oNBNN classifier classifies instances of the Object class. 
   * Thus, Object instances can contain multiple channels. 
   * In practice, this is how you use the Object class:
   * 
   * onbnn::Object i;
   * for all channels c:
   *  forall image features x from channel c:
   *    dist_n = nearest neighbor distance of x to the negative class
   *    dist_p = nearest neighbor distance of x to the positive class
   *    i.add_point(dist_n, dist_p, c)
   * onbnn::BinaryClassifier c;
   * c.predict(i);
   *
   */
  class Object
  {
    public:
      Object();
      Object(const Object& src);
      Object(float label);
      Object& operator=(const Object& src);

      int add_channel();
      int add_point(float dist_n, float dist_p, int channel = 0);

      void set_label(float label);
      int get_num_channels() const;
      float get_label() const;
      int get_cardinal(int channel = 0) const;
      float get_dist_n(int channel = 0) const;
      float get_dist_p(int channel = 0) const;

    private:
      std::vector<int> _cardinal;
      std::vector<float> _dist_n, _dist_p;
      float _label;
  };

  /* 
   * Single channel, binary classifier
   */
  class BinaryClassifier
  {
    public:
      /*
       * Default constructor.
       */
      BinaryClassifier();

      /*
       * Copy constructor.
       */
      BinaryClassifier(const BinaryClassifier& src);
  
      /*
       * Default constructors with n >= 1 channels. 
       * This produces a simple NBNN classifier (i.e: non-optimal).
       */
      BinaryClassifier(int num_channels);

      /*
       * Destructor.
       */
      ~BinaryClassifier();

      /*
       * Equality operator.
       */
      BinaryClassifier& operator=(const BinaryClassifier& src);

      /*
       * Set up a certain number of default NBNN channels.
       */
      void set_default(int num_channels);

      /*
       * Get attributes
       */
      inline float get_alpha_n(int channel = 0) const { return this->_alpha_n[channel]; }
      inline float get_alpha_p(int channel = 0) const { return this->_alpha_p[channel]; }
      inline float get_beta(int channel = 0) const { return this->_beta[channel]; }
      inline int get_num_channels() const { return this->_num_channels; }

      /*
       * Add a training object.
       */
      void add_data(const Object& x);

      /* 
       * Training function.
       * Returns the energy minimum found.
       */
      float train();

      /*
       * Prediction function.
       */
      float predict(const Object& x);

    private:
      std::vector<Object> _data;
      std::vector<float> _alpha_n, _alpha_p, _beta;
      int _num_channels;
  };

  std::ostream& operator<<(std::ostream& s, const BinaryClassifier& c);

}

#endif

