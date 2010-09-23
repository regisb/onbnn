using namespace std;
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

#include <glpk.h>

#include "onbnn.h"
#include "aux_functions.h"

namespace onbnn
{
  /************************ Object class ************************/
  Object::Object(){ /* void implementation */ }
  Object::Object(float label){ this->_label = label; }
  Object::Object(const Object& src){ *this = src; }
  Object& Object::operator=(const Object& src){
    this->_cardinal = src._cardinal;
    this->_dist_n = src._dist_n;
    this->_dist_p = src._dist_p;
    this->_label = src._label;
    return *this;
  }
 
  int Object::add_channel(){
    this->_cardinal.push_back(0);
    this->_dist_n.push_back(0);
    this->_dist_p.push_back(0);
    return this->_cardinal.size();
  }
  
  int Object::add_point(float dist_n, float dist_p, int channel /*= 0*/){
    // Check channel exists
    while(channel >= this->get_num_channels())
      this->add_channel();

    // Add point
    this->_cardinal[channel]++;
    
    // Add distance
    this->_dist_n[channel] += dist_n;
    this->_dist_p[channel] += dist_p;
    
    // Return the number of points in that channel
    return this->_cardinal[channel];
  }
    
  void Object::set_label(float label){ this->_label = label; }
  float Object::get_label() const { return this->_label; }
  int Object::get_num_channels() const { return this->_cardinal.size(); }
  int Object::get_cardinal(int channel /*= 0*/) const { return this->_cardinal[channel]; }
  float Object::get_dist_n(int channel /*= 0*/) const { return this->_dist_n[channel]; }
  float Object::get_dist_p(int channel /*= 0*/) const { return this->_dist_p[channel]; }


 /************************ Constructors/Destructors ************************/

  BinaryClassifier::BinaryClassifier(){
    this->_num_channels = 0;
  }
  BinaryClassifier::BinaryClassifier(int num_channels){
    this->set_default(num_channels);
  }
  BinaryClassifier::BinaryClassifier(const BinaryClassifier& src){
    *this = src;
  }
  BinaryClassifier::~BinaryClassifier(){ /* empty function */ }
  BinaryClassifier& BinaryClassifier::operator=(const BinaryClassifier& src){
    this->_alpha_n  = src._alpha_n;
    this->_alpha_p  = src._alpha_p;
    this->_beta     = src._beta;  
    this->_data     = src._data;
    this->_num_channels = src._num_channels;
    return *this;
  }
  void BinaryClassifier::set_default(int num_channels)
  {
    this->_alpha_n.resize(num_channels, 1);
    this->_alpha_p.resize(num_channels, 1);
    this->_beta.resize(num_channels, 0);
    this->_num_channels = num_channels;
  }
  
  /************************ << operator ************************/
  std::ostream& operator<<(std::ostream& s, const BinaryClassifier& classif){
    for(int c = 0; c < classif.get_num_channels(); c++)
    {
      s << "Channel " << setw(3) << c;
      s << "  alpha-: " << setw(6) << classif.get_alpha_n(c);
      s << ", alpha+: " << setw(6) << classif.get_alpha_p(c);
      s << ", beta: " << setw(6) << classif.get_beta(c);
    }
    return s;
  }

  /************************ Training ************************/
  void BinaryClassifier::add_data(const Object& x){
    // Add a training object
    this->_data.push_back(x);

    // Check if it changes the number of channels of the classifier.
    // If yes, add a certain number of default channels.
    if(x.get_num_channels() > this->get_num_channels())
      this->set_default(x.get_num_channels());
  }

  float BinaryClassifier::train()
  {
     int num_objects  = this->_data.size();
     int num_channels = this->get_num_channels();
     
     // Comment this to make the optimisation verbose
     glp_term_hook(onbnn::glp_hook, NULL );
     
     // Create linear minimisation problem
     glp_prob* lp = glp_create_prob();
     glp_set_obj_dir(lp, GLP_MIN);

     // Here is the order of the variables:
     // 1    -> c      alpha_n (channels 0 -> c-1)
     // c+1  -> 2c     alpha_p (channels 0 -> c-1)
     // 2c+1 -> 3c     beta (channels 0 -> c-1)
     // 3c+1 -> 3c + i chsi (objects 0 -> i-1)

     // Equation coefficients (will be converted to arrays later on)
     vector<int> ia_vec(1,0), ja_vec(1,0);
     vector<double> ar_vec(1,0);

     // forall i, label(i)*[sum_c (dist_n_c(i)*alpha_n_c - dist_p_c(i)*alpha_p_c + beta_c) ] + chsi(i) >= 1
     glp_add_rows( lp, num_objects);
     vector<Object>::const_iterator data = this->_data.begin();
     for(int i = 0; i < num_objects; i++, data++)
     {
        glp_set_row_bnds( lp, i+1, GLP_LO, 1.0, 0.0 );// >= 1

        // Iterate over all channels c for the object i
        for(int c = 0; c < data->get_num_channels(); c++)
        {
          // label(i)*alpha_n_c*dist_n_c(i)
          ia_vec.push_back(1 + i);
          ja_vec.push_back(1 + c);
          ar_vec.push_back(data->get_label()*data->get_dist_n(c));

          // -label(i)*alpha_p_c*dist_p_c(i)
          ia_vec.push_back(1 + i);
          ja_vec.push_back(1 + num_channels + c);
          ar_vec.push_back(-data->get_label()*data->get_dist_p(c));

          // label(i)*cardinal(i)*beta_c
          ia_vec.push_back(1 + i);
          ja_vec.push_back(1 + 2*num_channels + c);
          ar_vec.push_back(data->get_label()*data->get_cardinal(c));
        }

        // chsi(i)
        ia_vec.push_back(i + 1);
        ja_vec.push_back(1 + 3*num_channels + i);
        ar_vec.push_back(1);
     }

     // Constraints on the variables
     glp_add_cols( lp, num_channels*3 + num_objects);
     for(int c = 0; c < num_channels; c++)
     {
        glp_set_col_bnds( lp, 1 + c,                  GLP_LO, 0.0, 0.0 ); // alpha_n_c >= 0
        glp_set_col_bnds( lp, 1 + num_channels   + c, GLP_LO, 0.0, 0.0 ); // alpha_p >= 0
        glp_set_col_bnds( lp, 1 + num_channels*2 + c, GLP_FR, 0.0, 0.0 ); // no constraint on beta
     }
     for( int i = 0; i < num_objects; i++ )
       glp_set_col_bnds( lp, 1 + num_channels*3 + i,  GLP_LO, 0.0, 0.0 ); // chsi_i >= 0


      // Objective value: sum_i chsi(i)/num_objects// TODO check this
     for( int i = 0; i < num_objects; i++ )
       glp_set_obj_coef( lp, 1 + num_channels*3 + i, 1.0/num_objects);

     // Solve problem
     // Copy vectors to arrays
     int var_n = ia_vec.size() - 1;
     int ia[ ia_vec.size() ];    copy(ia_vec.begin(), ia_vec.end(), ia); ia_vec = vector<int>();
     int ja[ ja_vec.size() ];    copy(ja_vec.begin(), ja_vec.end(), ja); ja_vec = vector<int>();
     double ar[ ar_vec.size() ]; copy(ar_vec.begin(), ar_vec.end(), ar); ar_vec = vector<double>();

     // Load problem
     glp_load_matrix( lp, var_n, ia, ja, ar );

     // Solve problem
     lpx_simplex(lp);

     // Get density correction parameter values
     for(int c = 0; c < num_channels; c++)
     {
        this->_alpha_n[c]  = glp_get_col_prim(lp, 1 + c);
        this->_alpha_p[c]  = glp_get_col_prim(lp, 1 + num_channels   + c);
        this->_beta[c]     = glp_get_col_prim(lp, 1 + num_channels*2 + c);
     }
     
     // Get objective value
     float energy = glp_get_obj_val(lp);

     // Clean the problem
     glp_delete_prob(lp);

     return energy;
  }

  /************************ Prediction ************************/
  float BinaryClassifier::predict(const Object& x)
  {
    int num_channels = min(this->get_num_channels(), x.get_num_channels());
    float prediction = 0;
    for(int c = 0; c < num_channels; c++)
    {
      prediction += x.get_dist_n(c)*this->get_alpha_n(c);
      prediction -= x.get_dist_p(c)*this->get_alpha_p(c);
      prediction += x.get_cardinal(c)*this->get_beta(c);
    }
    return prediction;
  }
}
