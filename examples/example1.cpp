#include <stdio.h>
#include <stdlib.h>
#include <cstring>

#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
using namespace std;

#include <boost/filesystem.hpp>   // includes all needed Boost.Filesystem declarations

#include <onbnn.h>

typedef vector<float> Keypoint;
typedef vector<Keypoint> KeypointSet;
typedef pair<onbnn::Object, KeypointSet> Image;

/*
 * Return the square value of any input.
 */
template<class T> T square(const T& x){ return x*x; }

/*
 * Return the square L2 distance between two vectors.
 */
template<class T> T l2_dist(const vector<T>& x, const vector<T>& y)
{
  T dist = 0;
  typename vector<T>::const_iterator itx = x.begin(), ity = y.begin();
  while(itx != x.end() && ity != y.end())
  {
    dist += square(*itx - *ity);
    itx++;
    ity++;
  }
  return dist;
}

/*
 * Returns the distance from p to its nearest neighbor in v. 
 * This is done in a very naive manner and should be accelerated.
 */
template<class T> T nn_distance(const vector<T>& p, const vector< vector<T> >& v)
{
  if(v.size() == 0)
    throw "Computing distance to nearest neighbor in empty vector";

  T nn_dist = l2_dist(p, v[0]), dist;
  for(int q = 1; q < v.size(); q++)
  {
    dist = 0;
    // The distance computation breaks if it exceeds the min distance already computed
    typename vector<T>::const_iterator itp = p.begin(), itq = v[q].begin();
    while(itp != p.end() && dist < nn_dist)
    {
      dist += square(*itp - *itq);
      itp++;
      itq++;
    }
    nn_dist = min(nn_dist, dist);
  }
  return nn_dist;
}

/*
 * Randomly split a vector in two, with no repetition. 
 * Useful for splitting data in training/testing.
 */
template<class T> void split(const vector<T>& x, vector<T>& x1, vector<T>& x2)
{
  int n = x.size(), n2 = n/2;
  int* index = new int[n];
  for(int i = 0; i < n; i++)
    index[i] = i;

  // Pick n/2 elements at random and put them in x1
  for(int p = 0; p < n2; p++)
  {
    int q = rand()%(n - p);
    x1.push_back( x[ index[q] ] );
    index[q] = index[n - 1 - p];
  }

  // Put the n - n/2 elements left in x2
  for(int p = 0; p < n - n2; p++)
    x2.push_back(x[ index[p] ]);

  delete index;
}

/*
 * List all images in the given folder.
 * Returns the number of images found.
 */
int list_images(const string& path, vector<string>& image_paths)
{
  // Convert path to boost
  boost::filesystem::path p(path.c_str());

  // Does the path exist?
  if(!boost::filesystem::exists(p))
    return 0;

  // Tolerated image extensions.
  // Add your own extensions here
  vector<string> image_extensions;
  image_extensions.push_back(".png");
  image_extensions.push_back(".jpg");
  image_extensions.push_back(".jpeg");
  image_extensions.push_back(".bmp");
  image_extensions.push_back(".tiff");

  // Iterate over all image files
  boost::filesystem::directory_iterator dir_it_end;
  int num_images = 0;
  string extension;
  for(boost::filesystem::directory_iterator dir_it(p); dir_it != dir_it_end; ++dir_it)
  {
    // Is the file an image?
    if(find(image_extensions.begin(), image_extensions.end(), dir_it->path().extension()) == image_extensions.end())
        continue;

    // Store the image name.
    // e.g: /home/image_001.jpg -> image_001
    image_paths.push_back(dir_it->path().stem());

    // Increment the number of images found
    num_images++;
  }

  // Return the total number of found images
  return num_images;
}

/*
 * Loads the points from a given channel, image located in a certain directory.
 * The point file must follow the color descriptor structure.
 * http://staff.science.uva.nl/~ksande/research/colordescriptors/readme
 */
int load_points(const string& directory, const string& img_name, const string& channel, KeypointSet& points)
{
  // Build file name
  // e.g: channel "sift" of image "image_001" from directory /home 
  // is contained in /home/image_001___sift.txt
  string basename = img_name + "___" + channel + ".txt";
  boost::filesystem::path p1(directory), p2(basename);
  boost::filesystem::path path = boost::filesystem::complete(p2, p1).string();

  // Return 0 if the file does not exist
  if(!boost::filesystem::exists(path))
    return 0;

  // Load file
  ifstream f(path.string().c_str());

  // Read file, which has the following format:
  /* **********************************************************/
  // KOEN1
  // 128
  // 569
  // <CIRCLE 50 152 4.24264 0 0.000156228>; 0 0 21 97 42 ... ;
  // ...
  /* **********************************************************/
  string line;
  
  // 1st line: KOEN1
  getline(f, line);
  
  // 2nd line: dimensionality
  getline(f, line);
  int dim = atoi(line.c_str());

  // 3rd line: number of points
  getline(f, line);
  int num_points = atoi(line.c_str());

  // Next lines: one point/line
  int num_detected_points = 0;
  char tmp[32];
  Keypoint point(dim);
  for(int p = 0; p < num_points; p++)
  {
    getline(f, line);
    istringstream ss(line.c_str());

    // Remove the first 6 elements that correspond 
    // to the spatial coordinates
    for(int s = 0; s < 6; s++)
      ss >> tmp;

    // Read attribute values
    int value;
    for(int d = 0; d < dim; d++)
      ss >> point[d];

    // Add point to keypoint set
    points.push_back(point);
    num_detected_points++;
  }

  // Close file
  f.close();
  
  // Return the number of detected points
  return num_detected_points;
}

/*
 * Load the features from all channels of an image, computes the NN-distances 
 * for each point to the negative and positive class and produces the
 * corresponding onbnn::Object instance.
 */
onbnn::Object load_object( const string& img_name, const string& dir_path, const vector<string>& channels, 
                           const vector<KeypointSet>& points_n, const vector<KeypointSet>& points_p)
{
  // Create an object
  onbnn::Object obj;

  // Load points from each channel
  int num_channels = channels.size();
  for(int c = 0; c < num_channels; c++)
  {
    KeypointSet points;
    load_points(dir_path, img_name, channels[c], points);

    // Compute NN-distance to negative and positive reference sets for each point
    for(int p = 0; p < points.size(); p++)
    {
      float dist_n = nn_distance(points[p], points_n[c]);
      float dist_p = nn_distance(points[p], points_p[c]);

      // Add point to object
      obj.add_point(dist_n, dist_p, c);
    }
  }
  // Return object
  return obj;
}


int main(int argn, char* argc[])
{
  if(argn < 4 || (argn == 2 && strcmp(argc[1], "-h") == 0))
  {
    cout << "Usage: " << argc[0];
    cout << " <path_negative_class> <path_positive_class> <channel 1> ... <channel n>" << endl;
    exit(0);
  }

  // Initialise random seed
  srand(time(NULL));// seed changes at every program start
  //srand(0);         // seed is always the same (better for testing)

  // Reading arguments
  string path_n = argc[1];// path to the negative class
  string path_p = argc[2];// path to the positive class
  vector<string> channels;// channels to load
  for(int c = 3; c < argn; c++)
    channels.push_back(argc[c]);
  int num_channels = channels.size();

  // Complete path names
  path_n = boost::filesystem::complete(boost::filesystem::path(path_n)).string();
  path_p = boost::filesystem::complete(boost::filesystem::path(path_p)).string();
  cout << "Loading negative images from: " << path_n << endl;
  cout << "Loading positive images from: " << path_p << endl;

  // Read all images from the positive and negative class directories
  vector<string> img_path_n, img_path_p;
  list_images(path_n, img_path_n);
  list_images(path_p, img_path_p);

  // Print some info
  cout << "Number of images in negative class: " << img_path_n.size() << endl;
  cout << "Number of images in positive class: " << img_path_p.size() << endl;

  // Split images from each class in training/testing
  vector<string> img_train_n, img_train_p, img_test_n, img_test_p;
  split(img_path_n, img_train_n, img_test_n);
  split(img_path_p, img_train_p, img_test_p);
  
  // Split training set in reference/validation sets
  vector<string> img_train1_n, img_train1_p, img_train2_n, img_train2_p;
  split(img_train_n, img_train1_n, img_train2_n);
  split(img_train_p, img_train1_p, img_train2_p);

  // Load reference points
  cout << "Loading reference points..." << endl;
  vector<KeypointSet> points_n(num_channels), points_p(num_channels);
  for(int i = 0; i < img_train1_n.size(); i++)
    for(int c = 0; c < channels.size(); c++)
      load_points(path_n, img_train1_n[i], channels[c], points_n[c]);
  for(int i = 0; i < img_train1_p.size(); i++)
    for(int c = 0; c < channels.size(); c++)
      load_points(path_p, img_train1_p[i], channels[c], points_p[c]);

  /**************************************************************/ 
  /************************** Training **************************/
  /**************************************************************/
  onbnn::BinaryClassifier classif, classif_nbnn(num_channels);
  // Load objects for validation set and add them to classifier
  cout << "NN-distance for parameter learning..." << endl;
  for(int i = 0; i < img_train2_n.size(); i++)
  {
    printf("  -> negative image [%d/%d]\n", i, img_train2_n.size()-1);
    onbnn::Object obj = load_object(img_train2_n[i], path_n, channels, points_n, points_p);
    obj.set_label(-1);
    classif.add_data(obj);
  }
  for(int i = 0; i < img_train2_p.size(); i++)
  {
    printf("  -> positive image [%d/%d]\n", i, img_train2_p.size()-1);
    onbnn::Object obj = load_object(img_train2_p[i], path_p, channels, points_n, points_p);
    obj.set_label(1);
    classif.add_data(obj);
  }

  cout << "Training classifier..." << endl;
  classif.train();

  cout << "Classifier trained: " << endl;
  cout << classif << endl;

  /*************************************************************/ 
  /************************** Testing **************************/
  /*************************************************************/

  // Predicting labels on testing set
  cout << "Final step: Predicting test labels..." << endl;
  vector<float> labels_true, labels_predicted_onbnn, labels_predicted_nbnn;
  for(int i = 0; i < img_test_n.size(); i++)
  {
    printf("  -> negative image [%d/%d]\n", i, img_test_n.size()-1);
    labels_true.push_back(-1);
    onbnn::Object obj = load_object(img_test_n[i], path_n, channels, points_n, points_p);
    labels_predicted_onbnn.push_back(classif.predict(obj));
    labels_predicted_nbnn.push_back(classif_nbnn.predict(obj));
  }
  for(int i = 0; i < img_test_p.size(); i++)
  {
    printf("  -> positive image [%d/%d]\n", i, img_test_p.size()-1);
    labels_true.push_back(1);
    onbnn::Object obj = load_object(img_test_p[i], path_p, channels, points_n, points_p);
    labels_predicted_onbnn.push_back(classif.predict(obj));
    labels_predicted_nbnn.push_back(classif_nbnn.predict(obj));
  }

  // Estimate quality of prediction
  int good_classif_onbnn = 0, good_classif_nbnn = 0, num_test_images = labels_true.size();
  for(int i = 0; i < labels_true.size(); i++)
  {
    if(labels_true[i]*labels_predicted_onbnn[i] > 0)
      good_classif_onbnn++;
    if(labels_true[i]*labels_predicted_nbnn[i] > 0)
      good_classif_nbnn++;
  }
  cout << "Good classification rate:" << endl;
  cout << "    -> oNBNN: " << good_classif_onbnn*100./num_test_images << "%" << endl;
  cout << "    ->  NBNN: " << good_classif_nbnn*100./num_test_images << "%" << endl;
}
