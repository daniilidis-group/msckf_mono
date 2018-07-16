#include <msckf_mono/corner_detector.h>
#include <opencv2/calib3d.hpp>
#include <set>

namespace corner_detector
{

CornerDetector::CornerDetector(int n_rows, int n_cols, double detection_threshold) :
  grid_n_rows_(n_rows), grid_n_cols_(n_cols),
  detection_threshold_(detection_threshold)
{
  occupancy_grid_.clear();
  occupancy_grid_.resize(grid_n_rows_*grid_n_cols_, false);
}

void CornerDetector::set_grid_size(int n_rows, int n_cols) {
  grid_n_rows_ = n_rows;
  grid_n_cols_ = n_cols;
  occupancy_grid_.resize(grid_n_rows_*grid_n_cols_, false);
}

int CornerDetector::sub2ind(const cv::Point2f& sub) {
  return static_cast<int>(sub.y / grid_height_)*grid_n_cols_
    + static_cast<int>(sub.x / grid_width_);
}

void CornerDetector::zero_occupancy_grid()
{
  std::fill(occupancy_grid_.begin(), occupancy_grid_.end()-1, false);
}

void CornerDetector::set_grid_position(const cv::Point2f& pos) {
  occupancy_grid_[sub2ind(pos)] = true;
}

// Function from rpg_vikit - no need to clone whole repo
// https://github.com/uzh-rpg/rpg_vikit
float
CornerDetector::shiTomasiScore(const cv::Mat& img, int u, int v)
{
  assert(img.type() == CV_8UC1);

  float dXX = 0.0;
  float dYY = 0.0;
  float dXY = 0.0;
  const int halfbox_size = 15;
  const int box_size = 2*halfbox_size;
  const int box_area = box_size*box_size;
  const int x_min = u-halfbox_size;
  const int x_max = u+halfbox_size;
  const int y_min = v-halfbox_size;
  const int y_max = v+halfbox_size;

  if(x_min < 1 || x_max >= img.cols-1 || y_min < 1 || y_max >= img.rows-1)
    return 0.0; // patch is too close to the boundary

  const int stride = img.step.p[0];
  for( int y=y_min; y<y_max; ++y )
  {
    const uint8_t* ptr_left   = img.data + stride*y + x_min - 1;
    const uint8_t* ptr_right  = img.data + stride*y + x_min + 1;
    const uint8_t* ptr_top    = img.data + stride*(y-1) + x_min;
    const uint8_t* ptr_bottom = img.data + stride*(y+1) + x_min;
    for(int x = 0; x < box_size; ++x, ++ptr_left, ++ptr_right, ++ptr_top, ++ptr_bottom)
    {
      float dx = *ptr_right - *ptr_left;
      float dy = *ptr_bottom - *ptr_top;
      dXX += dx*dx;
      dYY += dy*dy;
      dXY += dx*dy;
    }
  }

  // Find and return smaller eigenvalue:
  dXX = dXX / (2.0 * box_area);
  dYY = dYY / (2.0 * box_area);
  dXY = dXY / (2.0 * box_area);
  return 0.5 * (dXX + dYY - sqrt( (dXX + dYY) * (dXX + dYY) - 4 * (dXX * dYY - dXY * dXY) ));
}

void CornerDetector::detect_features(const cv::Mat& image, std::vector<cv::Point2f>& features)
{
  grid_height_ = (image.rows / grid_n_rows_)+1;
  grid_width_ = (image.cols / grid_n_cols_)+1;

  features.clear();
  std::vector<double> score_table(grid_n_rows_ * grid_n_cols_);
  std::vector<cv::Point2f> feature_table(grid_n_rows_ * grid_n_cols_);
  std::vector<fast::fast_xy> fast_corners;

#ifdef __SSE2__
  fast::fast_corner_detect_10_sse2(
          (fast::fast_byte*) image.data, image.cols,
          image.rows, image.cols, 20, fast_corners);
#else
  fast::fast_corner_detect_10(
          (fast::fast_byte*) image.data, image.cols,
          image.rows, image.cols, 20, fast_corners);
#endif

  std::vector<int> scores, nm_corners;
  fast::fast_corner_score_10((fast::fast_byte*) image.data, image.cols, fast_corners, 20, scores);
  fast::fast_nonmax_3x3(fast_corners, scores, nm_corners);

  // ALEX: Updated loop
  for(auto it:nm_corners)
  {
    fast::fast_xy& xy = fast_corners.at(it);
    if (xy.x >= grid_n_cols_ * grid_width_ ||
        xy.y >= grid_n_rows_ * grid_height_)
      continue;
    const int k = sub2ind(cv::Point2f(xy.x, xy.y));
    if(occupancy_grid_[k])
      continue;
    const float score = shiTomasiScore(image, xy.x, xy.y);
    if(score > score_table[k])
    {
      score_table[k] = static_cast<double>(score);
      feature_table[k] = cv::Point2f(xy.x, xy.y);
    }
  }

  // Create feature for every corner that has high enough corner score
  // ALEX: Replaced corner object from original code
  for (int i = 0; i < score_table.size(); i++ )
  {
    if(score_table[i] > detection_threshold_)
    {
      cv::Point2f pos = feature_table[i];
      features.push_back(cv::Point2f(pos.x, pos.y));
    }
  }
  zero_occupancy_grid();
}

CornerTracker::CornerTracker(int window_size,
                             double min_eigen_threshold,
                             int max_level,
                             int termcrit_max_iters,
                             double termcirt_epsilon)
  : window_size_(window_size, window_size),
    min_eigen_threshold_(min_eigen_threshold),
    max_level_(max_level),
    termination_criteria_(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, termcrit_max_iters, termcirt_epsilon)
{
}

void CornerTracker::configure(double window_size,
                              double min_eigen_threshold,
                              int max_level,
                              int termcrit_max_iters,
                              double termcirt_epsilon)
{
  window_size_ = cv::Size(window_size, window_size);
  min_eigen_threshold_ = min_eigen_threshold;
  max_level_ = max_level;
  termination_criteria_ = cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, termcrit_max_iters, termcirt_epsilon);
}

void CornerTracker::track_features(cv::Mat img_1, cv::Mat img_2, Point2fVector& points1, Point2fVector& points2, IdVector& id1, IdVector& id2){
  std::vector<uchar> status;

  std::vector<float> err;

  cv::calcOpticalFlowPyrLK(img_1, img_2,
                           points1, points2,
                           status, err,
                           window_size_, max_level_,
                           termination_criteria_,
                           cv::OPTFLOW_USE_INITIAL_FLOW, min_eigen_threshold_);

  int h = img_1.rows;
  int w = img_1.cols;

  // remove failed or out of image points
  int indexCorrection = 0;
  for(int i=0; i<status.size(); i++){
    cv::Point2f pt = points2.at(i- indexCorrection);
    cv::Point2f dist_vector = points2.at(i-indexCorrection)-points1.at(i-indexCorrection);
    double dist = std::sqrt(dist_vector.x*dist_vector.x+dist_vector.y*dist_vector.y);
    if(dist>25.0 || (status.at(i) == 0)||(pt.x<0)||(pt.y<0)||(pt.x>w)||(pt.y>h))	{
      if((pt.x<0)||(pt.y<0)||(pt.x>w)||(pt.y>h))
        status.at(i) = 0;

      points1.erase (points1.begin() + (i - indexCorrection));
      points2.erase (points2.begin() + (i - indexCorrection));

      id1.erase (id1.begin() + (i - indexCorrection));
      id2.erase (id2.begin() + (i - indexCorrection));

      indexCorrection++;
    }
  }
}

TrackHandler::TrackHandler(const cv::Mat K,
    const cv::Mat distortion_coeffs, const std::string dist_model) 
  : ransac_threshold_(0.0000002), next_feature_id_(0),
    gyro_accum_(Eigen::Vector3f::Zero()), n_gyro_readings_(0),
    K_(K), K_inv_(K.inv()), distortion_coeffs_(distortion_coeffs),
    distortion_model_(dist_model), use_gyro_(true)
{
  dR_ = cv::Mat::eye(3,3,CV_32F);
  clear_tracks();

  tracker_.configure(51, 0.00001, 4, 30, 1.);
}

TrackHandler::~TrackHandler() {}

void TrackHandler::set_grid_size(int n_rows, int n_cols) {
  detector_.set_grid_size(n_rows, n_cols);
}

void TrackHandler::add_gyro_reading(Eigen::Vector3f& gyro_reading) {
  gyro_accum_ += gyro_reading;
  n_gyro_readings_++;
}

void TrackHandler::integrate_gyro() {
  double dt = cur_time_-prev_time_;

  if(n_gyro_readings_>0){
    gyro_accum_ /= static_cast<float>(n_gyro_readings_);
  }else{
    use_gyro_ = false;
    return;
  }
  gyro_accum_ *= dt;

  cv::Mat r(3,1, CV_32F);
  r.at<float>(0) = gyro_accum_[0];
  r.at<float>(1) = gyro_accum_[1];
  r.at<float>(2) = gyro_accum_[2];
  cv::Mat dR = cv::Mat::eye(3,3,CV_32F);
  cv::Rodrigues(r, dR);

  gyro_accum_.setZero();
  n_gyro_readings_ = 0;

  dR_ = dR;

  return;
}

void TrackHandler::predict_features(){
  std::copy(prev_feature_ids_.begin(),
      prev_feature_ids_.end(),
      std::back_inserter(cur_feature_ids_));

  if (use_gyro_){
    integrate_gyro();

    // homography by rotation
    cv::Mat H = K_ * dR_ * K_inv_;
    cv::Mat pt_buf1(3,1,CV_32F);
    pt_buf1.at<float>(2) = 1.0;

    cv::Mat pt_buf2(3,1,CV_32F);

    for(auto& pt : prev_features_){
      pt_buf1.at<float>(0) = pt.x;
      pt_buf1.at<float>(1) = pt.y;

      pt_buf2 = H * pt_buf1;

      const float x = pt_buf2.at<float>(0) / pt_buf2.at<float>(2);
      const float y = pt_buf2.at<float>(1) / pt_buf2.at<float>(2);
      cur_features_.emplace_back(x,y);
    }
  }else{
    std::copy(prev_features_.begin(),
              prev_features_.end(),
              std::back_inserter(cur_features_));
  }
}

void TrackHandler::set_current_image(cv::Mat img, double time)
{
  // Move current to previous
  prev_time_ = cur_time_;
  prev_img_ = cur_img_;
  prev_features_ = cur_features_;
  prev_feature_ids_ = cur_feature_ids_;
  std::copy(new_feature_ids_.begin(),
            new_feature_ids_.end(),
            std::back_inserter(prev_feature_ids_));
  std::copy(new_features_.begin(),
            new_features_.end(),
            std::back_inserter(prev_features_));

  // enforce grid on features
  int rows = detector_.get_n_rows();
  int cols = detector_.get_n_cols();
  std::vector<bool> oc_grid(rows*cols, false);
  auto f_it=prev_features_.begin();
  auto fid_it=prev_feature_ids_.begin();
  for(;f_it!=prev_features_.end() && fid_it!=prev_feature_ids_.end();){
    int ind = detector_.sub2ind(*f_it);
    if(oc_grid[ind]){
      f_it = prev_features_.erase(f_it);
      fid_it = prev_feature_ids_.erase(fid_it);
    }else{
      oc_grid[ind] = true;
      f_it++;
      fid_it++;
    }
  }

  // Set the current
  cur_time_ = time;
  cur_img_ = img;
  cur_features_.clear();
  cur_feature_ids_.clear();

  new_features_.clear();
  new_feature_ids_.clear();
}

void TrackHandler::tracked_features(OutFeatureVector& features, IdVector& feature_ids)
{
  // sanatize inputs
  cur_features_.reserve(prev_features_.size());
  cur_feature_ids_.reserve(prev_feature_ids_.size());

  // previous features exist for optical flow to work
  // this fills features with all tracked features from the previous frame
  // also handles the transfer of ids
  size_t prev_size = prev_features_.size();
  if(prev_features_.size() != 0){
    predict_features(); // populate cur_features_

    visualizer_.add_predicted(cur_features_, cur_feature_ids_);

    tracker_.track_features(prev_img_, cur_img_,
                            prev_features_, cur_features_,
                            prev_feature_ids_, cur_feature_ids_);
  }

  visualizer_.add_current_features(cur_features_, cur_feature_ids_);

  features.clear();
  feature_ids.clear();
  if(cur_features_.size()>0){
    Point2fVector undistorted_prev_pts;
    undistortPoints(prev_features_, undistorted_prev_pts);

    Point2fVector undistorted_cur_pts;
    undistortPoints(cur_features_, undistorted_cur_pts);

    OutFeatureVector prev_features;
    prev_features.reserve(prev_features_.size());
    std::transform(undistorted_prev_pts.begin(), undistorted_prev_pts.end(), std::back_inserter(prev_features),
        [](const cv::Point2f& pt){return msckf_mono::Vector2<float>{ pt.x, pt.y };});

    OutFeatureVector cur_features;
    cur_features.reserve(cur_features_.size());
    std::transform(undistorted_cur_pts.begin(), undistorted_cur_pts.end(), std::back_inserter(cur_features),
        [](const cv::Point2f& pt){return msckf_mono::Vector2<float>{ pt.x, pt.y };});

    msckf_mono::Matrix3<float> dR;
    for(int i=0; i<3; i++)
      for(int j=0; j<3; j++)
        dR(i,j) = dR_.at<float>(i,j);

    if(cur_features.size() > 5 && false){
      auto valid_pts = twoPointRansac(dR, prev_features, cur_features);

      auto ocf_it = cur_features.begin();
      auto cf_it = cur_features_.begin();
      auto cfi_it = cur_feature_ids_.begin();

      for(int i=0; i<cur_features.size(); i++){
        if(!valid_pts[i]){
          ocf_it = cur_features.erase(ocf_it);
          cf_it = cur_features_.erase(cf_it);
          cfi_it = cur_feature_ids_.erase(cfi_it);
        }else{
          ocf_it++;
          cf_it++;
          cfi_it++;
        }
      }
    }

    features.clear();
    std::copy(cur_features.begin(), cur_features.end(),
        std::back_inserter(features));

    feature_ids.clear();
    std::copy(cur_feature_ids_.begin(), cur_feature_ids_.end(),
        std::back_inserter(feature_ids));
  }
}

void TrackHandler::new_features(OutFeatureVector& features, IdVector& feature_ids) {
  // flag all grid positions that already have a feature in it
  for(const auto& f: cur_features_){
    detector_.set_grid_position(f);
  }
  detector_.detect_features(cur_img_, new_features_);
  //std::cout << "[Detector] Found " << new_features_.size()
  //          << " new features" << std::endl;

  // generate ids for the new features
  new_feature_ids_.reserve(feature_ids.size()+new_features_.size());
  next_feature_id_++;
  for(int i=0; i<new_features_.size(); i++){
    new_feature_ids_.push_back(next_feature_id_);
    next_feature_id_++;
  }

  visualizer_.add_new_features(new_features_, new_feature_ids_);

  features.clear();
  feature_ids.clear();

  if(new_features_.size()){
    Point2fVector undistorted_pts;
    undistortPoints(new_features_, undistorted_pts);

    features.reserve(new_features_.size());
    std::transform(undistorted_pts.begin(), undistorted_pts.end(), std::back_inserter(features),
        [](const cv::Point2f& pt){return msckf_mono::Vector2<float>{ pt.x, pt.y };});

    std::copy(new_feature_ids_.begin(), new_feature_ids_.end(),
        std::back_inserter(feature_ids));
  }
}

void TrackHandler::undistortPoints(Point2fVector& in, Point2fVector& out){
  if (distortion_model_ == "radtan") {
    cv::undistortPoints(in, out, K_, distortion_coeffs_);
  } else if (distortion_model_ == "equidistant") {
    cv::fisheye::undistortPoints(in, out, K_, distortion_coeffs_);
  } else {
    cv::undistortPoints(in, out, K_, distortion_coeffs_);
  }
}
 
void TrackHandler::set_ransac_threshold(double rt){
  ransac_threshold_ = rt;
}

Eigen::Array<bool, 1, Eigen::Dynamic>
TrackHandler::twoPointRansac(const msckf_mono::Matrix3<float>& dR,
                             const OutFeatureVector& old_points_in,
                             const OutFeatureVector& new_points_in)
{
  assert(old_points_in.size() == new_points_in.size());

  int num_points = old_points_in.size();

  Eigen::Matrix<float, 3, Eigen::Dynamic> old_points(3, num_points);
  old_points.setZero();
  old_points.row(2) = Eigen::MatrixXf::Constant(1, num_points, 1);

  Eigen::Matrix<float, 3, Eigen::Dynamic> new_points(3, num_points);
  new_points.setZero();
  new_points.row(2) = Eigen::MatrixXf::Constant(1, num_points, 1);

  Eigen::Array<float, 2, 1> principal_point, focal_length;
  principal_point << K_.at<float>(0, 2), K_.at<float>(1, 2);
  focal_length << K_.at<float>(0, 0), K_.at<float>(1, 1);

  int col_iter = 0;

  for (int i=0; i < static_cast<int>(old_points_in.size()); ++i) {
    msckf_mono::Vector2<float> old_point = old_points_in[i];
    old_point.array() -= principal_point;
    old_point.array() /= focal_length;
    old_points.block(0, col_iter, 2, 1) = old_point;

    msckf_mono::Vector2<float> new_point = new_points_in[i];
    new_point.array() -= principal_point;
    new_point.array() /= focal_length;
    new_points.block(0, col_iter++, 2, 1) = new_point;
  }

  if (col_iter < 5)
  {
    return Eigen::Array<bool, Eigen::Dynamic, 1>::Constant(num_points, true);
  }

  int num_iters = 300;

  Eigen::Array<bool, 1, Eigen::Dynamic> best_inliers;
  int most_inliers = -1;

  for (int i=0; i < num_iters; i++)
  {
    // Pick two points
    int ind1 = rand() % col_iter;
    int ind2 = ind1;
    while (ind2 == ind1)
    {
      ind2 = rand() % col_iter;
    }

    // Estimate translation
    msckf_mono::Vector3<float> t;

    Eigen::Matrix<float, 2, 3> M;
    const msckf_mono::Vector3<float> c1 = new_points.col(ind1);

    const msckf_mono::Vector3<float> c2 = new_points.col(ind2);

    M <<
      (dR * old_points.col(ind1)).transpose() * msckf_mono::vectorToSkewSymmetric(c1),
      (dR * old_points.col(ind2)).transpose() * msckf_mono::vectorToSkewSymmetric(c2);

    if(!M.isZero(1e-9)){
      Eigen::FullPivLU<Eigen::Matrix<float, 2, 3>> lu_decomp(M);
      t = lu_decomp.kernel();
    }else{
      t.setZero();
    }

    if (t.cols() > 1)
    {
      printf("Kernel in RANSAC is the wrong size, returning.");
      continue;
    }

    // Compute Sampson Error
    msckf_mono::Matrix3<float> E = msckf_mono::vectorToSkewSymmetric(t) * dR;
    Eigen::Array<float, 3, Eigen::Dynamic> Ex1 = E * old_points;
    Eigen::Array<float, 3, Eigen::Dynamic> Ex2 = E.transpose() * new_points;
    Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> errs = ((new_points.array() * Ex1).colwise().sum()).square();
    errs /=
      Ex1.row(0).array().square() +
      Ex1.row(1).array().square() +
      Ex2.row(0).array().square() +
      Ex2.row(1).array().square();

    Eigen::Array<bool, 1, Eigen::Dynamic> inliers = errs < ransac_threshold_;
    int num_inliers = inliers.count();
    if (num_inliers > most_inliers)
    {
      best_inliers = inliers;
      most_inliers = num_inliers;
    }
  }

  return best_inliers;
}

void TrackHandler::clear_tracks()
{
  // clear all stateful pieces
  prev_img_ = cv::Mat();
  prev_features_.clear();
  prev_feature_ids_.clear();
}

cv::Mat TrackHandler::get_track_image()
{
  return visualizer_.draw_tracks(prev_img_);
}

TrackVisualizer::TrackVisualizer()
{
  feature_tracks_.clear();
}

void TrackVisualizer::add_predicted(Point2fVector& features, IdVector& feature_ids)
{
  assert(fetures.size()==feature_ids.size());
  predicted_pts_.clear();

  auto fit=features.begin();
  auto idit=feature_ids.begin();
  for(;fit!=features.end() && idit!=feature_ids.end();++fit, ++idit){
    predicted_pts_.emplace(*idit, *fit);
  }
}

void TrackVisualizer::add_current_features(Point2fVector& features, IdVector& feature_ids)
{
  assert(features.size()==feature_ids.size());
  std::set<size_t> current_ids;
  auto fit=features.begin();
  auto idit=feature_ids.begin();
  for(;fit!=features.end() && idit!=feature_ids.end();++fit, ++idit){
    size_t id = *idit;

    current_ids.insert(id);
    if(feature_tracks_.find(id)==feature_tracks_.end()){
      feature_tracks_.emplace(*idit, Point2fVector());
    }

    feature_tracks_[id].push_back(*fit);
  }

  std::vector<int> to_remove;
  for(auto& track : feature_tracks_)
    if(current_ids.count(track.first)==0)
      to_remove.push_back(track.first);

  for(auto id : to_remove)
    feature_tracks_.erase(id);
}

void TrackVisualizer::add_new_features(Point2fVector& features, IdVector& feature_ids)
{
  assert(features.size()==feature_ids.size());
  auto fit=features.begin();
  auto idit=feature_ids.begin();
  for(;fit!=features.end();++fit, ++idit){
    size_t id = *idit;

    if(feature_tracks_.find(id)==feature_tracks_.end()){
      feature_tracks_.insert(std::make_pair(id, Point2fVector()));
    }

    feature_tracks_[id].push_back(*fit);
  }
}

cv::Mat TrackVisualizer::draw_tracks(cv::Mat image)
{
  cv::Mat outImage;

  if( image.type() == CV_8UC3 ){
    image.copyTo( outImage );
  }else if( image.type() == CV_8UC1 ){
    cvtColor( image, outImage, cv::COLOR_GRAY2BGR );
  }else{
    CV_Error( cv::Error::StsBadArg, "Incorrect type of input image.\n" );
    return outImage;
  }

  bool first;
  Point2fVector::reverse_iterator prev;
  cv::Scalar color;
  for(auto& track : feature_tracks_)
  {
    first = true;
    size_t id = track.first;
    // create consistent color for each track
    color = cv::Scalar(((id/64)%8)*255/8, ((id/8)%8)*255/8, (id%8)*255/8);

    auto search = predicted_pts_.find(id);
    if(search!=predicted_pts_.end()){
      cv::circle(outImage, search->second, 6, color, 2);
    }

    for(auto it = track.second.rbegin(); it != track.second.rend(); ++it)
    {
      if(first){
        first = false;
        cv::circle(outImage, *it, 4, color, 2);
      }else{
        cv::line(outImage, *it, *prev, color, 1);
      }
      prev = it;
    }
  }

  return outImage;
}

} // End corner_detector namespace
