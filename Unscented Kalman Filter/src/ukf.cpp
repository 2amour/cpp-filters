#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.9;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.7;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;
  n_sig_ =  2 * n_aug_ + 1;

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // initialize the current NIS for radar
  NIS_radar_ = 0.0;

  // initialize the current NIS for laser
  NIS_laser_ = 0.0;

  //Initial state covariance matrix P
  P_ << MatrixXd::Identity(n_x_,n_x_);

  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, n_sig_);

  // initialize weight for weights
  weights_ = VectorXd(n_sig_);

  double weight_0 = lambda_/(lambda_+n_aug_);
  weights_(0) = weight_0;
  for (int i=1; i<n_sig_; i++) {  //2n+1 weights_
      double weight = 0.5/(n_aug_+lambda_);
      weights_(i) = weight;
  }

  //set measurement dimension, laser
  laser_n_z = 2;
  R_laser_ = MatrixXd(laser_n_z, laser_n_z);
  R_laser_ << std_laspx_*std_laspx_, 0,
          0, std_laspy_*std_laspy_;

  //set measurement dimension, radar
  radar_n_z = 3;
  R_radar_ = MatrixXd(radar_n_z, radar_n_z);
  R_radar_ << std_radr_*std_radr_, 0, 0,
          0, std_radphi_*std_radphi_, 0,
          0, 0, std_radrd_*std_radrd_;

  is_initialized_ = false;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      x_ << meas_package.raw_measurements_[0]*cos(meas_package.raw_measurements_[1]),
            meas_package.raw_measurements_[0]*sin(meas_package.raw_measurements_[1]),
            0, 0, 0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      x_ << meas_package.raw_measurements_(0),
            meas_package.raw_measurements_(1),
            0, 0, 0;

    }
    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  
  //compute the time elapsed between the current and previous measurements
  double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0; //delta_t - expressed in seconds
  time_us_ = meas_package.timestamp_;

  Prediction(delta_t);

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_ ) {
    UpdateRadar(meas_package);
  } else if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
    UpdateLidar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {

  //create augmented mean vector
  VectorXd x_aug_ = VectorXd(7);

  //create augmented state covariance
  MatrixXd P_aug_ = MatrixXd(7, 7);

  //create sigma point matrix
  MatrixXd Xsig_aug_ = MatrixXd(n_aug_, n_sig_);

  //create augmented mean state
  x_aug_.head(5) = x_;
  x_aug_(5) = 0;
  x_aug_(6) = 0;

  //create augmented covariance matrix
  P_aug_.fill(0.0);
  P_aug_.topLeftCorner(5,5) = P_;
  P_aug_(5,5) = std_a_*std_a_;
  P_aug_(6,6) = std_yawdd_*std_yawdd_;

  //create square root matrix
  MatrixXd L = P_aug_.llt().matrixL();

  //create augmented sigma points
  Xsig_aug_.col(0)  = x_aug_;
  for (int i = 0; i< n_aug_; i++) {
    Xsig_aug_.col(i+1)       = x_aug_ + sqrt(lambda_+n_aug_) * L.col(i);
    Xsig_aug_.col(i+1+n_aug_) = x_aug_ - sqrt(lambda_+n_aug_) * L.col(i);
  }
  
  //predict sigma points
  for (int i = 0; i< 2*n_aug_+1; i++) {
    //extract values for better readability
    double p_x = Xsig_aug_(0,i);
    double p_y = Xsig_aug_(1,i);
    double v = Xsig_aug_(2,i);
    double yaw = Xsig_aug_(3,i);
    double yawd = Xsig_aug_(4,i);
    double nu_a = Xsig_aug_(5,i);
    double nu_yawdd = Xsig_aug_(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    } else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }

  //predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {  //iterate over sigma points
    x_ = x_+ weights_(i) * Xsig_pred_.col(i);
  }

  //predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {  //iterate over sigma points

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    Normalize(x_diff);
    P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
  }

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(laser_n_z, n_sig_);

  //transform sigma points into measurement space
  for (int i = 0; i < n_sig_; i++) {  //2n+1 simga points

    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);

    // measurement model
    Zsig(0,i) = p_x;
    Zsig(1,i) = p_y;
  }

  //mean predicted measurement
  VectorXd z_pred = VectorXd(laser_n_z);
  CalculateMeanPredicted(z_pred, Zsig);
  //measurement covariance matrix S
  MatrixXd S = MatrixXd(laser_n_z,laser_n_z);
  CalculateCovarianceMatrix(S, z_pred, Zsig);
  //add measurement noise covariance matrix
  S = S + R_laser_;
  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, laser_n_z);
  CalculateCrossCorrelationMatrix(Tc, z_pred, Zsig);
  // vector for incoming radar measurement
  VectorXd z = VectorXd(laser_n_z);
  z << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1] ;
  //residual
  VectorXd z_diff = z - z_pred;
  Normalize(z_diff);
  UpdateStateAndCovariance(S, Tc, z_diff);
  // Calculate NIS for radar
  NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(radar_n_z, n_sig_);

  //transform sigma points into measurement space
  for (int i = 0; i < n_sig_; i++) {  //2n+1 simga points

    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
    Zsig(1,i) = atan2(p_y,p_x);                                 //phi
    Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
  }

  //mean predicted measurement
  VectorXd z_pred = VectorXd(radar_n_z);
  CalculateMeanPredicted(z_pred, Zsig);
  //measurement covariance matrix S
  MatrixXd S = MatrixXd(radar_n_z, radar_n_z);
  CalculateCovarianceMatrix(S, z_pred, Zsig);
  //add measurement noise covariance matrix
  S = S + R_radar_;
  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, radar_n_z);
  CalculateCrossCorrelationMatrix(Tc, z_pred, Zsig);
  // vector for incoming radar measurement
  VectorXd z = VectorXd(radar_n_z);
  z << meas_package.raw_measurements_[0],
       meas_package.raw_measurements_[1],
       meas_package.raw_measurements_[2];
  //residual
  VectorXd z_diff = z - z_pred;
  Normalize(z_diff);
  UpdateStateAndCovariance(S, Tc, z_diff);
  // Calculate NIS for radar
  NIS_radar_ = z_diff.transpose()*S.inverse()*z_diff;
}

void UKF::CalculateMeanPredicted(VectorXd &z_pred, MatrixXd &Zsig) {
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; i++) {
      z_pred = z_pred + weights_(i) * Zsig.col(i);
  }
}

void UKF::CalculateCovarianceMatrix(MatrixXd &S, VectorXd &z_pred, MatrixXd &Zsig) {
  S.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    Normalize(z_diff);
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }
}

void UKF::CalculateCrossCorrelationMatrix(MatrixXd &Tc, VectorXd &z_pred, MatrixXd &Zsig) {
  Tc.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    Normalize(z_diff);
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    Normalize(x_diff);
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }
}

void UKF::Normalize(VectorXd &diff) {
  while (diff(1)> M_PI) diff(1)-=2.*M_PI;
  while (diff(1)<-M_PI) diff(1)+=2.*M_PI;
}

void UKF::UpdateStateAndCovariance(MatrixXd &S, MatrixXd &Tc, VectorXd &z_diff) {
  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();
  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
}