#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>
#include <math.h>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;
using std::cout;
using std::endl;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  ekf_.x_ = VectorXd(4);
  ekf_.x_ << 1, 1, 1, 1;
  
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.P_ << 1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1000, 0,
             0, 0, 0, 1000;

  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1, 0, 1, 0,
             0, 1, 0, 1,
             0, 0, 1, 0,
             0, 0, 0, 1;

  ekf_.H_ = MatrixXd(2, 4);
  H_laser_ = MatrixXd(2, 4);
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  ekf_.R_ = MatrixXd(2, 2);
  R_laser_ = MatrixXd(2, 2);
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  R_radar_ = MatrixXd(3, 3);
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << 0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0;

  noise_ax = 9;
  noise_ay = 9;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      ekf_.x_ << measurement_pack.raw_measurements_[0]*cos(measurement_pack.raw_measurements_[1]),
                 measurement_pack.raw_measurements_[0]*sin(measurement_pack.raw_measurements_[1]),
                 0, 0;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
    }
    previous_timestamp_ = measurement_pack.timestamp_;
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
   */

  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;

  float dt_2 = dt * dt;
  float dt_3 = dt_2 * dt;
  float dt_4 = dt_3 * dt;

  //Modify the F matrix so that the time is integrated
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  ekf_.Q_ <<  dt_4/4*noise_ax, 0, dt_3/2*noise_ax, 0,
         0, dt_4/4*noise_ay, 0, dt_3/2*noise_ay,
         dt_3/2*noise_ax, 0, dt_2*noise_ax, 0,
         0, dt_3/2*noise_ay, 0, dt_2*noise_ay;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    ekf_.R_ = R_radar_;
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.Update(measurement_pack.raw_measurements_, ekf_.UpdateEKF());
  } else {
    ekf_.R_ = R_laser_;
    ekf_.H_ = H_laser_;
    ekf_.Update(measurement_pack.raw_measurements_, ekf_.UpdateKF());
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
