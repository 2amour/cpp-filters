#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const Eigen::VectorXd &z, const VectorXd &z_pred) {
  if (z(0) == 0 && z(1) == 0 && z(2) == 0) {
       return;
  }
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
  MatrixXd PHt = P_ * Ht;
  MatrixXd S = H_ * PHt + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = PHt * Si;
  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

Eigen::VectorXd KalmanFilter::UpdateKF() {
  return H_ * x_;
}

Eigen::VectorXd KalmanFilter::UpdateEKF() {
  VectorXd z_pred(3);
  float t1 = sqrt(x_(0)*x_(0) + x_(1)*x_(1));
  float t2 = atan2(x_(1), x_(0));
  float t3 = (x_(0) * x_(2) + x_(1) * x_(3)) / t1;
  z_pred << t1, t2, t3;
  return z_pred;
}
