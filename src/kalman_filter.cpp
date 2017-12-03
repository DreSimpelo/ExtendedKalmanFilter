#include "kalman_filter.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

double NormalizePhi(double phi);

KalmanFilter::KalmanFilter() {}
KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in; // State Matrix X
  P_ = P_in; // Object Covariance Matrix P
  F_ = F_in; // State Transition Matrix F
  H_ = H_in; // Measurement Transform Matrix H
  R_ = R_in; // Measurement Covariance Matrix
  Q_ = Q_in; // Process Covariance Matrix
}

void KalmanFilter::Predict() {
  // Predict the state
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  // Update the state by using Kalman Filter equations
  VectorXd z_pred = H_ * x_;
  // Difference between measurement and prediction
  VectorXd y = z - z_pred;
  Filter(y);
}

double NormalizePhi(double phi)
{
  // https://stackoverflow.com/questions/4633177/c-how-to-wrap-a-float-to-the-interval-pi-pi
  // Will keep values within range [min,max]
  float max = M_PI;
  float min = -M_PI;

  if(phi < min){
    return max + std::fmod(phi - min, max - min); // if phi < -pi, add onto pi
  }else{
    return min + std::fmod(phi - min, max - min); // if phi > -pi, add onto -pi
  }
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  // update the state by using Extended Kalman Filter equations
  // Convert predicted state x_ (carteseian) into polar coordinates
  VectorXd x_polar = VectorXd(3);

  // State x_ in cartesian coordinates
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);

  // Helpful Calculation
  float sum_px2_py2 = pow(px,2) + pow(py,2);

  x_polar(0) = sqrt(sum_px2_py2);
  x_polar(1) = atan2(py,px);

  // x_polar(2): Check for division by zero
  float epsilon = 0.0001;
  if(x_polar(0) < epsilon){
    x_polar(2) = (px*vx+py*vy)/epsilon;
  }else{
    x_polar(2) = (px*vx+py*vy)/x_polar(0);
  }

  VectorXd y = z - x_polar;

  // Normalize Angle
  y[1] = NormalizePhi(y[1]);

  Filter(y);
}

void KalmanFilter::Filter(const VectorXd &y) {
  // Calculate Kalman Gain
  MatrixXd Ht = H_.transpose();
  MatrixXd predict_uncertainty = P_ * Ht;
  MatrixXd sum_covariance = H_ * P_ * Ht + R_;
  MatrixXd inv_sum_covariance = sum_covariance.inverse();

  MatrixXd kalman_gain = predict_uncertainty * inv_sum_covariance; // Prediction uncertainty / total uncertainty

  // Update x_ and P_
  x_ = x_ + (kalman_gain * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - kalman_gain * H_) * P_;
}