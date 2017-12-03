#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // Laser: Measurement Transform Matrix H
  H_laser_ = MatrixXd(2, 4);
  H_laser_ << 1,0,0,0,
              0,1,0,0;

  // Radar: Measurement Transform (jacobian) matrix Hj
  Hj_radar_ = MatrixXd(3, 4); // initialize after a state has been seen. (during update phase)

  // Set the process and measurement noises:
  // measurement covariance matrix R - laser
  R_laser_ = MatrixXd(2, 2);
  R_laser_ << 0.0225, 0,
      0, 0.0225;

  // measurement covariance matrix R - radar
  R_radar_ = MatrixXd(3, 3);
  R_radar_ << 0.09, 0,      0,
      0,    0.0009, 0,
      0,    0,      0.09;

  // Object covariance matrix P_
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.P_ << 1, 0, 0,    0,
             0, 1, 0,    0,
             0, 0, 1000, 0,
             0, 0, 0,    1000;

  // Process covariance matrix Q_
  ekf_.Q_ = MatrixXd(4, 4);

  // Set F_ (Fill dt later)
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1, 0, 1, 0,
             0, 1, 0, 1,
             0, 0, 1, 0,
             0, 0, 0, 1;

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
    // Initialize the state ekf_.x_ with the first measurement.
    ekf_.x_ = VectorXd(4);

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // Convert radar from polar to cartesian coordinates and initialize state.

      float rho = measurement_pack.raw_measurements_(0);
      float theta = measurement_pack.raw_measurements_(1);
      float rho_dot = measurement_pack.raw_measurements_(2);

      // Populate ekf state x_  (in Cartesian coords)
      ekf_.x_(0) = rho * cos(theta); // x
      ekf_.x_(1) = rho * sin(theta); // y
      ekf_.x_(2) = rho_dot * cos(theta); // Vx
      ekf_.x_(3) = rho_dot * sin(theta); // Vy
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // Initialize state.

      ekf_.x_(0) = measurement_pack.raw_measurements_(0); // x
      ekf_.x_(1) = measurement_pack.raw_measurements_(1); // y
      ekf_.x_(2) = 0; // initialize Vx to 0
      ekf_.x_(3) = 0; // initialize Vy to 0
    }
    previous_timestamp_ = measurement_pack.timestamp_;
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }
  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  // Compute change in time
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
  previous_timestamp_ = measurement_pack.timestamp_;

  // Update the state transition matrix F according to the new elapsed time.
  ekf_.F_ << 1, 0, dt, 0,
             0, 1, 0,  dt,
             0, 0, 1,  0,
             0, 0, 0,  1;

  // Useful calculations
  float dt2 = dt * dt;
  float dt3 = dt2 * dt;
  float dt4 = dt3 * dt;

  // Update the process noise covariance matrix
  ekf_.Q_ << MatrixXd(4,4);
  ekf_.Q_ << dt4/4*noise_ax,  0,              dt3/2*noise_ax, 0,
             0,               dt4/4*noise_ay, 0,              dt3/2*noise_ay,
             dt3/2*noise_ax,  0,              dt2*noise_ax,   0,
             0,               dt3/2*noise_ay, 0,              dt2*noise_ay;

  // Predict!
  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/
  // Use the sensor type to perform the update step.
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    Hj_radar_ = tools.CalculateJacobian(ekf_.x_); // Caclulate Jacobian matrix
    ekf_.H_ = Hj_radar_; // Set ekf measurement matrix H_
    ekf_.R_ = R_radar_; // Set ekf measurement covariance R_

    // Update the state and covariance matrices.
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);

  }else{
    // Laser updates
    ekf_.H_ = H_laser_; // Set ekf measurement matrix H_
    ekf_.R_ = R_laser_; // Set ekf measurement covariance R_

    // Update the state and covariance matrices.
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
