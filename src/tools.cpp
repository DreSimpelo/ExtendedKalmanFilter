#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  // Calculate the RMSE
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  // Check if estimates is empty
  if(estimations.size() == 0){
    cout << "No estimates to calculate RSME" << endl;
  }

  // Check size difference of estimates and labels
  if(estimations.size() != ground_truth.size()){
    cout << "Size of estimates does not match the size of labels" << endl;
  }

  // RMSE Calculation: Accumulate Error
  for(int i =0; i < estimations.size(); i++){
    // Squared Residuals
    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array()*residual.array();

    // Accumulate Error Sum
    rmse += residual;
  }

  // Mean -> Square Root
  rmse = rmse/estimations.size();
  return rmse.array().sqrt();
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd Hj(3,4);

  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  // Useful calculations
  float px_square = pow(px,2);
  float py_square = pow(py,2);
  float px_py_square_sum = px_square + py_square;
  float px_py_sqrt = sqrt(px_py_square_sum);
  float px_py_n32 = pow(px_py_square_sum,1.5); // Raised to the 3/2 or 1.5 power

  // Check for zero denominator
  if(px_py_square_sum == 0){
    cout << "Can't divide by 0 (Tools::CalculateJacobian)" << endl;
    return Hj;
  }else{
    Hj << px/px_py_sqrt,              py/px_py_sqrt,              0,             0,
          -py/px_py_square_sum,       px/px_py_square_sum,        0,             0,
          py*(vx*py-vy*px)/px_py_n32, px*(vy*px-vx*py)/px_py_n32, px/px_py_sqrt, py/px_py_sqrt;

    return Hj;
  }
}
