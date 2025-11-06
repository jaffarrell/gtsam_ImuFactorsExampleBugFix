// This example uses the Epson IMU in a controlled experiment.
// The IMU returns delta velocity and delta angles of the sampling
// interval.
// The controlled experiment uses motions during 20 second intervals.
// At the start and end of each 20 second interval, the IMU is placed at
// its original pose. The position portion of that pose is defined to be the
// origin of the world frame (i.e., [0, 0, 0]'). Therefore, every 20 second,
// we have at least one artificial position measurement equal to [0, 0, 0]'
// with noise covariance matrix determined by the experimenters placement
// accuracy, assumed to be 1e-3 [0, 0, 0] m.
#include <boost/program_options.hpp>

// GTSAM related includes.
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/dataset.h>

#include <Eigen/Dense>
#include <cassert>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace gtsam;
using namespace std;

using symbol_shorthand::B;  // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::V;  // Vel   (xdot,ydot,zdot)
using symbol_shorthand::X;  // Pose3 (x,y,z,roll,pitch,yaw)
using Epoch = chrono::time_point<chrono::system_clock>;
static constexpr double G = 9.80065;          //[m/s^2]
static constexpr double OBS_INTERVAL = 20.0;  //[s]
static constexpr double DEG_2_RAD = M_PI / 180.0;

struct ImuMeas {
  double count;
  double delta_vx;  // m/s/s
  double delta_vy;  // m/s/s
  double delta_vz;  // m/s/s

  double delta_thx;  // rad/s/s: units
  double delta_thy;  // rad/s/s: units
  double delta_thz;  // rad/s/s: units

  double time_since_start = 0.0;  // seconds since start of the file

  ImuMeas() = default;

  ImuMeas(double count, double delta_vx, double delta_vy, double delta_vz,
          double delta_thx, double delta_thy, double delta_thz,
          double time_since_start)
      : count(count),
        delta_vx(delta_vx),
        delta_vy(delta_vy),
        delta_vz(delta_vz),
        delta_thx(delta_thx),
        delta_thy(delta_thy),
        delta_thz(delta_thz),
        time_since_start(time_since_start) {}

  [[nodiscard]] Vector6 toVector() const noexcept {
    Vector6 dataPoint;
    dataPoint << delta_vx, delta_vy, delta_vz, delta_thx, delta_thy, delta_thz;
    return dataPoint;
  }

  Vector3 getGyro() const { return Vector3(delta_thx, delta_thy, delta_thz); }

  Vector3 getAccel() const { return Vector3(delta_vx, delta_vy, delta_vz); }
};

using PosZero = gtsam::Vector3;  // [r = {0, 0, 0}]
// Class to define the artificial position measurement each time that the
// experimenter places the IMU at its initial location.
struct Pos_Observation {
 private:
  // Helper function to create the position measurement
  static PosZero makePosZero() {
    return (PosZero() << 0.0, 0.0, 0.0).finished();
  }

  // Helper function to create the positin measurement noise model
  static gtsam::SharedDiagonal makePosNoiseModel() {
    // Use an anonymous struct or local variables for the variance values
    const double varR = 1e-6;  // micro meter

    return gtsam::noiseModel::Diagonal::Sigmas(
        (PosZero() << varR, varR, varR).finished());
  }

 public:
  // Const members initialized using the static helper functions
  const PosZero posZero;
  const gtsam::SharedDiagonal R;

  // Constructor
  Pos_Observation() : posZero(makePosZero()), R(makePosNoiseModel()) {}
};

using ImuMeasMap = map<Epoch, ImuMeas>;

[[nodiscard]] Epoch parseTimeStamp(const std::string& timeStamp) {
  std::tm t = {};
  std::istringstream ss(timeStamp);

  ss >> std::get_time(&t, "%Y-%m-%d %H:%M:%S");

  // Handle fractional part if present
  long long nanos = 0;
  if (ss.peek() == '.') {
    ss.get();  // eat '.'
    std::string frac;
    ss >> frac;

    // Normalize to nanoseconds (pad/truncate to 9 digits)
    while (frac.size() < 9) frac.push_back('0');
    if (frac.size() > 9) frac = frac.substr(0, 9);

    nanos = std::stoll(frac);
  }

  auto tt = timegm(&t);  // seconds since epoch (UTC)
  auto timePoint = std::chrono::system_clock::from_time_t(tt);
  timePoint += std::chrono::nanoseconds(nanos);

  return timePoint;
}

[[nodiscard]] std::string epochToString(const Epoch& epoch) {
  std::time_t tt = std::chrono::system_clock::to_time_t(epoch);
  std::tm tm = *std::gmtime(&tt);

  std::ostringstream ss;
  ss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");

  auto duration_since_epoch = epoch.time_since_epoch();
  auto seconds_part =
      std::chrono::duration_cast<std::chrono::seconds>(duration_since_epoch);
  auto fractional = duration_since_epoch - seconds_part;
  auto nanos =
      std::chrono::duration_cast<std::chrono::nanoseconds>(fractional).count();

  if (nanos > 0) {
    // Format with variable precision: 6 digits if micros, 9 if nanos
    std::string frac = std::to_string(nanos);
    while (frac.size() < 9) frac = "0" + frac;  // pad left
    // trim trailing zeros to match input precision
    while (frac.size() > 0 && frac.back() == '0') frac.pop_back();

    ss << "." << frac;
  }

  return ss.str();
}

/// @brief parses the input csv file to return an ImuMeasMap with one row for
/// each IMU measurement time that falls between start and end.
/// @param inputFileName
/// @param start
/// @param end
/// @return
[[nodiscard]] ImuMeasMap parseImuCsv(string inputFileName,
                                     Epoch start = Epoch(),
                                     Epoch end = Epoch()) {
  ImuMeasMap out;

  ifstream file(inputFileName);

  if (!file.is_open()) {
    throw runtime_error("Error: Could not open the file " + inputFileName);
  }
  string line;
  bool foundFirstEpoch = false;
  Epoch fileStartEpoch;
  // skip the header
  getline(file, line);
  while (getline(file, line)) {
    stringstream ss(line);
    string date, time, frac_seconds_str;

    // Read the date, time, and fractional seconds, which are space-separated
    ss >> date;
    getline(ss, time, '.');
    getline(ss, frac_seconds_str, ',');

    // Correctly build the timestamp string for parseTimeStamp
    string fullTimestampStr = date + " " + time + "." + frac_seconds_str;
    Epoch currentEpoch = parseTimeStamp(fullTimestampStr);
    // Check if in time range.
    if ((start == end) || (currentEpoch >= start && currentEpoch <= end)) {
      // Store the timestamp of the first valid entry
      if (!foundFirstEpoch) {
        fileStartEpoch = currentEpoch;
        foundFirstEpoch = true;
      }
      double time_since_start =
          chrono::duration<double>(currentEpoch - fileStartEpoch).count();

      ImuMeas meas;
      std::string temp;

      getline(ss, temp, ',');
      meas.count = std::stoi(temp);

      getline(ss, temp, ',');
      meas.delta_vx = std::stod(temp);

      getline(ss, temp, ',');
      meas.delta_vy = std::stod(temp);

      getline(ss, temp, ',');
      meas.delta_vz = std::stod(temp);

      getline(ss, temp, ',');
      meas.delta_thx = DEG_2_RAD * std::stod(temp);

      getline(ss, temp, ',');
      meas.delta_thy = DEG_2_RAD * std::stod(temp);

      getline(ss, temp, ',');
      meas.delta_thz = DEG_2_RAD * std::stod(temp);

      meas.time_since_start = time_since_start;

      out.insert_or_assign(currentEpoch, meas);
    }
  }

  return out;
}

template <typename Units>
[[nodiscard]] double durationSince(const Epoch t1, const Epoch t2) {
  auto diff = t2 - t1;
  return std::chrono::duration<double, typename Units::period>(diff).count();
}

namespace po = boost::program_options;

po::variables_map parseOptions(int argc, char* argv[]) {
  po::options_description desc;
  desc.add_options()("help,h", "produce help message")(
      "data_csv_path",
      po::value<string>()->default_value("Epson_G370_20230407_030457.csv"),
      "path to the CSV file with the IMU data")(
      "output_filename",
      po::value<string>()->default_value("EpsonG370_FactorResults.csv"),
      "path to the result file to use")("use_isam", po::bool_switch(),
                                        "use ISAM as the optimizer");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);

  if (vm.count("help")) {
    cout << desc << "\n";
    exit(1);
  }

  return vm;
}

// This section follows the methods in [CSM]
// See: [CSM] Jay A. Farrell, Felipe O. Silva, Farzana Rahman, and J. Wendel.
// "IMU Error Modeling Tutorial: INS state estimation and real-time sensor
// calibration IEEE Control Systems Magazine. Its methods are implemented at:
// See:
// https://github.com/jaffarrell/AV-Matlab-SW/blob/main/AV_Python/ASD_to_GaussMarkovFirstOrder.py
struct IMU_model_params {
  double Ts{0};   // IMU sample period
  double rtTs{0};
  double Tba{0};  // correlation time of accel bias, seconds
  double Tbg{0};  // correlation time of gyro bias, seconds
  double accel_noise_rtPSD{0};  // cont-time acc. meas noise, (m/s/s) / rtHz 
  double gyro_noise_rtPSD{0};   // cont-time gyro meas noise, (rad/s) / rtHz
  double phi_a{0}; // discrete-time state transition, accelerometer
  double phi_g{0}; // discrete-time state transition, gyro
  double accel_bias_dn_PSD{0}; // cont-time accel bias drvng noise PSD
  double gyro_bias_dn_PSD{0};  // cont-time gyro bias drvng noise PSD
  double accel_bias_cov_ss{0}; // steady-state accel bias cov
  double gyro_bias_cov_ss{0};  // steady-state accel bias cov
  double accel_bias_dn_cov{0}; // discrete-time accl bias driving noise cov
  double gyro_bias_dn_cov{0};  // discrete-time gyro bias driving noise cov
  double accel_noise_rtCov{0}; // discrete-time accel meas noise std, m/s^2
  double accel_noise_rtCov_mps{0}; // discrete-time vel RW std mps
  double gyro_noise_rtCov_rps{0};  // discrete-time meas. noise, ang. rate std, rad/s
  double gyro_noise_rtCov_dph{0}; // discrete-time meas noise ang. rate std, deg/hr
  double gyro_noise_rtCov_rad{0}; // discrete-time meas noise angl. RW std, rad
  
  IMU_model_params(double Ts) {
    this->Ts = Ts;
    this->rtTs = sqrt(Ts);
    double Tpa = 100.0;  // seconds.
    Tba = Tpa / 1.89;    // seconds, Eqn. (37) in [CSM]
    double Tpg = 32;     // seconds.
    Tbg = Tpg / 1.89;    // seconds, Eqn. (37) in [CSM]
  }

  void print_IMU_params(){
    cout << "The IMU sample rate is " << Ts << " second. \n";

        cout << "The accelerometer model parameters are: \n\t"
           << "In continuous-time: Correl. time = " << Tba << "s, \n\t\t"
           << "vel. RW rtPSD = " << accel_noise_rtPSD << " m/s^2/rtHz = m/s/rtsec "
           << "= " << accel_noise_rtPSD * 60.0 << " m/s/rtHr, \n\t\t"
           << "bias drvng. noise rtPSD = " << sqrt(accel_bias_dn_PSD) << " m/s^3/rtHz (= m/s^s/rtsec). \n\t"
           << "In discrete-time: phi = " << phi_a << ", vel. RW std = " << accel_noise_rtCov_mps << " m/s, \n\t\t"
           << "bias drvng. noise cov = " << sqrt(accel_bias_dn_cov) << " m/s^2.\n"
           << "The steady-state accelerometer bias std is " << sqrt(accel_bias_cov_ss) << " m/s^2.\n" << std::endl;

        cout << "The gyroscope model parameters are: \n\t"
           << "In continuous-time: Correl. time = " << Tbg << "s, \n\t\t"
           << "angle RW rtPSD = " << gyro_noise_rtPSD << " rad/s/rtHz = rad/rtsec "
           << "= " << gyro_noise_rtPSD * 60.0 / DEG_2_RAD << " deg/rtHr, \n\t\t"
           << "bias drvng. noise rtPSD = " << sqrt(gyro_bias_dn_PSD) << " rad/s^2/rtHz (= rad/s/rtsec) "
           << "= " << sqrt(gyro_bias_dn_PSD) * 60.0 / DEG_2_RAD << " deg/s/rtHr. \n\t"
           << "In discrete-time: phi = " << phi_g << ", angle RW std = " << gyro_noise_rtCov_rad << " rads\n\t\t"
           << "bias drvng. noise cov = " << sqrt(gyro_bias_dn_cov) << " rad/s. \n" 
           << "The steady-state gyro bias std is " << sqrt(gyro_bias_cov_ss) << " rad/s "
           << "= " << sqrt(gyro_bias_cov_ss) / DEG_2_RAD << " deg/s.\n" << std::endl;

    }
  

  void set_accelInG(double accel_noise_rtPSD_g, double Ba_g){
    // See eqn (23) in IEEE CSM: This section concerns the measurement noise.
    // It relates to the portion of the ASD curve with slope -1/2.
    // accel_noise_PSD is N^2 = AV tau. 
    // The PSD has units of (m/s/s)^2 * (s) = (m/s^2)^2 / Hz
    // accel_noise_rtPSD is N = ASD * rtTau. 
    // N is the value of tangent line to the portion of the ASD graph 
    // that has slope -1/2, when that tangent is extended to tau = 1 sec. 
    // The rtPSD has units of (m/s^2)*/ rtHz
    accel_noise_rtPSD = (accel_noise_rtPSD_g) * G;  
    // See eqn. (62) in IEEE CSM. 
    // The Covariance of the discrete-time acceleration measurement white noise is 
    // the PSD / Ts, which has units of (m/s^2)^2. The standard deviation is the rtPSD
    // divided by rtTs.
    accel_noise_rtCov = accel_noise_rtPSD / rtTs; // (m/s/s)
    accel_noise_rtCov_mps = accel_noise_rtCov * Ts; // (m/s)

    // Ba is is the accel Bias Instability, which is the value of the ASD in its flat region
    // It may be specified in g's or m/s/s.
    double Ba_mps2 = (Ba_g * G);
    
    phi_a   = compute_phi(Ts, Tba);
    accel_bias_dn_PSD = BiasInstabASD_to_BiasPSD(Ba_mps2, Tba );
    accel_bias_cov_ss = compute_SteadyStateBiasCov(accel_bias_dn_PSD, Tba);
    accel_bias_dn_cov = compute_BiasDiscrete_DrvngNs_cov(phi_a, accel_bias_cov_ss);
  }

  void set_gyroInDeg(double gyro_noise_rtPSD_dps, double Bg_degperhr){
    // See eqn (23) in IEEE CSM: This section concerns the measurement noise.
    // It relates to the portion of the ASD curve with slope -1/2.
    // gyro_noise_PSD is N^2 = AV tau. 
    // The PSD has units of (deg/s)^2 * (s) = (deg/s)^2 / Hz
    // gyro_noise_rtPSD is N = ASD * rtTau. 
    // N is the value of tangent line to the portion of the ASD graph 
    // that has slope -1/2, when that tangent is extended to tau = 1 sec. 
    // The rtPSD has units of (deg/s)*/ rtHz
    gyro_noise_rtPSD = gyro_noise_rtPSD_dps * (DEG_2_RAD); // (rad/s) / rtHz
    // See eqn. (62) in IEEE CSM. 
    // The Covariance of the discrete-time gyro measurement white noise is 
    // the PSD / Ts, which has units of (rad/s)^2. The standard deviation is the rtPSD
    // divided by rtTs.
    gyro_noise_rtCov_rps = gyro_noise_rtPSD / rtTs; // (rad/s)
    double gyro_noise_rtCov_dps = gyro_noise_rtCov_rps / DEG_2_RAD;
    gyro_noise_rtCov_dph = gyro_noise_rtCov_dps * 3600;
    gyro_noise_rtCov_rad = gyro_noise_rtCov_rps * Ts;
    // Bg is the gyro bias instability, which is the value of the ASD curve in its 
    // flat region.
    double Bg_rps = ((Bg_degperhr * (1.0 / 3600.0) * (DEG_2_RAD))); 

    phi_g   = compute_phi(Ts, Tbg);
    gyro_bias_dn_PSD = BiasInstabASD_to_BiasPSD(Bg_rps, Tbg );
    gyro_bias_cov_ss = compute_SteadyStateBiasCov(gyro_bias_dn_PSD, Tbg);
    gyro_bias_dn_cov = compute_BiasDiscrete_DrvngNs_cov(phi_g, gyro_bias_cov_ss);
  }

  double compute_phi(double dT, double Tb){
    double phi = exp(-dT / Tb);
    return phi;
  }  

  // biasInstabASD is the value of the ASD in the flat region
  double BiasInstabASD_to_BiasPSD(double biasInstabASD, double Tb ){
    double biasInstab = biasInstabASD / 0.664; // Eqn. (32) in IEEE CSM
    // PSD for accel bias process noise, Eqn. (39) in [CSM]. Units are (m/s^2)/rtHz
    double bias_DrvngNs_PSD =
      (2.0 * pow(biasInstab, 2.0) * log(2.0)) / (M_PI * pow(0.4365, 2.0) * (Tb));
    return bias_DrvngNs_PSD;
  }
  
  // Set dP/dt = 0 and solve for P (assume scalars or diagonal matrices)
  // dP/dt = -lambda P - P * lambda + Q 
  double compute_SteadyStateBiasCov(double bias_DrvngNs_PSD, double Tb){
    double lambda = 1.0 / Tb;
    double SteadyStateCov = bias_DrvngNs_PSD / 2.0 / lambda;
    return SteadyStateCov;
  }

  // Set P(k+1) = P(k) = bias_cov_ss and solve for Qd
  // P(k+1) = phi P(k) phi' + Qd     (assume scalars or diagonal matrices)
  double compute_BiasDiscrete_DrvngNs_cov(double phi, double bias_cov_ss){
    double bias_GM_dn_cov = bias_cov_ss * (1.0 - pow(phi, 2.0));
    return bias_GM_dn_cov;
  }
};

// This section places the IMU stochastic error model parameters into the 
// GTSAM shared pointer structure. The model parameters are the same for each 
// of each instrument.  
std::shared_ptr<PreintegratedCombinedMeasurements::Params> imuParams(
    IMU_model_params imu){

  // The following assumes that GTSAM wants the continuous-time PSD's, 
  // even though GTSAM (incorrectly) refers to them as covariances. GTSAM
  // internally computes the discrete-time covaraince from the PSD's
  Matrix33 acc_meas_PSD = I_3x3 * pow(imu.accel_noise_rtPSD, 2);
  Matrix33 gyro_meas_PSD = I_3x3 * pow(imu.gyro_noise_rtPSD, 2);

  Matrix33 bias_acc_PSD = I_3x3 * imu.accel_bias_dn_PSD;
  Matrix33 bias_omega_PSD = I_3x3 * imu.gyro_bias_dn_PSD;

  // The following is not part of the IMU error model. It is here to
  // account for numeric error committed in integrating position from velocities
  Matrix33 integration_error_cov = I_3x3 * 1e-8;  // TODO: zero

  auto sPntrPIM = PreintegratedCombinedMeasurements::Params::MakeSharedD(0.0);

  // integration uncertainty continuous-time
  // (TODO: this should be zero)
  sPntrPIM->integrationCovariance = integration_error_cov;
 
  // measurement noise, acceleration white noise PSD: (m/s^2)^2/Hz
  sPntrPIM->accelerometerCovariance = acc_meas_PSD;  

  // measurement noise, gyro angle rate white noise PSD: (rad/s)^2 / Hz
  sPntrPIM->gyroscopeCovariance = gyro_meas_PSD;

  // Driving noise, Accelerometer bias random walk: (m/s^2)^2 / s
  sPntrPIM->biasAccCovariance = bias_acc_PSD;      // acc bias in continuous

  // Driving noise, Gyro bias random walk: (rad/s)^2 / s
  sPntrPIM->biasOmegaCovariance = bias_omega_PSD;  // gyro bias in continuous

  cout<< "integrationCovariance   = \n" << integration_error_cov << ",\n"
      << "accelerometerCovariance = \n" << acc_meas_PSD << ", \n"
      << "gyroscopeCovariance     = \n" << gyro_meas_PSD << ",\n"
      << "biasAccCovariance       = \n" << bias_acc_PSD << ",\n"
      << "biasOmegaCovariance     = " << bias_omega_PSD<< std::endl;

#ifdef GTSAM_ALLOW_DEPRECATED_SINCE_V43
  Matrix66 bias_acc_omega_init =
      I_6x6 * 1e-5;  // error in the bias used for preintegration
  sPntrPIM->biasAccOmegaInt = bias_acc_omega_init;
#endif

  return sPntrPIM;
}

// Output the IMU data with time since start of the experiment
// Plot the IMU data using the function gtsam_results_plotter.py
void printIMUMeasMap2csv(const ImuMeasMap& imu) {
  namespace fs = std::filesystem;
  std::string fileName = "imu_measurements_parsed.csv";
  fs::path filePath = fs::absolute(fileName);  // Get full path
  ofstream file(filePath);
  cout << "Printing outout at path: " << filePath << endl;
  if (!file.is_open()) {
    throw runtime_error("Error: Could not create output CSV file");
  }

  // Write CSV header
  file << "time_since_start,delta_vx,delta_vy,delta_vz,delta_thx,delta_thy,"
          "delta_thz\n";

  // Write data rows
  for (const auto& [epoch, imuMeas] : imu) {
    file << imuMeas.time_since_start << "," << imuMeas.delta_vx << ","
         << imuMeas.delta_vy << "," << imuMeas.delta_vz << ","
         << imuMeas.delta_thx << "," << imuMeas.delta_thy << ","
         << imuMeas.delta_thz << "\n";
  }

  file.close();
  cout << "IMU measurements exported to imu_measurements.csv" << endl;
}

int main(int argc, char* argv[]) {
  double R2D{180.0 / M_PI};
  string data_filename;
  string output_filename;
  std::vector<float> node_time;

  po::variables_map var_map = parseOptions(argc, argv);
  data_filename = findExampleDataFile(var_map["data_csv_path"].as<string>());
  output_filename = var_map["output_filename"].as<string>();

  const ImuMeasMap imu = parseImuCsv(data_filename.c_str());
  Epoch start = imu.begin()->first;

  double sample_freq = 125.0;
  double sample_period = 1.0 / sample_freq;
  cout << "IMU Sampling period: " << sample_period
       << " seconds, IMU Sampling frequency: " << sample_freq << " Hz." << endl;

  // Number of IMU measurements to preintegrate per PIM. Smaller numbers yield
  // denser time series of INS states, but cost computations. The IMU biases
  // are assumed constant over this batch of IMU measurements. 
  int  NUM_IMU_MEAS_PER_CHUNK = 25;

  // Export IMU measurements to CSV
  printIMUMeasMap2csv(imu);

  cout << "Using Levenberg Marquardt Optimizer" << endl;

  //====================================================
  // Start GTSam
  // Format is (N,E,D,qX,qY,qZ,qW,velN,velE,velD)
  Vector10 initial_veh_state{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
  // TODO: init attitude using accel vector.

  // Assemble initial quaternion through GTSAM constructor
  // ::quaternion(w,x,y,z);
  Rot3 prior_rotation =
      Rot3::Quaternion(initial_veh_state(6), initial_veh_state(3),
                       initial_veh_state(4), initial_veh_state(5));
  Point3 prior_point(initial_veh_state.head<3>());
  Pose3 prior_pose(prior_rotation, prior_point);
  Vector3 prior_velocity(initial_veh_state.tail<3>());
 
  // TODO: bias should be GM model, not constant
  // prior mean for bias at each time
  // Vector6: Use to init both accel and gyro bias
  imuBias::ConstantBias prior_imu_bias;
  cout << "initial point: " << prior_point << "\n";
  cout << "prior_pose: " << prior_pose << "\n";
  cout << "prior_velocity: " << prior_velocity << "\n";
  cout << "imu bias: " << prior_imu_bias << "\n\n";  // accel, gyro

  // initialize counters
  int node_count =
      0;  // count graph state nodes (initial plus number GPS corrections)
  int gps_measurement_count = 0;  // count GPS (stationary) measurements
  int imu_measurement_count = 0;  // count IMU measurements per correction
  int num_imu_measurements = imu.size();

  // Initial values are the starting point for linearization
  Values initial_values;
  initial_values.insert(X(node_count), prior_pose);      // six error states
  initial_values.insert(V(node_count), prior_velocity);  // three error states
  initial_values.insert(B(node_count), prior_imu_bias);  // six error states

  // TODO:At line 195 in ImuFactorExample.
  // Understand and adapt subsequent code to use deltav and deltaangle
  // with measurements at our stationary times
  // Then extract optimized trajectory and its covariance for plotting in
  // Python. IN the process understand how to also extract the Jacobian matrix
  // of the cost function. // Each pose will have 15 error states

  // TODO:
  //  - the accel and gyro noise models should not have the same parameters
  //  - adjust IMU noise models to our IMU specs
  //  - the bias models should be first-order GM models
  //  - the pose and velocity noise models should be zero or based on integrated
  //  IMU specs

  auto position_meas_noise =
      noiseModel::Isotropic::Sigma(3, 1e-3);  // std, meters

  // Assemble prior noise model and add it the graph.`
  auto pose_prior_noise_model = noiseModel::Diagonal::Sigmas(
      (Vector(6) << 0.001, 0.001, 0.001, 0.001, 0.001, 0.001)
          .finished());  // rad, rad, rad, m, m, m
  auto velocity_prior_noise_model =
      noiseModel::Isotropic::Sigma(3, 0.001);  // m/s
  auto bias_prior_noise_model = noiseModel::Isotropic::Sigma(6, 1e-1);

  // Add all prior factors (pose, velocity, bias) to the graph.
  NonlinearFactorGraph* graph = new NonlinearFactorGraph();
  node_time.push_back(0.0);
  graph->addPrior(X(node_count), prior_pose, pose_prior_noise_model);
  graph->addPrior(V(node_count), prior_velocity, velocity_prior_noise_model);
  graph->addPrior(B(node_count), prior_imu_bias, bias_prior_noise_model);
  graph->print("graph initial");
  initial_values.print("initial_values");

  IMU_model_params imu_constants(sample_period);
  double ASD_N_accel = 50e-6; // g
  double ASD_BI_accel= 8.0e-6;// ?
  imu_constants.set_accelInG(ASD_N_accel, ASD_BI_accel);
  double ASD_N_gyro  = 1e-3;// deg
  double ASD_BI_gyro = 0.9; // deg/rthr
  imu_constants.set_gyroInDeg(ASD_N_gyro, ASD_BI_gyro);
  imu_constants.print_IMU_params();
  // define IMU noise model parameters
  // (continuous-time noise power spectral
  // densities (units of variance × seconds))
  auto sPntrImuNoise = imuParams(imu_constants);


  // We will use a small bias-process model noise as a baseline
  // Build 6x6 covariance for BetweenFactor: accel(3) then gyro(3)
  Matrix66 bias_process_cov = Matrix66::Zero();
  bias_process_cov.topLeftCorner<3, 3>() = sPntrImuNoise->biasAccCovariance;
  bias_process_cov.bottomRightCorner<3, 3>() =
      sPntrImuNoise->biasOmegaCovariance;
  auto bias_process_noise = noiseModel::Gaussian::Covariance(bias_process_cov);
  

  // The PIMS have four steps (search PIM Step):
  // PIM Step 1: declare a PIM
  auto preintegrated = std::make_shared<PreintegratedCombinedMeasurements>(
      sPntrImuNoise, prior_imu_bias);
  assert(preintegrated);  // ensure success of declaration

  // Store previous state for imu integration and latest predicted outcome.
  NavState prev_state(
      prior_pose,
      prior_velocity);  // init state before first IMU measurement in PIM
  // init bias before first IMU measurement in PIM    
  imuBias::ConstantBias prev_bias = prior_imu_bias;  

  cout << "Starting loop over " << num_imu_measurements
       << " IMU measurements\n";
  Epoch last_node_epoch = start;  // epoch when last node was created (initial)
  for (const auto& [epoch, imuMeas] : imu) {
    double tm = imuMeas.time_since_start;

    // PIM Step 2: Add each IMU measurement to the PIM, until the next GPS
    // measurement or IMU node
    Vector3 dtheta = imuMeas.getGyro();
    Vector3 gyro = dtheta / sample_period;
    Vector3 dvel = imuMeas.getAccel();
    Vector3 spec_force = dvel / sample_period;
    preintegrated->integrateMeasurement(spec_force, gyro, sample_period);
    imu_measurement_count++;

    // Decide when to create a new node (either at OBS_INTERVAL stationary GPS
    // or at chunk limit)
    bool is_gps_time = (abs(tm - OBS_INTERVAL * (gps_measurement_count + 1)) <=
                        sample_period / 2) ||
                       (tm >= OBS_INTERVAL * (gps_measurement_count + 1));

    bool is_chunk_full = (imu_measurement_count >= NUM_IMU_MEAS_PER_CHUNK);

    if (is_gps_time || is_chunk_full) {
      node_count++;
      node_time.push_back(imuMeas.time_since_start);
      // PIM Step 3: Preintegrate this sequence of IMU measurements and add the
      // IMU factor Build CombinedImuFactor using the PREINTEGRATED COMBINED
      // measurements
      auto preint_combined =
          dynamic_cast<const PreintegratedCombinedMeasurements&>(
              *preintegrated);
      // define CombinedImuFactor with correct keys
      CombinedImuFactor combined_imu_factor(
          X(node_count - 1), V(node_count - 1), X(node_count), V(node_count),
          B(node_count - 1), B(node_count), preint_combined);
      // add combined factor to the graph
      graph->add(combined_imu_factor);

      // ----- Add bias Gauss-Markov model BetweenFactor(b_{k-1}, b_k) -----
      // Discretize first-order Gauss-Markov:
      //   b_k = phi * b_{k-1} + w,   w ~ N(0, Q)
      // with phi = exp(-dt/tau), Q = sigma_w^2 * (1 - phi^2)
      //

      // TODO: fix the following:
      // The BetweenFactor expects a "measurement" imuBias::ConstantBias that
      // corresponds to the additive difference between variables. To encode b_k
      // = phi * b_{k-1} + w we add a BetweenFactor with measurement =
      // imuBias::ConstantBias::Zero() and covariance Q, BUT this effectively
      // enforces b_k - phi * b_{k-1} ~ N(0, Q) only if we reparameterize
      // variables.
      //
      // GTSAM's BetweenFactor for imuBias::ConstantBias encodes: z =
      // b_{k-1}^{-1} * b_k. For small biases (additive), we approximate that
      // with b_k - b_{k-1} ~ N(0, Q_rw). To include phi we could create a
      // custom factor; as a practical and numerically-stable compromise, we
      // enforce that increments have covariance matching the Gauss-Markov
      // process (i.e., use Q above) which encourages the correct correlation
      // time via the covariance scale.
      //
      // (If you require an explicit multiplicative phi in the mean, we can
      // implement a small custom factor
      //  that enforces b_k - phi*b_{k-1} ~ N(0,Q). For most practical problems
      //  using the Q computed here gives the desired time correlation in the
      //  bias evolution.)
      //
      // So here we add a BetweenFactor which encourages small change with
      // covariance Q.
      //
      graph->add(BetweenFactor<imuBias::ConstantBias>(
          B(node_count - 1), B(node_count), prior_imu_bias, bias_process_noise));

      // ----- Add GPS factor if stationary time (so we anchor the pose) -----
      if (is_gps_time) {
        gps_measurement_count++;
        cout << "Creating node " << node_count
             << "  (gps_count=" << gps_measurement_count
             << "), imu_measurement_count=" << imu_measurement_count
             << ", time_since_start=" << imuMeas.time_since_start << "\n";

        GPSFactor gps_factor(X(node_count),
                             Point3(0.0,   // x
                                    0.0,   // y
                                    0.0),  // z
                             position_meas_noise);
        graph->add(gps_factor);
        // This is not another node.  It is a GPS unary factor at the same IMU
        // node.
      }

      // Predict state at this time and use to initialize linearization point
      NavState prop_state = preintegrated->predict(prev_state, prev_bias);
      initial_values.insert(X(node_count), prop_state.pose());
      initial_values.insert(V(node_count), prop_state.v());
      // initialize new bias variable using previous bias propagated by phi
      // (optional) We'll initialize to prev_bias (zero), which is acceptable
      // and will be optimized.
      initial_values.insert(B(node_count), prev_bias);

      // Optimize incrementally occasionally or at every measurement as you
      // prefer. For this example we do a local optimization each time a node is
      // added (as before).
      LevenbergMarquardtOptimizer optimizer(*graph, initial_values);
      Values result = optimizer.optimize();

      // Update previous state and bias from optimization result
      prev_state = NavState(result.at<Pose3>(X(node_count)),
                            result.at<Vector3>(V(node_count)));
      prev_bias = result.at<imuBias::ConstantBias>(B(node_count));

      // PIM Step 4: Reset the preintegration object to be ready for the next
      // sequence of IMU measurements Reset preintegration with the updated bias
      preintegrated->resetIntegrationAndSetBias(prev_bias);

      // reset counters & last node epoch
      imu_measurement_count = 0;
      last_node_epoch = epoch;

      if (gps_measurement_count >= 2) {
        // If you only wanted two GPS events as in original, break
        break;
      }
    }
  }  // end for imu

  // Final optimization
  cout << "\nEnd of loop. Optimizing final graph.\n";
  LevenbergMarquardtOptimizer final_optimizer(*graph, initial_values);
  Values final_result = final_optimizer.optimize();

  std::cout << "node_count: " << node_count
            << ", number of times: " << node_time.size() << std::endl;

  // Set up output file for plotting errors
  cout << "Outputting to file: " << output_filename << endl;
  FILE* fp_out = fopen(output_filename.c_str(), "w+");
  fprintf(fp_out,
          "time(s),x(m),y(m),z(m),Roll(deg),Pitch(deg),Yaw(deg),"
          "vx(mps),vy(mps),vz(mps),"
          "bgx(deg/s),bgy(deg/s),bgz(deg/s),"
          "bax(m/s^2),bay(m/s^2),baz(m/s^2)\n");
  cout << "Results:\n";
  for (int i = 0; i <= node_count; ++i) {
    if (!final_result.exists(X(i)) || !final_result.exists(V(i)) ||
        !final_result.exists(B(i)))
      continue;

    gtsam::Pose3 pose = final_result.at<gtsam::Pose3>(X(i));
    gtsam::Vector3 velocity = final_result.at<gtsam::Vector3>(V(i));
    gtsam::imuBias::ConstantBias bias =
        final_result.at<gtsam::imuBias::ConstantBias>(B(i));

    // Extract state components
    double t = node_time[i];
    gtsam::Vector3 p = pose.translation();
    gtsam::Vector3 rpy = pose.rotation().rpy() * R2D;  // radians → degrees
    gtsam::Vector3 v = velocity;
    gtsam::Vector3 bg = bias.gyroscope() * R2D;  // rad/s → deg/s
    gtsam::Vector3 ba = bias.accelerometer();

    // ---- CSV output ----
    fprintf(fp_out,
            "%.4f,%.4f,%.4f,%.4f,"  // time, x, y, z
            "%.3f,%.3f,%.3f,"       // roll, pitch, yaw (deg)
            "%.4f,%.4f,%.4f,"       // vx, vy, vz
            "%.8f,%.8f,%.8f,"       // bgx, bgy, bgz (deg/s)
            "%.8f,%.8f,%.8f\n",     // bax, bay, baz (m/s^2)
            t, p.x(), p.y(), p.z(), rpy.x(), rpy.y(), rpy.z(), v.x(), v.y(),
            v.z(), bg.x(), bg.y(), bg.z(), ba.x(), ba.y(), ba.z());

    if (i < 6) {
      std::cout << "State at node " << i << std::endl;
      std::cout << "\tTime, sec: " << t << std::endl;
      std::cout << "\tPosition, m: " << p.transpose() << std::endl;
      std::cout << "\tEuler, deg: " << rpy.transpose() << std::endl;
      std::cout << "\tVelocity, mps: " << v.transpose() << std::endl;
      std::cout << "\tGyro Bias, deg/s: " << bg.transpose() << std::endl;
      std::cout << "\tAccel Bias, mps^2: " << ba.transpose() << std::endl
                << std::endl;
    }
  }
  fclose(fp_out);
  delete graph;
  return 0;
}