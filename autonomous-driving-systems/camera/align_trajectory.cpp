#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <ceres/loss_function.h>
#include <ceres/types.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

constexpr int quaternion_size = 4;
constexpr int translation_size = 3;

// Residual: transforms point p by rotation (quaternion) + translation, compares to target q
struct AlignmentResidual
{
    AlignmentResidual(const Eigen::Vector3d& p, const Eigen::Vector3d& q)
      : sourcePoint(p)
      , targetPoint(q)
    {}

    template<typename T>
    bool operator()(const T* const q_params, const T* const t_params, T* residuals) const
    {
        // Get rotation and translation from parameters
        const Eigen::Map<const Eigen::Quaternion<T>> rotation(q_params);
        const Eigen::Map<const Eigen::Vector3<T>> translation(t_params);

        // Get residuals map
        Eigen::Map<Eigen::Vector3<T>> residual_map(residuals);

        // Compute transformed point
        const Eigen::Vector3<T> p_transformed = rotation * sourcePoint.cast<T>() + translation;

        // Compute residuals
        residual_map = targetPoint.cast<T>() - p_transformed;

        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d& p, const Eigen::Vector3d& q)
    {
        constexpr int num_residuals = 3;

        using AutoDiffWrapper = ceres::
          AutoDiffCostFunction<AlignmentResidual, num_residuals, quaternion_size, translation_size>;

        auto residual = new AlignmentResidual(p, q);
        auto autodiff_wrapper = new AutoDiffWrapper(residual);

        return autodiff_wrapper;
    }

    Eigen::Vector3d sourcePoint, targetPoint;
};

// Simple CSV loader
std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> load_csv(const std::string& path)
{
    std::ifstream file(path);
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> data;
    std::string line;
    if (!file.is_open())
        throw std::runtime_error("Could not open CSV");

    // Skip header
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string val;
        std::vector<double> nums;
        while (std::getline(ss, val, ','))
            nums.push_back(std::stod(val));
        if (nums.size() >= 6) {
            Eigen::Vector3d p(nums[0], nums[1], nums[2]);
            Eigen::Vector3d q(nums[3], nums[4], nums[5]);
            data.emplace_back(p, q);
        }
    }
    return data;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " trajectory.csv\n";
        return 1;
    }

    auto pairs = load_csv(argv[1]);
    std::cout << "Loaded " << pairs.size() << " point pairs\n";

    // Initial guess: identity rotation, zero translation
    Eigen::Quaterniond q_params = { 1, 0, 0, 0 }; // w, x, y, z
    Eigen::Vector3d t_params = { 0, 0, 0 };

    // Create problem
    ceres::Problem problem;

    // Add parameter blocks
    problem.AddParameterBlock(q_params.coeffs().data(), quaternion_size);
    problem.AddParameterBlock(t_params.data(), translation_size);

    // Add manifold
    auto* quaternionManifold = new ceres::EigenQuaternionManifold();
    problem.SetManifold(q_params.coeffs().data(), quaternionManifold);

    // Add residuals
    for (const auto& [p, q] : pairs) {
        problem.AddResidualBlock(
          AlignmentResidual::Create(p, q), nullptr, q_params.coeffs().data(), t_params.data());
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 100;
    options.num_threads = 4;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    std::cout << "Estimated rotation matrix: " << q_params << "\n";
    Eigen::Vector3d euler = q_params.toRotationMatrix().eulerAngles(0, 1, 2);
    std::cout << "Euler XYZ [deg]: " << (euler * 180.0 / M_PI).transpose() << std::endl;
    std::cout << "Estimated translation: " << t_params.transpose() << "\n";

    double RMSE = 0.0;
    for (const auto& [p, q] : pairs) {
        Eigen::Vector3d p_transformed = q_params * p + t_params;
        RMSE += (q - p_transformed).squaredNorm();
    }
    RMSE = std::sqrt(RMSE / pairs.size());
    std::cout << "RMSE: " << RMSE << "\n";

    return 0;
}
