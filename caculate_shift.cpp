#include <opencv2/opencv.hpp>
#include <numeric>
#include <iostream>
#include <fstream>
#include <vector>
#include <boost/filesystem.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/math/special_functions/round.hpp>
#include <cmath>
#include <algorithm>
#include <complex>
#include <opencv2/opencv.hpp>
#include "matplotlibcpp.h"
#include <Eigen/Dense>
#include <utility>
#include <limits>
#include <fftw3.h>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

//3.在 12600个一维向量当中，全部对齐模板线。将right_results.dat 保存为 right_results_shift.dat

//根据指定的轴向（'X' 或 'Y'）计算图像中白色像素（像素值为 255）在该轴向上的平均位置，并对结果进行一系列处理，包括中心化和正规化
std::vector<double> scan_and_visualize(const std::string& image_path, char axis)
{
	cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
	if (image.empty()) {
		throw std::runtime_error("Image file not found: " + image_path);
	}

	std::vector<double> result;
	if (axis == 'X') {
		result.resize(image.cols);
		for (int x = 0; x < image.cols; ++x) {
			std::vector<int> y_coords;
			for (int y = 0; y < image.rows; ++y) {
				if (image.at<uchar>(y, x) == 255) {
					y_coords.push_back(y);
				}
			}
			if (!y_coords.empty()) {
				double mean_y = std::accumulate(y_coords.begin(), y_coords.end(), 0.0) / y_coords.size();
				result[x] = mean_y;
			}
			else {
				result[x] = 0;
			}
		}
	}
	else if (axis == 'Y') {
		result.resize(image.rows);
		for (int y = 0; y < image.rows; ++y) {
			std::vector<int> x_coords;
			for (int x = 0; x < image.cols; ++x) {
				if (image.at<uchar>(y, x) == 255) {
					x_coords.push_back(x);
				}
			}
			if (!x_coords.empty()) {
				double mean_x = std::accumulate(x_coords.begin(), x_coords.end(), 0.0) / x_coords.size();
				result[y] = mean_x;
			}
			else {
				result[y] = 0;
			}
		}
	}
	else {
		throw std::invalid_argument("Axis must be 'X' or 'Y'");
	}

	double mean_result = std::accumulate(result.begin(), result.end(), 0.0) / result.size();//从begin到end累加到0
	for (double& value : result) {
		value -= mean_result;
	}

	double min_result = *std::min_element(result.begin(), result.end());
	double max_result = *std::max_element(result.begin(), result.end());
	if (max_result - min_result != 0) {
		for (double& value : result) {
			value = 2 * (value - min_result) / (max_result - min_result) - 1;
		}
	}

	return result;
}
// 创建从 0 到 len(y_data) - 1 的范围
std::vector<double> createRange(int size) {
	std::vector<double> range(size);
	for (int i = 0; i < size; ++i) {
		range[i] = (double)i;
	}
	return range;
}

// FFT 实现
std::vector<std::complex<double>> fft(const std::vector<double>& data) {
	size_t N = data.size();
	std::vector<std::complex<double>> result(N);

	// 创建 FFTW 计划
	fftw_complex *in, *out;
	in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
	out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);

	// 将输入数据复制到 FFTW 的输入数组
	for (size_t i = 0; i < N; ++i) {
		in[i][0] = data[i]; // 实部
		in[i][1] = 0.0;     // 虚部
	}

	// 创建 FFTW 计划
	fftw_plan p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

	// 执行 FFT
	fftw_execute(p);

	// 将 FFT 结果复制到 std::vector
	for (size_t i = 0; i < N; ++i) {
		result[i] = std::complex<double>(out[i][0], out[i][1]);
	}

	// 清理
	fftw_destroy_plan(p);
	fftw_free(in);
	fftw_free(out);

	return result;
}

// 计算 FFT 频率分量
std::vector<double> fft_freq(int N, double d = 1.0) {
	std::vector<double> frequencies(N);

	// 生成频率分量
	for (int k = 0; k < N; ++k) {
		frequencies[k] = k / (N * d);
	}

	// 处理负频率部分
	for (int k = 0; k < N / 2; ++k) {
		frequencies[k] = k / (N * d);
		frequencies[N / 2 + k] = (k - N / 2) / (N * d);
	}

	return frequencies;
}
std::vector<double> filter_positive_frequencies(const std::vector<double>& frequencies) {
	std::vector<double> positive_frequencies;
	// 遍历 frequencies 向量，筛选非负的频率
	for (const double& freq : frequencies) {
		if (freq >= 0) {
			positive_frequencies.push_back(freq);
		}
	}
	return positive_frequencies;
}
// 筛选出正频率对应的 FFT 结果
std::vector<std::complex<double>> filter_positive_fft_result(
	const std::vector<std::complex<double>>& fft_result,
	const std::vector<double>& frequencies
) {
	std::vector<std::complex<double>> positive_fft_result;
	// 遍历频率并筛选出对应的 FFT 结果
	for (size_t i = 0; i < frequencies.size(); ++i) {
		if (frequencies[i] >= 0) {
			positive_fft_result.push_back(fft_result[i]);
		}
	}

	return positive_fft_result;
}
// 计算复数的绝对值
double abs_value(const std::complex<double>& c) {
	return std::abs(c);  // 使用标准库函数计算绝对值
}

// 找到绝对值最大的复数的索引
size_t find_max_absolute_index(const std::vector<std::complex<double>>& data) {
	if (data.empty()) {
		throw std::invalid_argument("Input vector is empty.");
	}
	size_t max_index = -1;
	double max_abs = 0;

	for (size_t i = 0; i < data.size(); ++i) {
		double current_abs = abs_value(data[i]);
		if (current_abs > max_abs) {
			max_abs = current_abs;
			max_index = i;
		}
	}

	return max_index;
}
// 定义正弦函数
double sine_func(double x, const std::vector<double>& params) {
	return params[0] * std::sin(params[1] * x + params[2]) + params[3];
}
// 最小二乘拟合sin函数
void curve_fit(const std::vector<double>& xData, const std::vector<double>& yData, double omega, double& A, double& phi, double& b) {
	int n = xData.size();
	Eigen::MatrixXd AMatrix(n, 3);
	Eigen::VectorXd bVector(n);

	// 构造系数矩阵和右侧向量
	for (int i = 0; i < n; i++) {
		AMatrix(i, 0) = std::sin(omega * xData[i]);
		AMatrix(i, 1) = std::cos(omega * xData[i]);
		AMatrix(i, 2) = 1.0;
		bVector(i) = yData[i];
	}

	// 使用最小二乘法计算拟合参数
	Eigen::Vector3d params = AMatrix.colPivHouseholderQr().solve(bVector);

	A = std::sqrt(params(0) * params(0) + params(1) * params(1));
	phi = std::atan2(params(1), params(0));
	b = params(2); // 偏移量b
}

// 生成线性间隔数组的函数
std::vector<double> linspace(double start, double end, int num_points) {
	std::vector<double> points(num_points);
	double step = (end - start) / (num_points - 1);
	for (int i = 0; i < num_points; ++i) {
		points[i] = start + i * step;
	}
	return points;
}
// 辅助函数：获取数值的符号
int sign(double value) {
	return (value > 0) - (value < 0);
}

// 函数：找到 y_fit 中的第一个局部最小值
std::pair<size_t, double> find_first_min(const std::vector<double>& y_fit) {
	// 计算差分
	std::vector<int> min_indices;

	if (y_fit.size() < 3) {
		// 不足以计算二阶差分
		throw std::invalid_argument("Input vector size is too small to calculate second-order differences.");
	}

	// 计算一阶差分
	std::vector<double> diff1(y_fit.size() - 1);
	for (size_t i = 0; i < y_fit.size() - 1; ++i) {
		diff1[i] = y_fit[i + 1] - y_fit[i];
	}

	// 计算一阶差分的符号
	std::vector<int> sign1(y_fit.size() - 1);
	for (size_t i = 0; i < diff1.size(); ++i) {
		sign1[i] = (diff1[i] > 0) - (diff1[i] < 0); // 1 for positive, -1 for negative, 0 for zero
	}

	// 计算二阶差分
	std::vector<int> diff2(y_fit.size() - 2);
	for (size_t i = 0; i < sign1.size() - 1; ++i) {
		diff2[i] = sign1[i + 1] - sign1[i];
	}

	// 找到二阶差分大于0的位置
	for (size_t i = 0; i < diff2.size(); ++i) {
		if (diff2[i] > 0) {
			size_t min_index = i + 1; // 加1以补偿索引偏移
			double y_min = y_fit[min_index];
			return { min_index, y_min }; // 返回结果
		}
	}
	// 如果没有找到局部最小值
	return { static_cast<size_t>(-1), 0.0 }; // 使用 -1 表示没有找到局部最小值
}

void fit_curve(const std::vector<double>& y_data, std::vector<double>& params, double& t_min, double& y_min) {
	std::vector<double> x_data = createRange(y_data.size());
	// 调用封装的FFT函数
	std::vector<std::complex<double>> fft_result = fft(y_data);
	if (fft_result.empty()) {
		std::cout << "FFT result is empty." << std::endl;
	}
	std::vector<double> frequencies = fft_freq(y_data.size(), 1.0);
	if (fft_result.empty()) {
		std::cout << "frequencies is empty." << std::endl;
	}
	std::vector<double> positive_frequencies = filter_positive_frequencies(frequencies);
	// 筛选正频率对应的 FFT 结果
	std::vector<std::complex<double>> positive_fft_result = filter_positive_fft_result(fft_result, frequencies);
	// 找到最大绝对值的索引
	size_t main_frequency_index = find_max_absolute_index(positive_fft_result);
	// 获取最大绝对值对应的频率
	double main_frequency = positive_frequencies[main_frequency_index];
	const double PI = 4 * std::atan(1.0);//Π
	// 计算最大值和最小值
	double max_y = *std::max_element(y_data.begin(), y_data.end());
	double min_y = *std::min_element(y_data.begin(), y_data.end());
	// 计算 A_initial
	double A_initial = (max_y - min_y) / 2;
	// 计算 omega_initial
	double omega_initial = 2 * PI * main_frequency;
	// 计算 C_initial
	double sum_y = std::accumulate(y_data.begin(), y_data.end(), 0.0);
	double C_initial = sum_y / y_data.size();
	// 其他参数
	double phi_initial = 0;

	// 执行拟合
	curve_fit(x_data, y_data, omega_initial, A_initial, phi_initial, C_initial);
	params = { A_initial, omega_initial, phi_initial, C_initial };
	// 生成线性间隔数组
	std::vector<double> x_fit = linspace(0, y_data.size() - 1, 1000);
	// 计算 y_fit
	std::vector<double> y_fit(x_fit.size());
	for (size_t i = 0; i < x_fit.size(); ++i) {
		y_fit[i] = sine_func(x_fit[i], params);
	}
	size_t t_min_index = -1;
	// 查找第一个局部最小值
	std::pair<size_t, double> result_min = find_first_min(y_fit);
	t_min_index = result_min.first;
	y_min = result_min.second;

	if (t_min_index != -1) {
		size_t index = t_min_index;
		if (index >= 0 && index < x_fit.size()) {
			t_min = x_fit[index];
		}
		else {
			throw std::out_of_range("index of y_fit has changed,but index out of range");
		}
	}
	else {
		std::cout << "没有找到第一个局部最小值" << std::endl;
	}
}

// 线性插值函数
double linearInterpolate(double x0, double y0, double x1, double y1, double x) {
	return y0 + (x - x0) * (y1 - y0) / (x1 - x0);
}

std::vector<double> interpolateShift(const std::vector<double>& arr, double shift, double shift_fillValue) {
	int n = arr.size();
	std::vector<double> x(n);
	std::iota(x.begin(), x.end(), 0); // 使用 std::iota 填充 x

	std::vector<double> shiftedArr(n, shift_fillValue); // 先用填充值填充整个 shiftedArr

	for (int i = 0; i < n; ++i) {
		double shiftedX = x[i] - shift;
		int leftIndex = static_cast<int>(std::floor(shiftedX));
		int rightIndex = leftIndex + 1;

		// 确保索引仍在有效范围内
		if (leftIndex >= 0 && rightIndex < n) {
			double x0 = x[leftIndex];
			double y0 = arr[leftIndex];
			double x1 = x[rightIndex];
			double y1 = arr[rightIndex];
			shiftedArr[i] = linearInterpolate(x0, y0, x1, y1, shiftedX);
		}
	}

	return shiftedArr;
}

void caculate_shift()
{
	// 加载模板曲线
	std::vector<double> temp_result = scan_and_visualize("./temp/temp_right_curve.png", 'Y');

	// 从文件中加载比如 right_results.dat 的数据
	std::string result_path = "F:/data/test/right_results.dat";
	std::string output_dir = "F:/data/results";

	//从二进制文件中读取数据并反序列化到二维向量中
	std::vector<std::vector<double>> right_results;
	{
		std::ifstream inputFile(result_path.c_str(), std::ios::binary);
		boost::archive::binary_iarchive archive(inputFile);
		archive >> right_results;
	}
	// 设置要取出的曲线数量 n
	int n = right_results.size();
	std::vector<std::vector<double>> selected_results(right_results.begin(), right_results.begin() + n);
	std::vector<std::vector<double>> right_results_shift;
	std::vector<double> params_temp, params;
	double t_min_temp = -1, y_min_temp = -1;
	double t_min, y_min;
	fit_curve(temp_result, params_temp, t_min_temp, y_min_temp);
	for (int i = 0; i < selected_results.size(); ++i) {
		t_min = -1;
		y_min = -1;
		std::cout << selected_results[i].size() << std::endl;
		fit_curve(selected_results[i], params, t_min, y_min);
		if (t_min != -1 && t_min_temp != -1) {
			double time_shift = t_min_temp - t_min;
			double nanValue = std::numeric_limits<double>::quiet_NaN();
			std::vector<double> result_shift = interpolateShift(selected_results[i], time_shift, nanValue);
			right_results_shift.push_back(result_shift);
			std::cout << "Result " << i + 1 << " 处理完成。" << std::endl;
		}
		else {
			std::cout << "Result " << i + 1 << " 没有找到合适的极小值，跳过此对齐。" << std::endl;
		}
	}
	// 保存结果
	{
		std::ofstream outputFile((output_dir + "/right_results_shift.dat").c_str(), std::ios::binary);
		boost::archive::binary_oarchive archive(outputFile);
		archive << right_results_shift;
	}
}
int main()
{
	caculate_shift();
	return 0;
}