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

//3.�� 12600��һά�������У�ȫ������ģ���ߡ���right_results.dat ����Ϊ right_results_shift.dat

//����ָ��������'X' �� 'Y'������ͼ���а�ɫ���أ�����ֵΪ 255���ڸ������ϵ�ƽ��λ�ã����Խ������һϵ�д����������Ļ������滯
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

	double mean_result = std::accumulate(result.begin(), result.end(), 0.0) / result.size();//��begin��end�ۼӵ�0
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
// ������ 0 �� len(y_data) - 1 �ķ�Χ
std::vector<double> createRange(int size) {
	std::vector<double> range(size);
	for (int i = 0; i < size; ++i) {
		range[i] = (double)i;
	}
	return range;
}

// FFT ʵ��
std::vector<std::complex<double>> fft(const std::vector<double>& data) {
	size_t N = data.size();
	std::vector<std::complex<double>> result(N);

	// ���� FFTW �ƻ�
	fftw_complex *in, *out;
	in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
	out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);

	// ���������ݸ��Ƶ� FFTW ����������
	for (size_t i = 0; i < N; ++i) {
		in[i][0] = data[i]; // ʵ��
		in[i][1] = 0.0;     // �鲿
	}

	// ���� FFTW �ƻ�
	fftw_plan p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

	// ִ�� FFT
	fftw_execute(p);

	// �� FFT ������Ƶ� std::vector
	for (size_t i = 0; i < N; ++i) {
		result[i] = std::complex<double>(out[i][0], out[i][1]);
	}

	// ����
	fftw_destroy_plan(p);
	fftw_free(in);
	fftw_free(out);

	return result;
}

// ���� FFT Ƶ�ʷ���
std::vector<double> fft_freq(int N, double d = 1.0) {
	std::vector<double> frequencies(N);

	// ����Ƶ�ʷ���
	for (int k = 0; k < N; ++k) {
		frequencies[k] = k / (N * d);
	}

	// ����Ƶ�ʲ���
	for (int k = 0; k < N / 2; ++k) {
		frequencies[k] = k / (N * d);
		frequencies[N / 2 + k] = (k - N / 2) / (N * d);
	}

	return frequencies;
}
std::vector<double> filter_positive_frequencies(const std::vector<double>& frequencies) {
	std::vector<double> positive_frequencies;
	// ���� frequencies ������ɸѡ�Ǹ���Ƶ��
	for (const double& freq : frequencies) {
		if (freq >= 0) {
			positive_frequencies.push_back(freq);
		}
	}
	return positive_frequencies;
}
// ɸѡ����Ƶ�ʶ�Ӧ�� FFT ���
std::vector<std::complex<double>> filter_positive_fft_result(
	const std::vector<std::complex<double>>& fft_result,
	const std::vector<double>& frequencies
) {
	std::vector<std::complex<double>> positive_fft_result;
	// ����Ƶ�ʲ�ɸѡ����Ӧ�� FFT ���
	for (size_t i = 0; i < frequencies.size(); ++i) {
		if (frequencies[i] >= 0) {
			positive_fft_result.push_back(fft_result[i]);
		}
	}

	return positive_fft_result;
}
// ���㸴���ľ���ֵ
double abs_value(const std::complex<double>& c) {
	return std::abs(c);  // ʹ�ñ�׼�⺯���������ֵ
}

// �ҵ�����ֵ���ĸ���������
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
// �������Һ���
double sine_func(double x, const std::vector<double>& params) {
	return params[0] * std::sin(params[1] * x + params[2]) + params[3];
}
// ��С�������sin����
void curve_fit(const std::vector<double>& xData, const std::vector<double>& yData, double omega, double& A, double& phi, double& b) {
	int n = xData.size();
	Eigen::MatrixXd AMatrix(n, 3);
	Eigen::VectorXd bVector(n);

	// ����ϵ��������Ҳ�����
	for (int i = 0; i < n; i++) {
		AMatrix(i, 0) = std::sin(omega * xData[i]);
		AMatrix(i, 1) = std::cos(omega * xData[i]);
		AMatrix(i, 2) = 1.0;
		bVector(i) = yData[i];
	}

	// ʹ����С���˷�������ϲ���
	Eigen::Vector3d params = AMatrix.colPivHouseholderQr().solve(bVector);

	A = std::sqrt(params(0) * params(0) + params(1) * params(1));
	phi = std::atan2(params(1), params(0));
	b = params(2); // ƫ����b
}

// �������Լ������ĺ���
std::vector<double> linspace(double start, double end, int num_points) {
	std::vector<double> points(num_points);
	double step = (end - start) / (num_points - 1);
	for (int i = 0; i < num_points; ++i) {
		points[i] = start + i * step;
	}
	return points;
}
// ������������ȡ��ֵ�ķ���
int sign(double value) {
	return (value > 0) - (value < 0);
}

// �������ҵ� y_fit �еĵ�һ���ֲ���Сֵ
std::pair<size_t, double> find_first_min(const std::vector<double>& y_fit) {
	// ������
	std::vector<int> min_indices;

	if (y_fit.size() < 3) {
		// �����Լ�����ײ��
		throw std::invalid_argument("Input vector size is too small to calculate second-order differences.");
	}

	// ����һ�ײ��
	std::vector<double> diff1(y_fit.size() - 1);
	for (size_t i = 0; i < y_fit.size() - 1; ++i) {
		diff1[i] = y_fit[i + 1] - y_fit[i];
	}

	// ����һ�ײ�ֵķ���
	std::vector<int> sign1(y_fit.size() - 1);
	for (size_t i = 0; i < diff1.size(); ++i) {
		sign1[i] = (diff1[i] > 0) - (diff1[i] < 0); // 1 for positive, -1 for negative, 0 for zero
	}

	// ������ײ��
	std::vector<int> diff2(y_fit.size() - 2);
	for (size_t i = 0; i < sign1.size() - 1; ++i) {
		diff2[i] = sign1[i + 1] - sign1[i];
	}

	// �ҵ����ײ�ִ���0��λ��
	for (size_t i = 0; i < diff2.size(); ++i) {
		if (diff2[i] > 0) {
			size_t min_index = i + 1; // ��1�Բ�������ƫ��
			double y_min = y_fit[min_index];
			return { min_index, y_min }; // ���ؽ��
		}
	}
	// ���û���ҵ��ֲ���Сֵ
	return { static_cast<size_t>(-1), 0.0 }; // ʹ�� -1 ��ʾû���ҵ��ֲ���Сֵ
}

void fit_curve(const std::vector<double>& y_data, std::vector<double>& params, double& t_min, double& y_min) {
	std::vector<double> x_data = createRange(y_data.size());
	// ���÷�װ��FFT����
	std::vector<std::complex<double>> fft_result = fft(y_data);
	if (fft_result.empty()) {
		std::cout << "FFT result is empty." << std::endl;
	}
	std::vector<double> frequencies = fft_freq(y_data.size(), 1.0);
	if (fft_result.empty()) {
		std::cout << "frequencies is empty." << std::endl;
	}
	std::vector<double> positive_frequencies = filter_positive_frequencies(frequencies);
	// ɸѡ��Ƶ�ʶ�Ӧ�� FFT ���
	std::vector<std::complex<double>> positive_fft_result = filter_positive_fft_result(fft_result, frequencies);
	// �ҵ�������ֵ������
	size_t main_frequency_index = find_max_absolute_index(positive_fft_result);
	// ��ȡ������ֵ��Ӧ��Ƶ��
	double main_frequency = positive_frequencies[main_frequency_index];
	const double PI = 4 * std::atan(1.0);//��
	// �������ֵ����Сֵ
	double max_y = *std::max_element(y_data.begin(), y_data.end());
	double min_y = *std::min_element(y_data.begin(), y_data.end());
	// ���� A_initial
	double A_initial = (max_y - min_y) / 2;
	// ���� omega_initial
	double omega_initial = 2 * PI * main_frequency;
	// ���� C_initial
	double sum_y = std::accumulate(y_data.begin(), y_data.end(), 0.0);
	double C_initial = sum_y / y_data.size();
	// ��������
	double phi_initial = 0;

	// ִ�����
	curve_fit(x_data, y_data, omega_initial, A_initial, phi_initial, C_initial);
	params = { A_initial, omega_initial, phi_initial, C_initial };
	// �������Լ������
	std::vector<double> x_fit = linspace(0, y_data.size() - 1, 1000);
	// ���� y_fit
	std::vector<double> y_fit(x_fit.size());
	for (size_t i = 0; i < x_fit.size(); ++i) {
		y_fit[i] = sine_func(x_fit[i], params);
	}
	size_t t_min_index = -1;
	// ���ҵ�һ���ֲ���Сֵ
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
		std::cout << "û���ҵ���һ���ֲ���Сֵ" << std::endl;
	}
}

// ���Բ�ֵ����
double linearInterpolate(double x0, double y0, double x1, double y1, double x) {
	return y0 + (x - x0) * (y1 - y0) / (x1 - x0);
}

std::vector<double> interpolateShift(const std::vector<double>& arr, double shift, double shift_fillValue) {
	int n = arr.size();
	std::vector<double> x(n);
	std::iota(x.begin(), x.end(), 0); // ʹ�� std::iota ��� x

	std::vector<double> shiftedArr(n, shift_fillValue); // �������ֵ������� shiftedArr

	for (int i = 0; i < n; ++i) {
		double shiftedX = x[i] - shift;
		int leftIndex = static_cast<int>(std::floor(shiftedX));
		int rightIndex = leftIndex + 1;

		// ȷ������������Ч��Χ��
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
	// ����ģ������
	std::vector<double> temp_result = scan_and_visualize("./temp/temp_right_curve.png", 'Y');

	// ���ļ��м��ر��� right_results.dat ������
	std::string result_path = "F:/data/test/right_results.dat";
	std::string output_dir = "F:/data/results";

	//�Ӷ������ļ��ж�ȡ���ݲ������л�����ά������
	std::vector<std::vector<double>> right_results;
	{
		std::ifstream inputFile(result_path.c_str(), std::ios::binary);
		boost::archive::binary_iarchive archive(inputFile);
		archive >> right_results;
	}
	// ����Ҫȡ������������ n
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
			std::cout << "Result " << i + 1 << " ������ɡ�" << std::endl;
		}
		else {
			std::cout << "Result " << i + 1 << " û���ҵ����ʵļ�Сֵ�������˶��롣" << std::endl;
		}
	}
	// ������
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