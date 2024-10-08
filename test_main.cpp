#include <iostream>
#include <fstream>
#include <sstream> //�����ַ���������
#include <vector>
#include <string>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <numeric>  // ���� std::accumulate
#include <fftw3.h>
#include <Eigen/Dense>
#include <utility> // for std::pair

// ����ֵ�� -1 ����ͼƬΪ��, -2 ͼƬ��û�в�Ʒ, -3 ��λ��������ʧ��,
// -4 �Ҵ�С��ʧ��, -5 �����ƽ�ʧ��, -6 ����������±�Ե
// -7 �������ƽ�ʧ�� -8 ����ģ���ƷΪ�� -9 ������roi����ͼƬ��Χ -10 û���ҵ�peak��

// ����һ���ṹ�� Result
struct Result {
	double right_diff_max;
	double right_mse;
	double left_diff_max;
	double left_mse;
};
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
	result[0] = std::complex<double>(out[0][0], -out[0][1]);
	// �� FFT ������Ƶ� std::vector
	for (size_t i = 1; i < N; ++i) {
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
// �������ҵ� y_fit �еĵ�һ���ֲ����ֵ
std::pair<size_t, double> find_first_max(const std::vector<double>& y_fit) {
	// ������
	std::vector<int> max_indices;

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

	// �ҵ����ײ��С��0��λ��
	for (size_t i = 0; i < diff2.size(); ++i) {
		if (diff2[i] < 0) {
			size_t max_index = i + 1; // ��1�Բ�������ƫ��
			double y_max = y_fit[max_index];
			return { max_index, y_max }; // ���ؽ��
		}
	}
	// ���û���ҵ��ֲ����ֵ
	return { static_cast<size_t>(-1), 0.0 }; // ʹ�� -1 ��ʾû���ҵ��ֲ����ֵ
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

std::vector<double> remove_nan_inf(const std::vector<double>& data) {
	std::vector<double> cleaned_data;

	// �����ֵ������ NaN
	double mean = 0.0;
	int count = 0;
	for (double value : data) {
		if (std::isnan(value) || std::isinf(value)) {
			continue; // ���� NaN �� inf
		}
		mean += value;
		count++;
	}
	mean /= count; // �����ֵ

	// ��������
	for (double value : data) {
		if (std::isnan(value) || std::isinf(value)) {
			cleaned_data.push_back(mean); // ���Ϊ��ֵ
		}
		else {
			cleaned_data.push_back(value); // ����ԭֵ
		}
	}

	return cleaned_data;
}

std::pair<std::vector<double>, std::vector<double>> fit_curve(const std::vector<double>& y_data) {
	std::vector<double> x_data = createRange(y_data.size());
	// ���� NaN �� inf ֵ
	std::vector<double> y_data_new;
	y_data_new = remove_nan_inf(y_data);

	std::vector<std::complex<double>> fft_result = fft(y_data_new);
	if (fft_result.empty()) {
		std::cout << "FFT result is empty." << std::endl;
	}
	std::vector<double> frequencies = fft_freq(y_data_new.size(), 1.0);
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
	double max_y = *std::max_element(y_data_new.begin(), y_data_new.end());
	double min_y = *std::min_element(y_data_new.begin(), y_data_new.end());
	// ���� A_initial
	double A_initial = (max_y - min_y) / 2;
	// ���� omega_initial
	double omega_initial = 2 * PI * main_frequency;
	double phi_initial = 0;
	// ���� C_initial
	double sum_y = std::accumulate(y_data_new.begin(), y_data_new.end(), 0.0);
	double C_initial = sum_y / y_data_new.size();

	// ִ�����
	curve_fit(x_data, y_data_new, omega_initial, A_initial, phi_initial, C_initial);
	std::vector<double> params = { A_initial, omega_initial, phi_initial, C_initial };
	// ���ɴ� 0 �� len(y_data) - 1 �����Կռ�
	std::vector<double> x_fit = linspace(0.0, static_cast<double>(y_data_new.size() - 1), y_data_new.size() * 100);
	
	// ���� y_fit
	std::vector<double> y_fit(x_fit.size());
	for (size_t i = 0; i < x_fit.size(); ++i) {
		y_fit[i] = sine_func(x_fit[i], params);
	}
	return { x_fit, y_fit };
}
std::vector<double> normalize_and_scale(const std::vector<int>& arr) {
	if (arr.empty()) {
		throw std::invalid_argument("�������鲻��Ϊ��");
	}

	// �����ֵ
	double mean = std::accumulate(arr.begin(), arr.end(), 0.0) / arr.size();

	// ���Ļ�����
	std::vector<double> arr_centered(arr.size());
	std::transform(arr.begin(), arr.end(), arr_centered.begin(), [mean](int x) { // ��������
		return x - mean;
	});

	// ������Сֵ�����ֵ
	double min_val = *std::min_element(arr_centered.begin(), arr_centered.end());
	double max_val = *std::max_element(arr_centered.begin(), arr_centered.end());

	std::vector<double> arr_scaled(arr.size());

	// ���ŵ� [-1, 1]
	double scale = (max_val - min_val);
	for (size_t i = 0; i < arr_centered.size(); ++i) {
		if (scale != 0) {
			arr_scaled[i] = 2 * (arr_centered[i] - min_val) / scale - 1;
		}
		else {
			arr_scaled[i] = 0; // ���������ֵ��ͬ������Ϊ0
		}
	}

	return arr_scaled;
}

// �Զ�����������뺯�� - ������ Python round ��ͬ���߼�
int custom_round(double value) {
	double int_part;
	double frac_part = std::modf(value, &int_part);

	if (frac_part > 0.5) {
		return static_cast<int>(std::ceil(value));   // ����0.5������ȡ��
	}
	else if (frac_part < 0.5) {
		return static_cast<int>(std::floor(value));  // С��0.5������ȡ��
	}
	else {
		// frac_part == 0.5 ��������������������ż��
		if (static_cast<int>(int_part) % 2 == 0) {
			return static_cast<int>(int_part);  // �������������ż����������
		}
		else {
			return static_cast<int>(int_part) + 1; // �������������������1
		}
	}
}

std::vector<int> find_horizontal_light_center(const cv::Mat& image) {
	// ȷ��ͼ���Ƕ�ֵͼ��
	CV_Assert(image.type() == CV_8UC1); // ͼ������ӦΪ��ͨ��8λ�޷�������
	std::vector<int> result(image.rows, 0);  // ��ʼ�������������СΪͼ��ĸ߶�
	for (int y = 0; y < image.rows; ++y) {
		std::vector<int> x_coords;  // �洢���������x����

		for (int x = 0; x < image.cols; ++x) {
			if (image.at<uchar>(y, x) == 255) {  // �������㣨����ֵΪ255��
				x_coords.push_back(x);
			}
		}
		if (!x_coords.empty()) {
			// ���������ƽ��x����
			double sum = 0.0;
			for (int x : x_coords) {
				sum += x;
			}
			// ����ƽ��ֵ��ȡ��
			result[y] = custom_round(sum / x_coords.size());  // ʹ�� custom_round ����ȡ��,ģ��np.round��ȡ������
		}
	}
	return result;
}

void load_statistics_from_csv(const std::string& file_path,
	std::vector<double>& mean,
	std::vector<double>& std_dev,
	std::vector<double>& q1,
	std::vector<double>& q2,
	std::vector<double>& q3) {
	std::ifstream file(file_path);
	if (!file.is_open()) {
		throw std::runtime_error("�޷����ļ�: " + file_path);
	}
	std::string line;
	// ������ͷ
	if (std::getline(file, line)) {
		// ��ͷ��ȡ����δ����
	}
	while (std::getline(file, line)) {
		//ʹ�� std::stringstream �� line ת��Ϊһ���ַ��������� ss���������Է���ضԸ����ڵ��ַ������зָ�ʹ���
		std::stringstream ss(line);
		std::string value;

		// ��ȡ������ÿһ������
		if (std::getline(ss, value, ',')) {
			mean.push_back(std::stod(value));
		}
		if (std::getline(ss, value, ',')) {
			std_dev.push_back(std::stod(value));
		}
		if (std::getline(ss, value, ',')) {
			q1.push_back(std::stod(value));
		}
		if (std::getline(ss, value, ',')) {
			q2.push_back(std::stod(value));
		}
		if (std::getline(ss, value, ',')) {
			q3.push_back(std::stod(value));
		}
	}
}
//�жϵ������roi�Ƿ���ͼ���ڲ�
bool isRoiWithinImage(const cv::Rect& roi, int imageWidth, int imageHeight)
{
	int x = roi.x;
	int y = roi.y;
	int w = roi.width;
	int h = roi.height;

	// ������Ͻ��Ƿ���ͼ���ڲ�
	if (x < 0 || y < 0)
		return false;

	// ������½��Ƿ���ͼ���ڲ�
	if (x + w > imageWidth || y + h > imageHeight)
		return false;

	// ��� h �� w �Ƿ�����
	if (h < 0 || w < 0)
		return false;

	return true;
}

//���ϵ��ҷ�ֵ�ĺ���
template <class T>
std::vector<int> find_peaks(const std::vector<T> &src, int distance)
{
	int length = src.size();
	if (length <= 1) return std::vector<int>();
	//��������,-1��ʾ�½�
	std::vector<int> sign(length, -1);
	//�洢һ�ײ��
	std::vector<T> difference(length, 0);
	//�洢��⵽�ķ�ֵ������
	std::vector<int> temp_out;

	//��������Ԫ��֮��Ĳ�ֵ
	adjacent_difference(src.begin(), src.end(), difference.begin());
	//�Ƴ� difference �ĵ�һ������Ӧ��Դ�����ĵ�һ��Ԫ�ر仯�������һ��Ԫ�أ�û������ֵ���õģ�
	difference.erase(difference.begin());
	difference.pop_back();

	//һ�ײ��Ϊ�Ǹ�ֵ��λ�ñ��Ϊ 1����ʾ��������
	for (int i = 0; i < difference.size(); ++i) {
		if (difference[i] >= 0) sign[i] = 1;
	}
	//Ѱ���������½�֮��Ĺյ㲢����������ӵ� temp_out ��
	for (int j = 1; j < length - 1; ++j)
	{
		int  diff = sign[j] - sign[j - 1];
		if (diff < 0) {
			temp_out.push_back(j);
		}
	}
	//���û���ҵ���ֵ�� distance Ϊ 0����ֱ�ӷ��� temp_out
	if (temp_out.size() == 0 || distance == 0) return temp_out;
	//���ҵ��ķ�ֵ�����Ӵ�С����
	std::sort(temp_out.begin(), temp_out.end(), [&src](int a, int b) {
		return (src[a] > src[b]);
	});
	//�洢���յķ�ֵ����
	std::vector<int> ans;
	//����һ�� unordered_map ���ڸ������ų����������Ա��ⷵ�����ڵķ�ֵ
	std::unordered_map<int, int> except;
	//���㵱ǰ��ֵ����ߺ��ұߵķ�Χ��distance),�ų���ֵ
	for (auto it : temp_out) {
		if (!except.count(it))//��������ų���Χ��
		{
			ans.push_back(it);
			//����
			int left = it - distance > 0 ? it - distance : 0;
			int right = it + distance > length - 1 ? length - 1 : it + distance;
			for (int i = left; i <= right; ++i)
				++except[i];
		}
	}
	std::sort(ans.begin(), ans.end());
	return ans;
}
std::vector<std::vector<double>> filter_peaks_by_threshold(const std::vector<std::vector<double>>& peaks, double threshold = 5.0) {
	// ������ȡ x ֵ
	std::vector<double> x_values;

	for (const auto& peak : peaks) {
		x_values.push_back(peak[0]); // �ٶ� x ֵ��ÿ����ֵ����ĵ�һ��Ԫ��
	}

	// ɸѡ������Ʒ�ֵ��
	size_t max_len = 0, max_segment_start = 0;
	size_t current_len = 1, current_start = 0;

	for (size_t i = 1; i < x_values.size(); ++i) {
		if (std::abs(x_values[i] - x_values[i - 1]) <= threshold) {
			current_len++;
		}
		else {
			if (current_len > max_len) {
				max_len = current_len;
				max_segment_start = current_start;
			}
			current_start = i;
			current_len = 1;
		}
	}

	// �������һ��
	if (current_len > max_len) {
		max_len = current_len;
		max_segment_start = current_start;
	}

	// ������λ��
	std::vector<double> segment(x_values.begin() + max_segment_start, x_values.begin() + max_segment_start + max_len);
	std::sort(segment.begin(), segment.end());
	double median_x;

	if (segment.size() % 2 == 0) {
		median_x = (segment[segment.size() / 2 - 1] + segment[segment.size() / 2]) / 2.0;
	}
	else {
		median_x = segment[segment.size() / 2];
	}

	// ɸѡ�����������ķ�ֵ
	std::vector<std::vector<double>> filtered_peaks;
	for (const auto& peak : peaks) {
		if (std::abs(peak[0] - median_x) <= threshold) {
			filtered_peaks.push_back(peak);
		}
	}
	return filtered_peaks;
}
int roi_extractor(const std::string& input_path, const std::string& product_temp_path,int top_shift_down,int bottom_shift_up, const std::string& output_file_path, cv::Mat& roi_img_left_result, cv::Mat& roi_img_right_result) {
	cv::Mat input_image = cv::imread(input_path, cv::IMREAD_GRAYSCALE);
	if (input_image.empty()) {
		return -1;  // ���ش�����
	}
	// ��ȡͼ��ĸ߶ȺͿ��
	double image_height = input_image.rows; // ͼ��߶�
	double image_width = input_image.cols;   // ͼ����
	cv::Mat product_temp_image = cv::imread(product_temp_path, cv::IMREAD_GRAYSCALE);
	if (product_temp_image.empty()) {
		return -8;  // ���ش�����
	}
	//ȥë��
	cv::Mat clear_img;
	cv::Mat clear_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(19, 19));
	cv::morphologyEx(input_image, clear_img, cv::MORPH_CLOSE, clear_kernel);
	// ��������λ ROI
	cv::Mat down_sample_img;
	cv::resize(clear_img, down_sample_img, cv::Size(), 0.25, 0.25);
	cv::Mat resultImg;
	cv::matchTemplate(down_sample_img, product_temp_image, resultImg, cv::TM_CCORR_NORMED);//�����ϵ��ƥ�䷨(���ƥ��1) // cv::TM_CCOEFF_NORMED
	double maxValue = 0;
	cv::Point  maxLoc;
	cv::minMaxLoc(resultImg, nullptr, &maxValue, nullptr, &maxLoc);
	// ���ƥ����
	if (maxValue < 0.6)
	{
		return -2;
	}
	//��λ��������ƷROI��ԭ��ԭͼ
	cv::Rect productRoi(maxLoc.x * 4, maxLoc.y * 4, product_temp_image.cols * 4, product_temp_image.rows * 4);
	//����product_Roi������
	int x = productRoi.x;
	int y = productRoi.y + top_shift_down;
	int w = productRoi.width;
	int h = productRoi.height - (top_shift_down + bottom_shift_up);
	cv::Rect adjustedRoi(x, y, w, h);
	// ���Ҷ�ͼ��ת��Ϊ��ɫͼ��
	cv::Mat input_img_color;
	cv::cvtColor(clear_img, input_img_color, cv::COLOR_GRAY2BGR);
	cv::Point top_left(adjustedRoi.x, adjustedRoi.y);
	cv::Point bottom_right(adjustedRoi.x + adjustedRoi.width, adjustedRoi.y + adjustedRoi.height);
	cv::rectangle(input_img_color, top_left, bottom_right, cv::Scalar(0, 255, 0), 2);
	// ������ͼ������ע�͵��ˣ��ɸ����������ã�
	// cv::imwrite("./clear_img_withROI.png", input_img_color);
	if (!isRoiWithinImage(adjustedRoi, image_width, image_height))
	{
		return -9;
	}
	// ��ֵ��
	cv::Mat binary_image;
	cv::threshold(clear_img, binary_image, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
	// �õ�����
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(binary_image, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

	// ��ʾ������ֵͼ�������
	cv::Mat contours_img = cv::Mat::zeros(binary_image.size(), CV_8U);
	for (const auto& contour : contours)
	{
		cv::drawContours(contours_img, std::vector<std::vector<cv::Point>>{contour}, -1, 255, 1);
		//std::vector<std::vector<cv::Point>>{contour} ����ǰ����ת��Ϊһ�����������������������Ա� cv::drawContours �������Խ���
		//-1 ��ʾ�������е�������
		//255 ��ʾ��������ɫΪ��ɫ��
		//1 ��ʾ�������߿�Ϊ 1 ���ء�
	}

	// ������ͼ������ȡ������� ROI ����
	cv::Mat roi_img = contours_img(adjustedRoi);
	double mid_point = roi_img.cols / 2;
	cv::Mat roi_img_right = roi_img(cv::Rect(mid_point, 0, roi_img.cols - mid_point, roi_img.rows));
	cv::Mat roi_img_left = roi_img(cv::Rect(0, 0, mid_point, roi_img.rows));


	//�����Ҳ��Ե
	std::vector<int> right_roi_horizontal_centers = find_horizontal_light_center(roi_img_right);
	std::vector<int> peaks_right = find_peaks<int>(right_roi_horizontal_centers, 160);
	std::vector<std::vector<double>> right_peak_points;
	// ���ϲ�������ݽṹ
	for (int peak : peaks_right) {
		if (peak >= 0 && peak < right_roi_horizontal_centers.size()) {
			// ����һ�����У����������
			std::vector<double> point;
			point.push_back(right_roi_horizontal_centers[peak]); // ����ֵ
			point.push_back(peak); // ��ֵ����
			right_peak_points.push_back(point); // ��ӵ����
		}
	}
	//// ����ϲ���Ľ��
	//for (const auto& point : right_peak_points) {
	//	std::cout << "(" << point[0] << ", " << point[1] << ")\n";
	//}
	if(right_peak_points.empty()) {
		return -10;
	}
	right_peak_points = filter_peaks_by_threshold(right_peak_points, 5.0);
	// ��ȡ y_start �� y_end
	double y_start = static_cast<int>(right_peak_points[0][1]); // ��һ����ֵ�� Y ֵ
	double y_end = static_cast<int>(right_peak_points.back()[1]); // ���һ����ֵ�� Y ֵ

	// ȷ�� y_start �� y_end ��ͼ��Χ��
	if (y_start < 0) y_start = 0;
	if (y_end >= roi_img_right.rows) y_end = roi_img_right.rows - 1;

	// ���� y_start �� y_end ��ȡ ROI
	cv::Rect right_roi(0, y_start, roi_img_right.cols, (y_end - y_start + 1)); // ���� ROI ����
	cv::Mat new_roi_img_right = roi_img_right(right_roi); // ��ȡ ROI

	//// ���ӻ�ͼ��: ���Ҷ�ͼ��ת��Ϊ BGR ��ɫͼ��
	//cv::Mat right_marked_img;
	//cv::cvtColor(roi_img_right, right_marked_img, cv::COLOR_GRAY2BGR);

	//// ������ֵ������ԲȦ���
	//for (const auto& peak : right_peak_points) {
	//	// ����ֵת��Ϊ��������
	//	int x = static_cast<int>(peak[0]);
	//	int y = static_cast<int>(peak[1]);

	//	// ����ԲȦ���뾶Ϊ 2����ɫΪ��ɫ(255, 0, 255)�����Ϊ -1�����ԲȦ��
	//	cv::circle(right_marked_img, cv::Point(x, y), 2, cv::Scalar(255, 0, 255), cv::FILLED);
	//}
	
	roi_img_right_result = new_roi_img_right;
	cv::imshow("Right ROI Image", roi_img_right_result);
	cv::waitKey(0);
	/*cv::imshow("Right ROI Image", roi_img_right_result);
	cv::waitKey(0);*/
	//��������Ե
	std::vector<int> left_roi_horizontal_centers = find_horizontal_light_center(roi_img_left);
	// ��ת����
	std::vector<int> inverted_data(left_roi_horizontal_centers.size());
	std::transform(left_roi_horizontal_centers.begin(), left_roi_horizontal_centers.end(), inverted_data.begin(), [](double x) { return -x; });
	std::vector<int> peaks_left = find_peaks<int>(inverted_data, 160);
	std::vector<std::vector<double>> left_peak_points;
	// ���ϲ�������ݽṹ
	for (int peak : peaks_left) {
		if (peak >= 0 && peak < left_roi_horizontal_centers.size()) {
			// ����һ�����У����������
			std::vector<double> point;
			point.push_back(left_roi_horizontal_centers[peak]); // ����ֵ
			point.push_back(peak); // ��ֵ����
			left_peak_points.push_back(point); // ��ӵ����
		}
	}
	//// ����ϲ���Ľ��
	//for (const auto& point : right_peak_points) {
	//	std::cout << "(" << point[0] << ", " << point[1] << ")\n";
	//}
	if (left_peak_points.empty()) {
		return -10;
	}
	left_peak_points = filter_peaks_by_threshold(left_peak_points, 5.0);
	// ��ȡ y_start �� y_end
	y_start = static_cast<int>(left_peak_points[0][1]); // ��һ����ֵ�� Y ֵ
	y_end = static_cast<int>(left_peak_points.back()[1]); // ���һ����ֵ�� Y ֵ

	// ȷ�� y_start �� y_end ��ͼ��Χ��
	if (y_start < 0) y_start = 0;
	if (y_end >= roi_img_left.rows) y_end = roi_img_left.rows - 1;

	// ���� y_start �� y_end ��ȡ ROI
	cv::Rect left_roi(0, y_start, roi_img_left.cols, (y_end - y_start + 1)); // ���� ROI ����
	cv::Mat new_roi_img_left = roi_img_left(left_roi); // ��ȡ ROI

	roi_img_left_result = new_roi_img_left;
	/*cv::imshow("Left ROI Image", roi_img_left_result);
	cv::waitKey(0);*/

	return 0;
}
// ����һ��ͼƬ��������ΪResult�ṹ��
int process_one_image(const std::string& input_path,
	const std::string& temp_left_curve_path,
	const std::string& temp_right_curve_path,
	const std::string& product_temp_path,
	int top_shift_down,
	int bottom_shift_up,
	Result& result,
	const std::string& output_path,
	const std::string& left_statistics_file_path,
	const std::string& right_statistics_file_path) {
	/*cv::Mat input_image = cv::imread(input_path, cv::IMREAD_GRAYSCALE);*/
	// ��������Ŀ¼�е�������
	//if (input_image.empty()) {
	//	return -1;  // ���ش�����
	//}
	int AriStatus = 0;
	std::vector<double> mean_left, std_dev_left, q1_left, q2_left, q3_left;
	std::vector<double> mean_right, std_dev_right, q1_right, q2_right, q3_right;
	load_statistics_from_csv(left_statistics_file_path, mean_left, std_dev_left, q1_left, q2_left, q3_left);
	load_statistics_from_csv(right_statistics_file_path, mean_right, std_dev_right, q1_right, q2_right, q3_right);
	// �������ģ������
	cv::Mat temp_left_image = cv::imread(temp_left_curve_path, cv::IMREAD_GRAYSCALE);
	std::vector<int> temp_left_edge_contour = find_horizontal_light_center(temp_left_image);
	// �����Ҳ�ģ������
	cv::Mat temp_right_image = cv::imread(temp_right_curve_path, cv::IMREAD_GRAYSCALE);
	std::vector<int> temp_right_edge_contour = find_horizontal_light_center(temp_right_image);
	// ������������������й�һ��
	std::vector<double> temp_left_edge_contour_new = normalize_and_scale(temp_left_edge_contour);
	std::vector<double> temp_right_edge_contour_new = normalize_and_scale(temp_right_edge_contour);
	// �����ģ�������
	//fit_curve�õ�������ƫ���е��
	std::pair<std::vector<double>, std::vector<double>> left_fit_result = fit_curve(temp_left_edge_contour_new);
	std::vector<double> temp_left_x_fit = left_fit_result.first;
	std::vector<double> temp_left_y_fit = left_fit_result.second;
	// �ҵ����������ߵĵ�һ������ֵ
	std::pair<int, double> temp_left_max_result = find_first_max(temp_left_y_fit);
	int temp_t_max_index = -1;
	temp_t_max_index = temp_left_max_result.first;
	double temp_y_max = temp_left_max_result.second;

	// ���Ҳ�ģ�������
	std::pair<std::vector<double>, std::vector<double>> right_fit_result = fit_curve(temp_right_edge_contour_new);
	std::vector<double> temp_right_x_fit = right_fit_result.first;
	std::vector<double> temp_right_y_fit = right_fit_result.second;

	// �ҵ��Ҳ�������ߵĵ�һ����Сֵ
	std::pair<int, double> temp_right_min_result = find_first_min(temp_right_y_fit);
	int temp_t_min_index = -1;
	temp_t_min_index = temp_right_min_result.first;
	double temp_y_min = temp_right_min_result.second;
	cv::Mat roi_img_left_image;
	cv::Mat roi_img_right_image;
	AriStatus = roi_extractor(input_path, product_temp_path,top_shift_down, bottom_shift_up,output_path, roi_img_left_image, roi_img_right_image);
	std::vector<int> right_edge_contour = find_horizontal_light_center(roi_img_right_image);
	if (right_edge_contour.size() < 160) {
		// �������쳣״̬
		result.right_diff_max = -1;
		result.right_mse = -1;
		std::cout << "��ȡʧ��" << std::endl;
	}
	else {
		std::vector<double> right_edge_contour_new = normalize_and_scale(right_edge_contour);
		std::pair<std::vector<double>, std::vector<double>> right_result = fit_curve(right_edge_contour_new);
		std::vector<double> right_x_fit = right_result.first;
		std::vector<double> right_y_fit = right_result.second;
		//�ҵ��Ҳ�������ߵĵ�һ����Сֵ
		std::pair<int, double> right_min_result = find_first_min(right_y_fit);
		int t_min_index = -1;
		t_min_index = temp_right_min_result.first;
		double y_min = temp_right_min_result.second;
		if (t_min_index != -1 && temp_t_min_index != -1) {
			double time_shift = temp_right_x_fit[temp_t_min_index] - right_x_fit[t_min_index];
		}
	}
	return AriStatus;
}

int main() {
	// ��Ʒ��ʱ·��
	std::string product_temp_path = "F:/PythonProject/screw/temp/906_temp.png";

	// ��λ�ò�Ʒ֮�󣬵���ROI������
	int top_shift_down = 400;
	int bottom_shift_up = 400;
	// ����ͼƬ
	std::string input_path = "F:/data/906_NG/906-1_20240725-143044/page_60.bmp";
	// ���ģ������
	std::string temp_left_curve_path = "F:/PythonProject/screw/temp/906_temp_left_curve.png";
	// �Ҳ�ģ������
	std::string temp_right_curve_path = "F:/PythonProject/screw/temp/906_temp_right_curve.png";
	// ����������
	Result result = { 0.0, 0.0, 0.0, 0.0 };
	std::string output_path = "C:/Users/25451/Desktop/result";
	// ������������ͳ������
	std::string right_statistics_file_path = "F:/PythonProject/screw/right_statistical_models.csv";
	std::string left_statistics_file_path = "F:/PythonProject/screw/left_statistical_models.csv";
	int AriStatus = process_one_image(input_path, temp_left_curve_path, temp_right_curve_path, product_temp_path,
		top_shift_down, bottom_shift_up, result, output_path,
		left_statistics_file_path, right_statistics_file_path);
	return 0;
}