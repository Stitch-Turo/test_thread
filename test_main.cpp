#include <iostream>
#include <fstream>
#include <sstream> //用于字符串流处理
#include <vector>
#include <string>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <numeric>  // 包含 std::accumulate
#include <fftw3.h>
#include <Eigen/Dense>
#include <utility> // for std::pair

// 返回值： -1 输入图片为空, -2 图片中没有产品, -3 定位两侧螺纹失败,
// -4 找大径小径失败, -5 找螺纹角失败, -6 拟合螺纹上下边缘
// -7 计算螺纹角失败 -8 读入模板产品为空 -9 调整的roi超出图片范围 -10 没有找到peak点

// 定义一个结构体 Result
struct Result {
	double right_diff_max;
	double right_mse;
	double left_diff_max;
	double left_mse;
};
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
	result[0] = std::complex<double>(out[0][0], -out[0][1]);
	// 将 FFT 结果复制到 std::vector
	for (size_t i = 1; i < N; ++i) {
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
// 函数：找到 y_fit 中的第一个局部最大值
std::pair<size_t, double> find_first_max(const std::vector<double>& y_fit) {
	// 计算差分
	std::vector<int> max_indices;

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

	// 找到二阶差分小于0的位置
	for (size_t i = 0; i < diff2.size(); ++i) {
		if (diff2[i] < 0) {
			size_t max_index = i + 1; // 加1以补偿索引偏移
			double y_max = y_fit[max_index];
			return { max_index, y_max }; // 返回结果
		}
	}
	// 如果没有找到局部最大值
	return { static_cast<size_t>(-1), 0.0 }; // 使用 -1 表示没有找到局部最大值
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

std::vector<double> remove_nan_inf(const std::vector<double>& data) {
	std::vector<double> cleaned_data;

	// 计算均值，忽略 NaN
	double mean = 0.0;
	int count = 0;
	for (double value : data) {
		if (std::isnan(value) || std::isinf(value)) {
			continue; // 忽略 NaN 和 inf
		}
		mean += value;
		count++;
	}
	mean /= count; // 计算均值

	// 修正数据
	for (double value : data) {
		if (std::isnan(value) || std::isinf(value)) {
			cleaned_data.push_back(mean); // 替代为均值
		}
		else {
			cleaned_data.push_back(value); // 保留原值
		}
	}

	return cleaned_data;
}

std::pair<std::vector<double>, std::vector<double>> fit_curve(const std::vector<double>& y_data) {
	std::vector<double> x_data = createRange(y_data.size());
	// 修正 NaN 和 inf 值
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
	// 筛选正频率对应的 FFT 结果
	std::vector<std::complex<double>> positive_fft_result = filter_positive_fft_result(fft_result, frequencies);
	// 找到最大绝对值的索引
	size_t main_frequency_index = find_max_absolute_index(positive_fft_result);
	// 获取最大绝对值对应的频率
	double main_frequency = positive_frequencies[main_frequency_index];
	const double PI = 4 * std::atan(1.0);//Π
	// 计算最大值和最小值
	double max_y = *std::max_element(y_data_new.begin(), y_data_new.end());
	double min_y = *std::min_element(y_data_new.begin(), y_data_new.end());
	// 计算 A_initial
	double A_initial = (max_y - min_y) / 2;
	// 计算 omega_initial
	double omega_initial = 2 * PI * main_frequency;
	double phi_initial = 0;
	// 计算 C_initial
	double sum_y = std::accumulate(y_data_new.begin(), y_data_new.end(), 0.0);
	double C_initial = sum_y / y_data_new.size();

	// 执行拟合
	curve_fit(x_data, y_data_new, omega_initial, A_initial, phi_initial, C_initial);
	std::vector<double> params = { A_initial, omega_initial, phi_initial, C_initial };
	// 生成从 0 到 len(y_data) - 1 的线性空间
	std::vector<double> x_fit = linspace(0.0, static_cast<double>(y_data_new.size() - 1), y_data_new.size() * 100);
	
	// 计算 y_fit
	std::vector<double> y_fit(x_fit.size());
	for (size_t i = 0; i < x_fit.size(); ++i) {
		y_fit[i] = sine_func(x_fit[i], params);
	}
	return { x_fit, y_fit };
}
std::vector<double> normalize_and_scale(const std::vector<int>& arr) {
	if (arr.empty()) {
		throw std::invalid_argument("输入数组不能为空");
	}

	// 计算均值
	double mean = std::accumulate(arr.begin(), arr.end(), 0.0) / arr.size();

	// 中心化数组
	std::vector<double> arr_centered(arr.size());
	std::transform(arr.begin(), arr.end(), arr_centered.begin(), [mean](int x) { // 修正类型
		return x - mean;
	});

	// 计算最小值和最大值
	double min_val = *std::min_element(arr_centered.begin(), arr_centered.end());
	double max_val = *std::max_element(arr_centered.begin(), arr_centered.end());

	std::vector<double> arr_scaled(arr.size());

	// 缩放到 [-1, 1]
	double scale = (max_val - min_val);
	for (size_t i = 0; i < arr_centered.size(); ++i) {
		if (scale != 0) {
			arr_scaled[i] = 2 * (arr_centered[i] - min_val) / scale - 1;
		}
		else {
			arr_scaled[i] = 0; // 如果所有数值相同，则结果为0
		}
	}

	return arr_scaled;
}

// 自定义的四舍五入函数 - 处理与 Python round 相同的逻辑
int custom_round(double value) {
	double int_part;
	double frac_part = std::modf(value, &int_part);

	if (frac_part > 0.5) {
		return static_cast<int>(std::ceil(value));   // 大于0.5，向上取整
	}
	else if (frac_part < 0.5) {
		return static_cast<int>(std::floor(value));  // 小于0.5，向下取整
	}
	else {
		// frac_part == 0.5 的情况，返回离它最近的偶数
		if (static_cast<int>(int_part) % 2 == 0) {
			return static_cast<int>(int_part);  // 如果整数部分是偶数，返回它
		}
		else {
			return static_cast<int>(int_part) + 1; // 如果是奇数，返回它加1
		}
	}
}

std::vector<int> find_horizontal_light_center(const cv::Mat& image) {
	// 确保图像是二值图像
	CV_Assert(image.type() == CV_8UC1); // 图像类型应为单通道8位无符号整型
	std::vector<int> result(image.rows, 0);  // 初始化结果向量，大小为图像的高度
	for (int y = 0; y < image.rows; ++y) {
		std::vector<int> x_coords;  // 存储该行亮点的x坐标

		for (int x = 0; x < image.cols; ++x) {
			if (image.at<uchar>(y, x) == 255) {  // 查找亮点（像素值为255）
				x_coords.push_back(x);
			}
		}
		if (!x_coords.empty()) {
			// 计算亮点的平均x坐标
			double sum = 0.0;
			for (int x : x_coords) {
				sum += x;
			}
			// 计算平均值并取整
			result[y] = custom_round(sum / x_coords.size());  // 使用 custom_round 进行取整,模仿np.round的取整策略
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
		throw std::runtime_error("无法打开文件: " + file_path);
	}
	std::string line;
	// 跳过表头
	if (std::getline(file, line)) {
		// 表头读取，但未处理
	}
	while (std::getline(file, line)) {
		//使用 std::stringstream 将 line 转换为一个字符串流对象 ss。这样可以方便地对该行内的字符串进行分割和处理
		std::stringstream ss(line);
		std::string value;

		// 读取并解析每一列数据
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
//判断调整后的roi是否在图像内部
bool isRoiWithinImage(const cv::Rect& roi, int imageWidth, int imageHeight)
{
	int x = roi.x;
	int y = roi.y;
	int w = roi.width;
	int h = roi.height;

	// 检查左上角是否在图像内部
	if (x < 0 || y < 0)
		return false;

	// 检查右下角是否在图像内部
	if (x + w > imageWidth || y + h > imageHeight)
		return false;

	// 检查 h 和 w 是否正常
	if (h < 0 || w < 0)
		return false;

	return true;
}

//网上的找峰值的函数
template <class T>
std::vector<int> find_peaks(const std::vector<T> &src, int distance)
{
	int length = src.size();
	if (length <= 1) return std::vector<int>();
	//符号向量,-1表示下降
	std::vector<int> sign(length, -1);
	//存储一阶差分
	std::vector<T> difference(length, 0);
	//存储检测到的峰值的索引
	std::vector<int> temp_out;

	//计算相邻元素之间的差值
	adjacent_difference(src.begin(), src.end(), difference.begin());
	//移除 difference 的第一个（对应于源向量的第一个元素变化）和最后一个元素（没有相邻值可用的）
	difference.erase(difference.begin());
	difference.pop_back();

	//一阶差分为非负值的位置标记为 1，表示上升部分
	for (int i = 0; i < difference.size(); ++i) {
		if (difference[i] >= 0) sign[i] = 1;
	}
	//寻找上升到下降之间的拐点并将其索引添加到 temp_out 中
	for (int j = 1; j < length - 1; ++j)
	{
		int  diff = sign[j] - sign[j - 1];
		if (diff < 0) {
			temp_out.push_back(j);
		}
	}
	//如果没有找到峰值或 distance 为 0，则直接返回 temp_out
	if (temp_out.size() == 0 || distance == 0) return temp_out;
	//将找到的峰值索引从大到小排序
	std::sort(temp_out.begin(), temp_out.end(), [&src](int a, int b) {
		return (src[a] > src[b]);
	});
	//存储最终的峰值索引
	std::vector<int> ans;
	//定义一个 unordered_map 用于跟踪需排除的索引，以避免返回相邻的峰值
	std::unordered_map<int, int> except;
	//计算当前峰值的左边和右边的范围（distance),排除峰值
	for (auto it : temp_out) {
		if (!except.count(it))//如果不在排除范围内
		{
			ans.push_back(it);
			//更新
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
	// 用于提取 x 值
	std::vector<double> x_values;

	for (const auto& peak : peaks) {
		x_values.push_back(peak[0]); // 假定 x 值在每个峰值数组的第一个元素
	}

	// 筛选最长的相似峰值段
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

	// 处理最后一段
	if (current_len > max_len) {
		max_len = current_len;
		max_segment_start = current_start;
	}

	// 计算中位数
	std::vector<double> segment(x_values.begin() + max_segment_start, x_values.begin() + max_segment_start + max_len);
	std::sort(segment.begin(), segment.end());
	double median_x;

	if (segment.size() % 2 == 0) {
		median_x = (segment[segment.size() / 2 - 1] + segment[segment.size() / 2]) / 2.0;
	}
	else {
		median_x = segment[segment.size() / 2];
	}

	// 筛选出满足条件的峰值
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
		return -1;  // 返回错误码
	}
	// 获取图像的高度和宽度
	double image_height = input_image.rows; // 图像高度
	double image_width = input_image.cols;   // 图像宽度
	cv::Mat product_temp_image = cv::imread(product_temp_path, cv::IMREAD_GRAYSCALE);
	if (product_temp_image.empty()) {
		return -8;  // 返回错误码
	}
	//去毛刺
	cv::Mat clear_img;
	cv::Mat clear_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(19, 19));
	cv::morphologyEx(input_image, clear_img, cv::MORPH_CLOSE, clear_kernel);
	// 降采样定位 ROI
	cv::Mat down_sample_img;
	cv::resize(clear_img, down_sample_img, cv::Size(), 0.25, 0.25);
	cv::Mat resultImg;
	cv::matchTemplate(down_sample_img, product_temp_image, resultImg, cv::TM_CCORR_NORMED);//化相关系数匹配法(最好匹配1) // cv::TM_CCOEFF_NORMED
	double maxValue = 0;
	cv::Point  maxLoc;
	cv::minMaxLoc(resultImg, nullptr, &maxValue, nullptr, &maxLoc);
	// 检查匹配结果
	if (maxValue < 0.6)
	{
		return -2;
	}
	//定位到整个产品ROI还原到原图
	cv::Rect productRoi(maxLoc.x * 4, maxLoc.y * 4, product_temp_image.cols * 4, product_temp_image.rows * 4);
	//调整product_Roi的区域
	int x = productRoi.x;
	int y = productRoi.y + top_shift_down;
	int w = productRoi.width;
	int h = productRoi.height - (top_shift_down + bottom_shift_up);
	cv::Rect adjustedRoi(x, y, w, h);
	// 将灰度图像转换为彩色图像
	cv::Mat input_img_color;
	cv::cvtColor(clear_img, input_img_color, cv::COLOR_GRAY2BGR);
	cv::Point top_left(adjustedRoi.x, adjustedRoi.y);
	cv::Point bottom_right(adjustedRoi.x + adjustedRoi.width, adjustedRoi.y + adjustedRoi.height);
	cv::rectangle(input_img_color, top_left, bottom_right, cv::Scalar(0, 255, 0), 2);
	// 保存结果图像（这里注释掉了，可根据需求启用）
	// cv::imwrite("./clear_img_withROI.png", input_img_color);
	if (!isRoiWithinImage(adjustedRoi, image_width, image_height))
	{
		return -9;
	}
	// 二值化
	cv::Mat binary_image;
	cv::threshold(clear_img, binary_image, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
	// 得到轮廓
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(binary_image, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

	// 显示整个二值图像的轮廓
	cv::Mat contours_img = cv::Mat::zeros(binary_image.size(), CV_8U);
	for (const auto& contour : contours)
	{
		cv::drawContours(contours_img, std::vector<std::vector<cv::Point>>{contour}, -1, 255, 1);
		//std::vector<std::vector<cv::Point>>{contour} 将当前轮廓转换为一个包含单个轮廓的向量，以便 cv::drawContours 函数可以接受
		//-1 表示绘制所有的轮廓。
		//255 表示轮廓的颜色为白色。
		//1 表示轮廓的线宽为 1 像素。
	}

	// 从轮廓图像当中提取调整后的 ROI 区域
	cv::Mat roi_img = contours_img(adjustedRoi);
	double mid_point = roi_img.cols / 2;
	cv::Mat roi_img_right = roi_img(cv::Rect(mid_point, 0, roi_img.cols - mid_point, roi_img.rows));
	cv::Mat roi_img_left = roi_img(cv::Rect(0, 0, mid_point, roi_img.rows));


	//处理右侧边缘
	std::vector<int> right_roi_horizontal_centers = find_horizontal_light_center(roi_img_right);
	std::vector<int> peaks_right = find_peaks<int>(right_roi_horizontal_centers, 160);
	std::vector<std::vector<double>> right_peak_points;
	// 填充合并后的数据结构
	for (int peak : peaks_right) {
		if (peak >= 0 && peak < right_roi_horizontal_centers.size()) {
			// 创建一个新行，并填充数据
			std::vector<double> point;
			point.push_back(right_roi_horizontal_centers[peak]); // 数据值
			point.push_back(peak); // 峰值索引
			right_peak_points.push_back(point); // 添加到结果
		}
	}
	//// 输出合并后的结果
	//for (const auto& point : right_peak_points) {
	//	std::cout << "(" << point[0] << ", " << point[1] << ")\n";
	//}
	if(right_peak_points.empty()) {
		return -10;
	}
	right_peak_points = filter_peaks_by_threshold(right_peak_points, 5.0);
	// 提取 y_start 和 y_end
	double y_start = static_cast<int>(right_peak_points[0][1]); // 第一个峰值的 Y 值
	double y_end = static_cast<int>(right_peak_points.back()[1]); // 最后一个峰值的 Y 值

	// 确保 y_start 和 y_end 在图像范围内
	if (y_start < 0) y_start = 0;
	if (y_end >= roi_img_right.rows) y_end = roi_img_right.rows - 1;

	// 根据 y_start 和 y_end 提取 ROI
	cv::Rect right_roi(0, y_start, roi_img_right.cols, (y_end - y_start + 1)); // 定义 ROI 区域
	cv::Mat new_roi_img_right = roi_img_right(right_roi); // 提取 ROI

	//// 可视化图像: 将灰度图像转换为 BGR 彩色图像
	//cv::Mat right_marked_img;
	//cv::cvtColor(roi_img_right, right_marked_img, cv::COLOR_GRAY2BGR);

	//// 遍历峰值，绘制圆圈标记
	//for (const auto& peak : right_peak_points) {
	//	// 将峰值转换为整数坐标
	//	int x = static_cast<int>(peak[0]);
	//	int y = static_cast<int>(peak[1]);

	//	// 绘制圆圈，半径为 2，颜色为紫色(255, 0, 255)，厚度为 -1（填充圆圈）
	//	cv::circle(right_marked_img, cv::Point(x, y), 2, cv::Scalar(255, 0, 255), cv::FILLED);
	//}
	
	roi_img_right_result = new_roi_img_right;
	cv::imshow("Right ROI Image", roi_img_right_result);
	cv::waitKey(0);
	/*cv::imshow("Right ROI Image", roi_img_right_result);
	cv::waitKey(0);*/
	//处理左侧边缘
	std::vector<int> left_roi_horizontal_centers = find_horizontal_light_center(roi_img_left);
	// 反转数据
	std::vector<int> inverted_data(left_roi_horizontal_centers.size());
	std::transform(left_roi_horizontal_centers.begin(), left_roi_horizontal_centers.end(), inverted_data.begin(), [](double x) { return -x; });
	std::vector<int> peaks_left = find_peaks<int>(inverted_data, 160);
	std::vector<std::vector<double>> left_peak_points;
	// 填充合并后的数据结构
	for (int peak : peaks_left) {
		if (peak >= 0 && peak < left_roi_horizontal_centers.size()) {
			// 创建一个新行，并填充数据
			std::vector<double> point;
			point.push_back(left_roi_horizontal_centers[peak]); // 数据值
			point.push_back(peak); // 峰值索引
			left_peak_points.push_back(point); // 添加到结果
		}
	}
	//// 输出合并后的结果
	//for (const auto& point : right_peak_points) {
	//	std::cout << "(" << point[0] << ", " << point[1] << ")\n";
	//}
	if (left_peak_points.empty()) {
		return -10;
	}
	left_peak_points = filter_peaks_by_threshold(left_peak_points, 5.0);
	// 提取 y_start 和 y_end
	y_start = static_cast<int>(left_peak_points[0][1]); // 第一个峰值的 Y 值
	y_end = static_cast<int>(left_peak_points.back()[1]); // 最后一个峰值的 Y 值

	// 确保 y_start 和 y_end 在图像范围内
	if (y_start < 0) y_start = 0;
	if (y_end >= roi_img_left.rows) y_end = roi_img_left.rows - 1;

	// 根据 y_start 和 y_end 提取 ROI
	cv::Rect left_roi(0, y_start, roi_img_left.cols, (y_end - y_start + 1)); // 定义 ROI 区域
	cv::Mat new_roi_img_left = roi_img_left(left_roi); // 提取 ROI

	roi_img_left_result = new_roi_img_left;
	/*cv::imshow("Left ROI Image", roi_img_left_result);
	cv::waitKey(0);*/

	return 0;
}
// 处理一张图片，输出结果为Result结构体
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
	// 遍历输入目录中的所有项
	//if (input_image.empty()) {
	//	return -1;  // 返回错误码
	//}
	int AriStatus = 0;
	std::vector<double> mean_left, std_dev_left, q1_left, q2_left, q3_left;
	std::vector<double> mean_right, std_dev_right, q1_right, q2_right, q3_right;
	load_statistics_from_csv(left_statistics_file_path, mean_left, std_dev_left, q1_left, q2_left, q3_left);
	load_statistics_from_csv(right_statistics_file_path, mean_right, std_dev_right, q1_right, q2_right, q3_right);
	// 加载左侧模板曲线
	cv::Mat temp_left_image = cv::imread(temp_left_curve_path, cv::IMREAD_GRAYSCALE);
	std::vector<int> temp_left_edge_contour = find_horizontal_light_center(temp_left_image);
	// 加载右侧模板曲线
	cv::Mat temp_right_image = cv::imread(temp_right_curve_path, cv::IMREAD_GRAYSCALE);
	std::vector<int> temp_right_edge_contour = find_horizontal_light_center(temp_right_image);
	// 对左右两侧轮廓点进行归一化
	std::vector<double> temp_left_edge_contour_new = normalize_and_scale(temp_left_edge_contour);
	std::vector<double> temp_right_edge_contour_new = normalize_and_scale(temp_right_edge_contour);
	// 对左侧模板线拟合
	//fit_curve得到的数据偏差有点大
	std::pair<std::vector<double>, std::vector<double>> left_fit_result = fit_curve(temp_left_edge_contour_new);
	std::vector<double> temp_left_x_fit = left_fit_result.first;
	std::vector<double> temp_left_y_fit = left_fit_result.second;
	// 找到左侧拟合曲线的第一个极大值
	std::pair<int, double> temp_left_max_result = find_first_max(temp_left_y_fit);
	int temp_t_max_index = -1;
	temp_t_max_index = temp_left_max_result.first;
	double temp_y_max = temp_left_max_result.second;

	// 对右侧模板线拟合
	std::pair<std::vector<double>, std::vector<double>> right_fit_result = fit_curve(temp_right_edge_contour_new);
	std::vector<double> temp_right_x_fit = right_fit_result.first;
	std::vector<double> temp_right_y_fit = right_fit_result.second;

	// 找到右侧拟合曲线的第一个极小值
	std::pair<int, double> temp_right_min_result = find_first_min(temp_right_y_fit);
	int temp_t_min_index = -1;
	temp_t_min_index = temp_right_min_result.first;
	double temp_y_min = temp_right_min_result.second;
	cv::Mat roi_img_left_image;
	cv::Mat roi_img_right_image;
	AriStatus = roi_extractor(input_path, product_temp_path,top_shift_down, bottom_shift_up,output_path, roi_img_left_image, roi_img_right_image);
	std::vector<int> right_edge_contour = find_horizontal_light_center(roi_img_right_image);
	if (right_edge_contour.size() < 160) {
		// 负数是异常状态
		result.right_diff_max = -1;
		result.right_mse = -1;
		std::cout << "提取失败" << std::endl;
	}
	else {
		std::vector<double> right_edge_contour_new = normalize_and_scale(right_edge_contour);
		std::pair<std::vector<double>, std::vector<double>> right_result = fit_curve(right_edge_contour_new);
		std::vector<double> right_x_fit = right_result.first;
		std::vector<double> right_y_fit = right_result.second;
		//找到右侧拟合曲线的第一个极小值
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
	// 产品临时路径
	std::string product_temp_path = "F:/PythonProject/screw/temp/906_temp.png";

	// 定位好产品之后，调整ROI的区域
	int top_shift_down = 400;
	int bottom_shift_up = 400;
	// 输入图片
	std::string input_path = "F:/data/906_NG/906-1_20240725-143044/page_60.bmp";
	// 左侧模板曲线
	std::string temp_left_curve_path = "F:/PythonProject/screw/temp/906_temp_left_curve.png";
	// 右侧模板曲线
	std::string temp_right_curve_path = "F:/PythonProject/screw/temp/906_temp_right_curve.png";
	// 定义输出结果
	Result result = { 0.0, 0.0, 0.0, 0.0 };
	std::string output_path = "C:/Users/25451/Desktop/result";
	// 加载左右两侧统计数据
	std::string right_statistics_file_path = "F:/PythonProject/screw/right_statistical_models.csv";
	std::string left_statistics_file_path = "F:/PythonProject/screw/left_statistical_models.csv";
	int AriStatus = process_one_image(input_path, temp_left_curve_path, temp_right_curve_path, product_temp_path,
		top_shift_down, bottom_shift_up, result, output_path,
		left_statistics_file_path, right_statistics_file_path);
	return 0;
}