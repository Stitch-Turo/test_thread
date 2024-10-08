#include<opencv2/opencv.hpp>
 
#include <iostream>
#include <string>
#include <vector>
#include <boost/filesystem.hpp>
#include <algorithm>
#include <numeric>
#include <unordered_map>

//1.在每个page上，裁剪出左侧和右侧的螺纹线，并且以二值图像分别保存在,left 和 right 的目录下

//使用最大值和自定义阈值距离筛选峰值
std::vector<cv::Point> filterPeaksByThresholdAndMax(const std::vector<cv::Point>& peaks, int threshold = 5) 
{
	std::vector<int> xValues;
	for (const auto& peak : peaks) {
		xValues.push_back(peak.x);
	}

	int maxX = *std::max_element(xValues.begin(), xValues.end());

	std::vector<cv::Point> filteredPeaks;
	for (const auto& peak : peaks) {
		if (std::abs(peak.x - maxX) <= threshold) {
			filteredPeaks.push_back(peak);
		}
	}

	return filteredPeaks;
}
template <class T>
std::vector<int> findPeaks(const std::vector<T> &src, int distance)
{
	int length = src.size();
	if (length <= 1) return std::vector<int>();
	//we dont need peaks at start and end points
	std::vector<int> sign(length, -1);
	std::vector<T> difference(length, 0);
	std::vector<int> temp_out;
	//first-order difference (sign)
	adjacent_difference(src.begin(), src.end(), difference.begin());
	difference.erase(difference.begin());
	difference.pop_back();
	for (int i = 0; i < difference.size(); ++i) {
		if (difference[i] >= 0) sign[i] = 1;
	}
	//second-order difference
	for (int j = 1; j < length - 1; ++j)
	{
		int  diff = sign[j] - sign[j - 1];
		if (diff < 0) {
			temp_out.push_back(j);
		}
	}
	if (temp_out.size() == 0 || distance == 0) return temp_out;
	//sort peaks from large to small by src value at peaks
	std::sort(temp_out.begin(), temp_out.end(), [&src](int a, int b) {
		return (src[a] > src[b]);
	});

	std::vector<int> ans;


	//Initialize the answer and the collection to be excluded
	//ans.push_back(temp_out[0]);
	std::unordered_map<int, int> except;
	////    int left=temp_out[0]-distance>0? temp_out[0]-distance:0;
	////    int right=temp_out[0]+distance>length-1? length-1:temp_out[0]+distance;
	//    int left=temp_out[0]-distance;
	//    int right=temp_out[0]+distance;
	//    for (int i = left;i<=right; ++i) {
	//        except.insert(i);
	//    }
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
	//sort the ans from small to large by index value
	std::sort(ans.begin(), ans.end());
	return ans;
}
//检测图像中的峰值并排除掉部分点
std::pair<cv::Mat, cv::Mat> detectAndFilterPeaks(const cv::Mat& image, int distance = 160, int threshold = 5) 
{
	std::vector<int> xValues;
	std::vector<int> yValues;
	for (int y = 0; y < image.rows; ++y) {
		for (int x = 0; x < image.cols; ++x) {
			if (image.at<uchar>(y, x) == 255) {
				xValues.push_back(x);
				yValues.push_back(y);
			}
		}
	}
	std::vector<int> peaks = findPeaks(xValues, distance);

	std::vector<cv::Point> peakTop;
	for (const auto& peakIndex : peaks) {
		peakTop.push_back(cv::Point(xValues[peakIndex], yValues[peakIndex]));
	}

	cv::Mat markedImg;
	cv::cvtColor(image, markedImg, cv::COLOR_GRAY2BGR);
	for (const auto& peak : peakTop) {
		cv::circle(markedImg, peak, 5, cv::Scalar(255, 255, 255), -1);
	}

	std::vector<cv::Point> peakTopFiltered = filterPeaksByThresholdAndMax(peakTop, threshold);

	for (const auto& peak : peakTopFiltered) {
		cv::circle(markedImg, peak, 5, cv::Scalar(255, 0, 255), -1);
	}

	int yStart = peakTopFiltered.empty() ? 0 : peakTopFiltered.front().y;
	int yEnd = peakTopFiltered.empty() ? 0 : peakTopFiltered.back().y;
	cv::Mat finalCurveImage = peakTopFiltered.empty() ? cv::Mat() : image(cv::Rect(0, yStart, image.cols, yEnd - yStart + 1));

	return std::make_pair(markedImg, finalCurveImage);
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
//处理单个图片
void processImage(const std::string& image_path, const std::string& product_temp_path, int top_shift_down, int bottom_shift_up, const std::string& output_base_dir)
{
	cv::Mat input_img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
	if (input_img.empty()) {
		std::cerr << "无法读取图像。" << std::endl;
		return ;
	}
	int image_height = input_img.rows;
	int image_width = input_img.cols;

	cv::Mat product_temp = cv::imread(product_temp_path, cv::IMREAD_GRAYSCALE);
	if (product_temp.empty())
	{
		throw std::invalid_argument("产品模板图像为空，请检查图像路径");
	}
	// 去毛刺
	cv::Mat clear_img;
	cv::Mat clear_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(19, 19));
	cv::morphologyEx(input_img, clear_img, cv::MORPH_CLOSE, clear_kernel);
	// 降采样定位 ROI
	cv::Mat down_sample_img;
	cv::resize(clear_img, down_sample_img, cv::Size(), 0.25, 0.25);
	cv::Mat resultImg;
	cv::matchTemplate(down_sample_img, product_temp, resultImg, cv::TM_CCORR_NORMED);//化相关系数匹配法(最好匹配1) // cv::TM_CCOEFF_NORMED
	double maxValue = 0;
	cv::Point  maxLoc;
	cv::minMaxLoc(resultImg, nullptr, &maxValue, nullptr, &maxLoc);
	// 检查匹配结果
	if (maxValue < 0.6)
	{
		throw std::invalid_argument("模板匹配失败，最大匹配值: " + std::to_string(maxValue));
	}
	//定位到整个产品ROI还原到原图
	cv::Rect productRoi(maxLoc.x * 4, maxLoc.y * 4, product_temp.cols * 4, product_temp.rows * 4);
	int x = productRoi.x;
	int y = productRoi.y + top_shift_down;
	int w = productRoi.width;
	int h = productRoi.height - (top_shift_down + bottom_shift_up);
	//调整product_Roi的区域
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
		throw std::invalid_argument("调整后的 ROI 超出图像范围");
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
		//std::vector<std::vector<cv::Point>>{contour} 将当前轮廓转换为一个包含单个轮廓的向量，以便 cv::drawContours 函数可以接受。
		//-1 表示绘制所有的轮廓。
		//255 表示轮廓的颜色为白色。
		//1 表示轮廓的线宽为 1 像素。
	}

	// 从轮廓图像当中提取调整后的 ROI 区域
	cv::Mat roi_img = contours_img(adjustedRoi);
	// cv::imwrite("./roi_img.png", roi_img);

	int mid_point = roi_img.cols / 2;
	cv::Mat roi_img_left = roi_img(cv::Rect(0, 0, mid_point, roi_img.rows));
	cv::Mat roi_img_right = roi_img(cv::Rect(mid_point, 0, roi_img.cols - mid_point, roi_img.rows));

	// 创建路径，用于保存已经处理的图像
	boost::filesystem::path imgPath(image_path);
	boost::filesystem::path directory = imgPath.parent_path();
	std::string last_directory_name = directory.filename().string();

	boost::filesystem::path output_dir_right(output_base_dir);
	output_dir_right /= last_directory_name;
	output_dir_right /= "right";
	if (!boost::filesystem::exists(output_dir_right))
		boost::filesystem::create_directories(output_dir_right);
	boost::filesystem::path output_path_right = output_dir_right / imgPath.filename();

	boost::filesystem::path output_dir_left(output_base_dir);
	output_dir_left /= last_directory_name;
	output_dir_left /= "left";
	if (!boost::filesystem::exists(output_dir_left))
		boost::filesystem::create_directories(output_dir_left);
	boost::filesystem::path output_path_left = output_dir_left / imgPath.filename();

	// 将左边的边缘旋转180度保存
	cv::Mat roi_img_left_rotate;
	cv::rotate(roi_img_left, roi_img_left_rotate, cv::ROTATE_180);
	cv::Mat marked_img_left_rotate, new_roi_img_left_rotate;
	std::pair<cv::Mat, cv::Mat> result1=detectAndFilterPeaks(roi_img_left_rotate, 160, 8);
	marked_img_left_rotate = result1.first;
	new_roi_img_left_rotate = result1.second;
	cv::Mat marked_img_left, new_roi_img_left;
	cv::rotate(marked_img_left_rotate, marked_img_left, cv::ROTATE_180);
	cv::rotate(new_roi_img_left_rotate, new_roi_img_left, cv::ROTATE_180);

	cv::Mat marked_img_right, new_roi_img_right;
	std::pair<cv::Mat, cv::Mat> result2 = detectAndFilterPeaks(roi_img_right, 160, 8);
	marked_img_right = result2.first;
	new_roi_img_right = result2.second;

	// 保存图像
	cv::imwrite(output_path_right.string(), new_roi_img_right);
	cv::imwrite(output_path_left.string(), new_roi_img_left);
	//cv::imwrite(output_path_left.string(), marked_img_left);
	//cv::imwrite(output_path_right.string(), marked_img_right);
}
//读取文件夹中所有图片，需用到boost库
void processAllFolders(const std::string& baseDir, const std::string& productTempPath, int topShiftDown, int bottomShiftUp, const std::string& outputBaseDir)
{
	boost::filesystem::path basePath(baseDir);

	boost::filesystem::directory_iterator endIter;//标记遍历的结束位置
	for (boost::filesystem::directory_iterator dirIter(basePath); dirIter != endIter; ++dirIter)
	{
		if (boost::filesystem::is_directory(dirIter->status()))//检查当前遍历到的条目是否是一个目录。如果是目录，则进入内层循环。
		{
			for (boost::filesystem::directory_iterator fileIter(dirIter->path()); fileIter != endIter; ++fileIter)
			{
				if (fileIter->path().extension() == ".bmp")
				{
					processImage(fileIter->path().string(), productTempPath, topShiftDown, bottomShiftUp, outputBaseDir);
				}
			}
			std::cout << dirIter->path().string() << "=============" << "完成" << std::endl;
		}
	}
}
int main()
{
	auto start = std::chrono::high_resolution_clock::now();
	/* 产品的模板，用于定位产品*/
	std::string product_temp_path = "E:\\QtProject\\QtProject\\204PEDThreadSystem\\bin\\release\\Temp\\906\\906_temp.png";
	 /*定位好产品之后，调整ROI的区域*/
	int top_shift_down = 400;
	int bottom_shift_up = 400;
	std::string base_dir = "F:\\data\\906_NG";
	/*std::string base_dir = "C:\\Users\\25451\\Desktop\\906NG";*/
	std::string output_base_dir = "F:\\data\\Processed_Images";
	processAllFolders(base_dir, product_temp_path, top_shift_down, bottom_shift_up, output_base_dir);
	// 结束计时
	auto end = std::chrono::high_resolution_clock::now();
	// 计算持续时间
	std::chrono::duration<double, std::milli> duration = end - start; // 以毫秒为单位

	// 输出时间
	std::cout << "程序执行时间: " << duration.count() << " 毫秒" << std::endl;
	return 0;
}


