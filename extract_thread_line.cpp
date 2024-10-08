#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <boost/filesystem.hpp>
#include <boost/serialization/vector.hpp>
#include <fstream>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
//2.把所有的right和left下的图像当中的点分别提取出来，用一维的向量形式，保存到 left_results.dat，right_results.dat当中

//将left_results等一维向量以易读格式输出到控制台
//void printLeftResults(const std::vector<std::vector<double>>& left_results) {
//	std::cout << "left_results_size(60):" << left_results.size() << "=============";
//	//for (int i = 0; i < 5; ++i) {
//	//	/*std::cout << "Result set " << i + 1 << ": ";*/
//	//	const std::vector<double>& resultSet = left_results[i];
//	//	/*for (size_t j = 0; j < resultSet.size(); ++j) {
//	//		std::cout << resultSet[j];
//	//		if (j < resultSet.size() - 1) {
//	//			std::cout << ", ";
//	//		}
//	//	}*/
//	//	std::cout << "nubmer：====================" << resultSet.size() << "=============";
//	//	std::cout << std::endl;
//	//}
//}
void saveResultsWithPickle(const std::string& outputDir, const std::vector<std::vector<double>>& leftResults, const std::vector<std::vector<double>>& rightResults) 
{
	boost::filesystem::create_directories(outputDir);

	// 保存左侧结果
	std::ofstream leftOutputFile((outputDir + "/left_results.dat").c_str(), std::ios::binary);
	boost::archive::binary_oarchive leftArchive(leftOutputFile);
	leftArchive << leftResults;

	// 保存右侧结果
	std::ofstream rightOutputFile((outputDir + "/right_results.dat").c_str(), std::ios::binary);
	boost::archive::binary_oarchive rightArchive(rightOutputFile);
	rightArchive << rightResults;
}


std::vector<double> scan_and_visualize(const std::string& image_path, char axis ) 
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

void process_and_save_all_folders(const std::string& base_dir, const std::string&output_dir)
{
	std::vector<std::vector<double>> left_results;
	std::vector<std::vector<double>> right_results;
	boost::filesystem::path base_path(base_dir);
	boost::filesystem::directory_iterator end_iter;
	for (boost::filesystem::directory_iterator dir_iter(base_path); dir_iter != end_iter; ++dir_iter) {
		if (!boost::filesystem::is_directory(dir_iter->status())) continue;

		std::cout << "Processing folder: " << dir_iter->path().filename().string() << std::endl;
		std::cout << "The length of left_results is: " << left_results.size() << std::endl;
		std::cout << "The length of right_results is: " << right_results.size() << std::endl;

		boost::filesystem::path left_folder_path = dir_iter->path() / "left";

		boost::filesystem::path right_folder_path = dir_iter->path() / "right";

		for (boost::filesystem::directory_iterator file_iter(left_folder_path); file_iter != end_iter; ++file_iter) {
			if (file_iter->path().extension() == ".bmp") {
				std::vector<double> result = scan_and_visualize(file_iter->path().string(), 'Y');
				left_results.push_back(result);
			}
		}
		/*printLeftResults(left_results);*/
		for (boost::filesystem::directory_iterator file_iter(right_folder_path); file_iter != end_iter; ++file_iter) {
			if (file_iter->path().extension() == ".bmp") {
				std::vector<double> result = scan_and_visualize(file_iter->path().string(), 'Y');
				right_results.push_back(result);
			}
		}
	}
	saveResultsWithPickle(output_dir, left_results, right_results);
	std::cout << "Saved all results to " << output_dir << std::endl;
}
int main()
{
	std::string base_dir = "F:\\data\\Processed_Images";
	std::string output_dir = "F:\\data\\test";
	process_and_save_all_folders(base_dir, output_dir);
	return 0;
}