#include <opencv2/opencv.hpp>
#include <vector>
#include <filesystem> // C++17
#include <chrono>
#include <cmath>

cv::Mat slice_single(const cv::Mat &img, const cv::Mat &mask, const std::vector<float> &bbox, int target_size) {
    int x0 = std::max(static_cast<int>(std::floor(bbox[0]) - 1), 0);
    int y0 = std::max(static_cast<int>(std::floor(bbox[1]) - 1), 0);
    cv::Rect roi(x0, y0, mask.cols, mask.rows);
    cv::Mat mask_img = img(roi);

    cv::Mat masked_image = cv::Mat::zeros(mask_img.size(), mask_img.type());
    mask_img.copyTo(masked_image, mask);

    cv::Mat background_color = cv::Mat::ones(mask_img.size(), mask_img.type()) * cv::Vec3b(113, 119, 52);
    cv::Mat background_mask;
    cv::bitwise_not(mask, background_mask);
    background_color.copyTo(background_mask, background_mask);

    cv::add(masked_image, background_mask, masked_image);

    if (std::max(masked_image.rows, masked_image.cols) > target_size) {
        float scale_factor = static_cast<float>(target_size) / std::max(masked_image.rows, masked_image.cols);
        cv::Size new_size(masked_image.cols * scale_factor, masked_image.rows * scale_factor);
        cv::resize(masked_image, masked_image, new_size);
    }

    cv::Mat canvas = cv::Mat::ones(target_size, target_size, CV_8UC3) * cv::Vec3b(113, 119, 52);
    int x_offset = (target_size - masked_image.cols) / 2;
    int y_offset = (target_size - masked_image.rows) / 2;
    cv::Rect roi_to(x_offset, y_offset, masked_image.cols, masked_image.rows);
    masked_image.copyTo(canvas(roi_to));

    return canvas;
}

void slice_image(const cv::Mat &img, const std::vector<std::vector<float>> &bboxes, 
                 const std::vector<int> &labels, const std::vector<cv::Mat> &masks,
                 const std::string &img_path, const std::string &output_path) {
    int target_size = 256;
    auto start_time = std::chrono::high_resolution_clock::now();

    std::filesystem::path path(img_path);
    std::string img_name = path.stem();

    for (int i = 0; i < bboxes.size(); i++) {
        std::vector<float> bbox = bboxes[i];
        if (bbox[4] < 0.4)
            continue;

        if (!masks[i].empty()) {
            cv::Mat sub_img = slice_single(img, masks[i], bbox, target_size);
            std::filesystem::path save_path = output_path;
            save_path /= img_name + std::to_string(bbox[4]) + "_" + std::to_string(i) + ".png";
            
            if (!std::filesystem::exists(save_path)) {
                std::filesystem::create_directories(save_path);
            }

            cv::imwrite(save_path.string(), sub_img);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    std::cout << "Inference time: " << inference_time << "ms" << std::endl;
}
