#include "engine.h"
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <chrono>
#include <filesystem>
#include <cuda.h>
#include "assert.h"


using namespace std;
namespace fs = std::filesystem;


int main(int argc) {
    // Ensure the onnx model exists
    const std::string onnxModelPath = "/home/ceec/tri/tensorrt-cpp-api/weights/small.onnx";
      // Specify our GPU inference configuration options
    Options options;
    // Specify what precision to use for inference
    // FP16 is approximately twice as fast as FP32.
    options.precision = Precision::FP16;
    // If using INT8 precision, must specify path to directory containing calibration data.
    options.calibrationDataDirectoryPath = "";
    // If the model does not support dynamic batch size, then the below two parameters must be set to 1.
    // Specify the batch size to optimize for.
    options.optBatchSize = 1;
    // Specify the maximum batch size we plan on running.
    options.maxBatchSize = 1;

    Engine engine(options);

    // For our YoloV8 model, we need the values to be normalized between [0.f, 1.f] so we use the following params
    std::array<float, 3> subVals {0.f, 0.f, 0.f};
    std::array<float, 3> divVals {1.f, 1.f, 1.f};
    bool normalize = true;
    // Note, we could have also used the default values.
    
    // Build the onnx model into a TensorRT engine file.
    bool succ = engine.build(onnxModelPath, subVals, divVals, normalize);
    if (!succ) {
        throw std::runtime_error("Unable to build TRT engine.");
    }

    // Load the TensorRT engine file from disk
    succ = engine.loadNetwork();
    if (!succ) {
        throw std::runtime_error("Unable to load TRT engine.");
    }

    // Read the input image
    // TODO: You will need to read the input image required for your model
    const std::string inputFolder = "../inputs/";

    for (const auto& entry : fs::directory_iterator(inputFolder)) {
        if (entry.is_regular_file()) {
            const std::string inputImage = entry.path().string();

            auto cpuImg = cv::imread(inputImage);
            if (cpuImg.empty()) {
                std::cerr << "Unable to read image at path: " << inputImage << std::endl;
                continue; // Skip to the next image if unable to read
            }
            // if (cpuImg.empty()) {
            //     throw std::runtime_error("Unable to read image at path: " + inputImage);
            //     continue;
            // }

            // Upload the image GPU memory
            cv::cuda::GpuMat img;
            img.upload(cpuImg);

            // The model expects RGB input
            cv::cuda::cvtColor(img, img, cv::COLOR_BGR2RGB);
            // cv::imwrite("../results/original_first.jpg", img);
            // In the following section we populate the input vectors to later pass for inference
            const auto& inputDims = engine.getInputDims();
            std::vector<std::vector<cv::cuda::GpuMat>> inputs;

            // Let's use a batch size which matches that which we set the Options.optBatchSize option
            size_t batchSize = options.optBatchSize;

            // TODO:
            // For the sake of the demo, we will be feeding the same image to all the inputs
            // You should populate your inputs appropriately.
            for (const auto & inputDim : inputDims) { // For each of the model inputs...
                std::vector<cv::cuda::GpuMat> input;
                for (size_t j = 0; j < batchSize; ++j) { // For each element we want to add to the batch...
                    // TODO:
                    // You can choose to resize by scaling, adding padding, or a combination of the two in order to maintain the aspect ratio
                    // You can use the Engine::resizeKeepAspectRatioPadRightBottom to resize to a square while maintain the aspect ratio (adds padding where necessary to achieve this).
                    auto resized = Engine::resizeKeepAspectRatioPadRightBottom(img, inputDim.d[1], inputDim.d[2]);
                    // cout << inputDim.d[1] << " " << inputDim.d[2];
                    // You could also perform a resize operation without maintaining aspect ratio with the use of padding by using the following instead:
                    cv::cuda::resize(img, resized, cv::Size(inputDim.d[2], inputDim.d[1])); // TRT dims are (height, width) whereas OpenCV is (width, height)
                    input.emplace_back(std::move(resized));
                }
                inputs.emplace_back(std::move(input));
            }
            // Warm up the network before we begin the benchmark
            std::cout << "\nWarming up the network..." << std::endl;
            std::vector<std::vector<std::vector<float>>> featureVectors;
            for (int i = 0; i < 100; ++i) {
                succ = engine.runInference(inputs, featureVectors);
                if (!succ) {
                    throw std::runtime_error("Unable to run inference.");
                }
            }

            // Benchmark the inference time
            size_t numIterations = 100;
            std::cout << "Warmup done. Running benchmarks (" << numIterations << " iterations)...\n" << std::endl;
            preciseStopwatch stopwatch;
            for (size_t i = 0; i < numIterations; ++i) {
                featureVectors.clear();
                engine.runInference(inputs, featureVectors);
            }
            auto totalElapsedTimeMs = stopwatch.elapsedTime<float, std::chrono::milliseconds>();
            auto avgElapsedTimeMs = totalElapsedTimeMs / numIterations / static_cast<float>(inputs[0].size());

            std::cout << "Benchmarking complete!" << std::endl;
            std::cout << "======================" << std::endl;
            std::cout << "Avg time per sample: " << std::endl;
            std::cout << avgElapsedTimeMs << " ms" << std::endl;
            std::cout << "Batch size: " << std::endl;
            std::cout << featureVectors[0].si << std::endl;
            std::cout << "Avg FPS: " << std::endl;
            std::cout << static_cast<int>(1000 / avgElapsedTimeMs) << " fps" << std::endl;
            std::cout << "======================\n";

            for (size_t batch = 0; batch < featureVectors.size(); ++batch) {
                cv::Vec3b green_color(0, 255, 0);
                cv::Vec3b red_color(0, 0, 255);
                cv::Mat DA, LL;
                cv::Mat img_copy;
                img.download(img_copy);
                cv::cvtColor(img_copy, img_copy, cv::COLOR_BGR2RGB);
                resize(img_copy, img_copy, cv::Size(640, 384), 0, 0, cv::INTER_NEAREST); 
                /*---------BEGIN_PROCESS_SEG----------*/
                cv::Mat segMat = cv::Mat(384 * 640, 2, CV_32F);
                segMat=segMat.reshape(2, { 384, 640 }); 
                cv::Mat segChannels[2];
                int idx = 0;
                for(int i=0; i<segMat.rows; i++){
                    for(int j=0; j<segMat.cols; j++){
                        segMat.at<cv::Vec2f>(i, j)[0] = featureVectors[batch][0][idx];
                        idx++;
                    }
                }
                cv::split(segMat, segChannels);
                cv::compare(100, segChannels[0] * 255, DA, cv::CMP_GT);
                DA.convertTo(DA, CV_8U, 255);
                img_copy.setTo(green_color, DA); 
                /*---------END_PROCESS_SEG----------*/

                /*---------BEGIN_PROCESS_LANE----------*/
                cv::Mat laneMat = cv::Mat(2, 384 * 640, CV_32F);
                laneMat=laneMat.reshape(2, { 384, 640 });
                std::vector<cv::Mat> laneChannels;

                idx = 0;
                for(int i=0; i<laneMat.rows; i++){
                    for(int j=0; j<laneMat.cols; j++){
                        laneMat.at<cv::Vec2f>(i, j)[0] = featureVectors[batch][1][idx];
                        idx++;
                    }
                }
                cv::split(laneMat, laneChannels);
                cv::compare(100, laneChannels[0]*255, LL, cv::CMP_GT);
                LL.convertTo(LL, CV_8U, 255);
                img_copy.setTo(red_color, LL); 

                std::string filename = entry.path().filename().string();
                std::string path = "../results/" + filename;
                cv::imwrite(path, img_copy);
                /*---------END_PROCESS_LANE----------*/
            }
        }
    }
    return 0;
}