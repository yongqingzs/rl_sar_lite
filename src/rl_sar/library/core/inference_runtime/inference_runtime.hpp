/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef INFERENCE_RUNTIME_HPP
#define INFERENCE_RUNTIME_HPP

#include <vector>
#include <string>
#include <memory>
#include <filesystem>
#include <algorithm>
#include <cstdint>
#include "logger.hpp"

#ifdef USE_TORCH
#include <torch/script.h>
#endif

#ifdef USE_ONNX
#include <onnxruntime_cxx_api.h>
#endif

#ifdef USE_RKNN
#include <rknn_api.h>
#endif

namespace InferenceRuntime
{

/**
 * @brief Model interface base class
 *
 * Defines common interface for model loading and inference, supporting both Torch and ONNX backends
 */
class Model
{
public:
    virtual ~Model() = default;

    /**
     * @brief Load model file
     * @param model_path Model file path
     * @return Returns true if loading succeeds, false if it fails
     */
    virtual bool load(const std::string& model_path) = 0;

    /**
     * @brief Check if model is loaded
     * @return Returns true if loaded, false otherwise
     */
    virtual bool is_loaded() const = 0;

    /**
     * @brief Forward inference (single input, supports initializer list)
     * @param inputs Vector of input data vectors (usually single element)
     * @return Inference result vector
     */
    virtual std::vector<float> forward(const std::vector<std::vector<float>>& inputs) = 0;

    /**
     * @brief Get model type string
     * @return Model type ("torch", "onnx" or "rknn")
     */
    virtual std::string get_model_type() const = 0;
};

/**
 * @brief Torch model implementation class
 *
 * Model inference implementation based on PyTorch TorchScript
 */
class TorchModel : public Model
{
private:
    bool loaded_ = false;               ///< Whether model is loaded
    std::string model_path_;            ///< Model file path

#ifdef USE_TORCH
    torch::jit::script::Module model_; ///< TorchScript model object
#endif

public:
    TorchModel();
    ~TorchModel();

    bool load(const std::string& model_path) override;
    bool is_loaded() const override { return loaded_; }
    std::vector<float> forward(const std::vector<std::vector<float>>& inputs) override;
    std::string get_model_type() const override { return "torch"; }

private:
#ifdef USE_TORCH
    /**
     * @brief Convert vector data to Torch tensor
     * @param data Input data vector
     * @param shape Tensor shape
     * @return Torch tensor
     */
    torch::Tensor vector_to_torch(const std::vector<float>& data, const std::vector<int64_t>& shape);

    /**
     * @brief Convert Torch tensor to vector data
     * @param tensor Input tensor
     * @return Data vector
     */
    std::vector<float> torch_to_vector(const torch::Tensor& tensor);
#endif
};

/**
 * @brief ONNX model implementation class
 *
 * Model inference implementation based on ONNX Runtime
 */
class ONNXModel : public Model
{
private:
    bool loaded_ = false;               ///< Whether model is loaded
    std::string model_path_;            ///< Model file path

#ifdef USE_ONNX
    std::unique_ptr<Ort::Session> session_;                 ///< ONNX inference session
    std::unique_ptr<Ort::Env> env_;                         ///< ONNX runtime environment
    Ort::MemoryInfo memory_info_;                           ///< Memory information
    std::vector<std::string> input_node_names_;             ///< Input node names
    std::vector<std::string> output_node_names_;            ///< Output node names
    std::vector<std::vector<int64_t>> input_shapes_;        ///< Input shapes
    std::vector<std::vector<int64_t>> output_shapes_;       ///< Output shapes
#endif

public:
    ONNXModel();
    ~ONNXModel();

    bool load(const std::string& model_path) override;
    bool is_loaded() const override { return loaded_; }
    std::vector<float> forward(const std::vector<std::vector<float>>& inputs) override;
    std::string get_model_type() const override { return "onnx"; }

private:
#ifdef USE_ONNX
    /**
     * @brief Setup input/output node information
     */
    void setup_input_output_info();

    /**
     * @brief Extract data from ONNX outputs
     * @param outputs ONNX inference outputs
     * @return Extracted data vector
     */
    std::vector<float> extract_output_data(const std::vector<Ort::Value>& outputs);
#endif
};

/**
 * @brief RKNN model implementation class
 *
 * Provides inference support for models converted to Rockchip RKNN format.
 */
class RknnModel : public Model
{
private:
    bool loaded_ = false;
    std::string model_path_;

#ifdef USE_RKNN
    rknn_context ctx_ = 0;
    rknn_input_output_num io_num_{};
    std::vector<rknn_tensor_attr> input_attrs_;
    std::vector<rknn_tensor_attr> output_attrs_;
    std::vector<uint8_t> model_data_;

    /**
     * @brief Release RKNN context and buffers
     */
    void release_resources();

    /**
     * @brief Validate platform compatibility
     */
    bool validate_platform() const;

    /**
     * @brief Calculate number of elements in output tensor
     */
    size_t get_tensor_element_count(const rknn_tensor_attr& attr) const;

    /**
     * @brief Return tensor type size in bytes
     */
    size_t get_tensor_type_size(rknn_tensor_type type) const;
#endif

public:
    RknnModel();
    ~RknnModel();

    bool load(const std::string& model_path) override;
    bool is_loaded() const override { return loaded_; }
    std::vector<float> forward(const std::vector<std::vector<float>>& inputs) override;
    std::string get_model_type() const override { return "rknn"; }
};

/**
 * @brief Model factory class
 *
 * Responsible for creating and loading different types of models
 */
class ModelFactory
{
public:
    /**
     * @brief Model type enumeration
     */
    enum class ModelType
    {
        TORCH,  ///< TorchScript model
        ONNX,   ///< ONNX model
        RKNN,   ///< Rockchip RKNN model
        AUTO    ///< Automatically detect model type
    };

    /**
     * @brief Create model of specified type
     * @param type Model type
     * @return Model smart pointer
     */
    static std::unique_ptr<Model> create_model(ModelType type = ModelType::AUTO);

    /**
     * @brief Detect model type based on file path
     * @param model_path Model file path
     * @return Detected model type
     */
    static ModelType detect_model_type(const std::string& model_path);

    /**
     * @brief Load model file
     * @param model_path Model file path
     * @param type Model type (default: auto-detect)
     * @return Successfully loaded model smart pointer, returns nullptr on failure
     */
    static std::unique_ptr<Model> load_model(const std::string& model_path, ModelType type = ModelType::AUTO);
};

} // namespace InferenceRuntime

#endif // INFERENCE_RUNTIME_HPP
