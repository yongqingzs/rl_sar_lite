/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "inference_runtime.hpp"
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <fstream>
#include <sstream>
#include <cstring>
#include <filesystem>

#ifdef USE_TORCH
#include <ATen/Parallel.h>
#endif

namespace InferenceRuntime
{

namespace
{
#ifdef USE_RKNN
bool contains_rk3588_keyword(const std::string& text)
{
    if (text.empty())
    {
        return false;
    }
    std::string lowered = text;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), ::tolower);
    return lowered.find("rk3588") != std::string::npos;
}

std::string read_text_file(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        return {};
    }
    std::ostringstream oss;
    oss << file.rdbuf();
    return oss.str();
}
#endif
} // namespace

// ============================================================================
// TorchModel Implementation
// ============================================================================

TorchModel::TorchModel()
{
#ifdef USE_TORCH
    // Set threads before model load
    torch::set_num_threads(1);
#endif
}

TorchModel::~TorchModel()
{
}

bool TorchModel::load(const std::string& model_path)
{
    try
    {
#ifdef USE_TORCH
        // Load TorchScript model
        model_ = torch::jit::load(model_path);
        model_path_ = model_path;
        loaded_ = true;
        std::cout << LOGGER::INFO << "Successfully loaded Torch model: " << model_path << std::endl;
        return true;
#else
        std::cout << LOGGER::WARNING << "Torch support not compiled. Please define USE_TORCH." << std::endl;
        loaded_ = false;
        return false;
#endif
    }
    catch (const std::exception& e)
    {
        std::cout << LOGGER::ERROR << "Failed to load Torch model: " << e.what() << std::endl;
        loaded_ = false;
        return false;
    }
}

std::vector<float> TorchModel::forward(const std::vector<std::vector<float>>& inputs)
{
    if (!loaded_)
    {
        throw std::runtime_error("Model not loaded");
    }

#ifdef USE_TORCH
    try
    {
        // Convert input vector to Torch tensor (use first input only)
        const auto& input = inputs[0];
        auto input_tensor = torch::tensor(input, torch::kFloat32).reshape({1, static_cast<int64_t>(input.size())});

        // Disable gradient computation before each forward pass
        torch::autograd::GradMode::set_enabled(false);

        // Ensure single-threaded execution (critical for performance!)
        torch::set_num_threads(1);

        // Execute forward inference
        auto output = model_.forward({input_tensor}).toTensor();

        // Convert output tensor to vector
        return torch_to_vector(output);
    }
    catch (const std::exception& e)
    {
        std::cout << LOGGER::ERROR << "Torch inference error: " << e.what() << std::endl;
        throw;
    }
#else
    throw std::runtime_error("Torch support not compiled");
#endif
}

#ifdef USE_TORCH
torch::Tensor TorchModel::vector_to_torch(const std::vector<float>& data, const std::vector<int64_t>& shape)
{
    // Use torch::tensor() + reshape() to match test program behavior
    auto tensor = torch::tensor(data, torch::kFloat32).reshape(shape);
    return tensor;
}

std::vector<float> TorchModel::torch_to_vector(const torch::Tensor& tensor)
{
    // Ensure tensor is contiguous and on CPU
    auto cpu_tensor = tensor.is_contiguous() ? tensor : tensor.contiguous();
    if (cpu_tensor.device().type() != torch::kCPU)
    {
        cpu_tensor = cpu_tensor.to(torch::kCPU);
    }

    // Get data pointer and size
    float* data_ptr = cpu_tensor.data_ptr<float>();
    int64_t num_elements = cpu_tensor.numel();

    // Copy data to vector
    return std::vector<float>(data_ptr, data_ptr + num_elements);
}
#endif

// ============================================================================
// ONNXModel Implementation
// ============================================================================

ONNXModel::ONNXModel()
#ifdef USE_ONNX
    : memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
#endif
{
#ifdef USE_ONNX
    // Initialize ONNX Runtime environment
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ONNXModel");
#endif
}

ONNXModel::~ONNXModel()
{
#ifdef USE_ONNX
    session_.reset();
    env_.reset();
#endif
}

bool ONNXModel::load(const std::string& model_path)
{
    try
    {
#ifdef USE_ONNX
        // Configure session options
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        // Create inference session
        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);

        // Setup input/output information
        setup_input_output_info();

        model_path_ = model_path;
        loaded_ = true;
        std::cout << LOGGER::INFO << "Successfully loaded ONNX model: " << model_path << std::endl;
        return true;
#else
        std::cout << LOGGER::WARNING << "ONNX support not compiled. Please define USE_ONNX." << std::endl;
        loaded_ = false;
        return false;
#endif
    }
    catch (const std::exception& e)
    {
        std::cout << LOGGER::ERROR << "Failed to load ONNX model: " << e.what() << std::endl;
        loaded_ = false;
        return false;
    }
}

std::vector<float> ONNXModel::forward(const std::vector<std::vector<float>>& inputs)
{
    if (!loaded_)
    {
        throw std::runtime_error("Model not loaded");
    }

#ifdef USE_ONNX
    try
    {
        // Create memory info
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // Get input (use first input only)
        const auto& input = inputs[0];
        auto input_shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

        // Create input tensor
        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(input.data()),
            input.size(),
            input_shape.data(),
            input_shape.size()
        );

        // Prepare input/output names
        const char* input_names[] = {input_node_names_[0].c_str()};
        const char* output_names[] = {output_node_names_[0].c_str()};

        // Execute inference
        auto outputs = session_->Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            1,
            output_names,
            1
        );

        // Extract output data
        return extract_output_data(outputs);
    }
    catch (const std::exception& e)
    {
        std::cout << LOGGER::ERROR << "ONNX inference error: " << e.what() << std::endl;
        throw;
    }
#else
    throw std::runtime_error("ONNX support not compiled");
#endif
}

#ifdef USE_ONNX
void ONNXModel::setup_input_output_info()
{
    // Get input node information
    size_t num_input_nodes = session_->GetInputCount();
    input_node_names_.reserve(num_input_nodes);
    input_shapes_.reserve(num_input_nodes);

    for (size_t i = 0; i < num_input_nodes; ++i)
    {
        // Get input name
        auto input_name = session_->GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
        input_node_names_.push_back(std::string(input_name.get()));

        // Get input shape
        Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(i);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();

        std::vector<int64_t> shape;
        for (auto dim : input_dims)
        {
            // Handle dynamic dimensions
            if (dim == -1)
            {
                shape.push_back(1);
            }
            else
            {
                shape.push_back(dim);
            }
        }
        input_shapes_.push_back(shape);
    }

    // Get output node information
    size_t num_output_nodes = session_->GetOutputCount();
    output_node_names_.reserve(num_output_nodes);
    output_shapes_.reserve(num_output_nodes);

    for (size_t i = 0; i < num_output_nodes; ++i)
    {
        // Get output name
        auto output_name = session_->GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
        output_node_names_.push_back(std::string(output_name.get()));

        // Get output shape
        Ort::TypeInfo output_type_info = session_->GetOutputTypeInfo(i);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto output_dims = output_tensor_info.GetShape();

        std::vector<int64_t> shape;
        for (auto dim : output_dims)
        {
            // Handle dynamic dimensions
            if (dim == -1)
            {
                shape.push_back(1);
            }
            else
            {
                shape.push_back(dim);
            }
        }
        output_shapes_.push_back(shape);
    }
}

std::vector<float> ONNXModel::extract_output_data(const std::vector<Ort::Value>& outputs)
{
    if (outputs.empty())
    {
        throw std::runtime_error("No outputs from ONNX model");
    }

    // Get first output tensor
    auto& output = outputs[0];
    float* output_data = const_cast<float*>(output.GetTensorData<float>());

    // Calculate total number of output elements
    auto output_shape = output.GetTensorTypeAndShapeInfo().GetShape();

    int64_t num_elements = 1;
    for (auto dim : output_shape)
    {
        if (dim > 0)
        {
            num_elements *= dim;
        }
    }

    // Copy output data to vector
    std::vector<float> result(output_data, output_data + num_elements);

    return result;
}
#endif

// ============================================================================
// RknnModel Implementation
// ============================================================================

RknnModel::RknnModel() = default;

RknnModel::~RknnModel()
{
#ifdef USE_RKNN
    release_resources();
#endif
}

bool RknnModel::load(const std::string& model_path)
{
#ifdef USE_RKNN
    release_resources();

    if (!validate_platform())
    {
        loaded_ = false;
        return false;
    }

    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    if (!file)
    {
        std::cout << LOGGER::ERROR << "Failed to open RKNN model file: " << model_path << std::endl;
        loaded_ = false;
        return false;
    }

    const std::streamsize file_size = file.tellg();
    if (file_size <= 0)
    {
        std::cout << LOGGER::ERROR << "RKNN model file is empty: " << model_path << std::endl;
        loaded_ = false;
        return false;
    }

    file.seekg(0, std::ios::beg);
    model_data_.resize(static_cast<size_t>(file_size));
    if (!file.read(reinterpret_cast<char*>(model_data_.data()), file_size))
    {
        std::cout << LOGGER::ERROR << "Failed to read RKNN model data: " << model_path << std::endl;
        loaded_ = false;
        return false;
    }

    int ret = rknn_init(&ctx_, model_data_.data(), model_data_.size(), 0, nullptr);
    if (ret < 0)
    {
        std::cout << LOGGER::ERROR << "rknn_init failed with code " << ret << std::endl;
        release_resources();
        return false;
    }

    std::memset(&io_num_, 0, sizeof(io_num_));
    ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
    if (ret < 0)
    {
        std::cout << LOGGER::ERROR << "rknn_query(RKNN_QUERY_IN_OUT_NUM) failed with code " << ret << std::endl;
        release_resources();
        return false;
    }

    input_attrs_.resize(io_num_.n_input);
    for (int i = 0; i < io_num_.n_input; ++i)
    {
        auto& attr = input_attrs_[i];
        std::memset(&attr, 0, sizeof(attr));
        attr.index = i;
        ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr));
        if (ret < 0)
        {
            std::cout << LOGGER::ERROR << "rknn_query(RKNN_QUERY_INPUT_ATTR) failed with code " << ret << std::endl;
            release_resources();
            return false;
        }
    }

    output_attrs_.resize(io_num_.n_output);
    for (int i = 0; i < io_num_.n_output; ++i)
    {
        auto& attr = output_attrs_[i];
        std::memset(&attr, 0, sizeof(attr));
        attr.index = i;
        ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
        if (ret < 0)
        {
            std::cout << LOGGER::ERROR << "rknn_query(RKNN_QUERY_OUTPUT_ATTR) failed with code " << ret << std::endl;
            release_resources();
            return false;
        }
    }

    model_path_ = model_path;
    loaded_ = true;
    std::cout << LOGGER::INFO << "Successfully loaded RKNN model: " << model_path << std::endl;
    return true;
#else
    (void)model_path;
    std::cout << LOGGER::ERROR << "RKNN support not compiled. Please rebuild with USE_RKNN." << std::endl;
    loaded_ = false;
    return false;
#endif
}

std::vector<float> RknnModel::forward(const std::vector<std::vector<float>>& inputs)
{
#ifdef USE_RKNN
    if (!loaded_)
    {
        throw std::runtime_error("RKNN model not loaded");
    }

    if (inputs.size() != static_cast<size_t>(io_num_.n_input))
    {
        throw std::runtime_error("RKNN input count mismatch: expected " + std::to_string(io_num_.n_input));
    }

    std::vector<rknn_input> rknn_inputs(io_num_.n_input);
    for (int i = 0; i < io_num_.n_input; ++i)
    {
        const auto& attr = input_attrs_[i];
        const auto& data = inputs[i];
        if (data.empty())
        {
            throw std::runtime_error("RKNN input data is empty at index " + std::to_string(i));
        }

        rknn_inputs[i] = {};
        rknn_inputs[i].index = attr.index;
        rknn_inputs[i].buf = const_cast<float*>(data.data());
        rknn_inputs[i].size = data.size() * sizeof(float);
        rknn_inputs[i].type = RKNN_TENSOR_FLOAT32; // All reinforcement-learning policies use float inputs
        rknn_inputs[i].fmt = attr.fmt == RKNN_TENSOR_UNDEFINED ? RKNN_TENSOR_UNDEFINED : attr.fmt;
        rknn_inputs[i].pass_through = 0; // Let RKNN convert float inputs to the model's native type

        const size_t expected_elements = get_tensor_element_count(attr);
        if (expected_elements > 0 && data.size() != expected_elements)
        {
            std::cout << LOGGER::WARNING
                      << "RKNN input element mismatch at index " << i
                      << ": expected " << expected_elements
                      << ", got " << data.size() << std::endl;
        }
    }

    int ret = rknn_inputs_set(ctx_, io_num_.n_input, rknn_inputs.data());
    if (ret < 0)
    {
        throw std::runtime_error("rknn_inputs_set failed with code " + std::to_string(ret));
    }

    ret = rknn_run(ctx_, nullptr);
    if (ret < 0)
    {
        throw std::runtime_error("rknn_run failed with code " + std::to_string(ret));
    }

    std::vector<rknn_output> rknn_outputs(io_num_.n_output);
    for (int i = 0; i < io_num_.n_output; ++i)
    {
        rknn_outputs[i].index = output_attrs_[i].index;
        rknn_outputs[i].want_float = 1;
        rknn_outputs[i].is_prealloc = 0;
        rknn_outputs[i].buf = nullptr;
    }

    ret = rknn_outputs_get(ctx_, io_num_.n_output, rknn_outputs.data(), nullptr);
    if (ret < 0)
    {
        throw std::runtime_error("rknn_outputs_get failed with code " + std::to_string(ret));
    }

    size_t total_elements = 0;
    for (int i = 0; i < io_num_.n_output; ++i)
    {
        size_t count = get_tensor_element_count(output_attrs_[i]);
        if (count == 0 && rknn_outputs[i].size > 0)
        {
            count = rknn_outputs[i].size / sizeof(float);
        }
        total_elements += count;
    }

    std::vector<float> result;
    result.reserve(total_elements);
    for (int i = 0; i < io_num_.n_output; ++i)
    {
        const auto* buffer = reinterpret_cast<float*>(rknn_outputs[i].buf);
        size_t element_count = get_tensor_element_count(output_attrs_[i]);
        if (element_count == 0 && rknn_outputs[i].size > 0)
        {
            element_count = rknn_outputs[i].size / sizeof(float);
        }
        if (buffer && element_count > 0)
        {
            result.insert(result.end(), buffer, buffer + element_count);
        }
    }

    ret = rknn_outputs_release(ctx_, io_num_.n_output, rknn_outputs.data());
    if (ret < 0)
    {
        std::cout << LOGGER::WARNING << "rknn_outputs_release failed with code " << ret << std::endl;
    }

    return result;
#else
    (void)inputs;
    throw std::runtime_error("RKNN support not compiled");
#endif
}

#ifdef USE_RKNN
void RknnModel::release_resources()
{
    if (ctx_ != 0)
    {
        rknn_destroy(ctx_);
        ctx_ = 0;
    }
    model_data_.clear();
    input_attrs_.clear();
    output_attrs_.clear();
    std::memset(&io_num_, 0, sizeof(io_num_));
    loaded_ = false;
}

bool RknnModel::validate_platform() const
{
#if defined(__linux__)
    const std::string compatible_path = "/proc/device-tree/compatible";
    const std::string model_path = "/proc/device-tree/model";

    if (std::filesystem::exists(compatible_path))
    {
        const auto content = read_text_file(compatible_path);
        if (contains_rk3588_keyword(content))
        {
            return true;
        }
    }

    if (std::filesystem::exists(model_path))
    {
        const auto content = read_text_file(model_path);
        if (contains_rk3588_keyword(content))
        {
            return true;
        }
    }

    std::cout << LOGGER::ERROR << "RKNN backend is only available on RK3588 devices." << std::endl;
    return false;
#else
    std::cout << LOGGER::ERROR << "RKNN backend is only supported on Linux RK3588 devices." << std::endl;
    return false;
#endif
}

size_t RknnModel::get_tensor_element_count(const rknn_tensor_attr& attr) const
{
    if (attr.n_elems > 0)
    {
        return static_cast<size_t>(attr.n_elems);
    }

    const size_t type_size = get_tensor_type_size(attr.type);
    if (type_size > 0 && attr.size >= type_size)
    {
        return static_cast<size_t>(attr.size) / type_size;
    }

    return 0;
}

size_t RknnModel::get_tensor_type_size(rknn_tensor_type type) const
{
    switch (type)
    {
        case RKNN_TENSOR_FLOAT32:
        case RKNN_TENSOR_INT32:
        case RKNN_TENSOR_UINT32:
            return 4;
        case RKNN_TENSOR_FLOAT16:
        case RKNN_TENSOR_INT16:
        case RKNN_TENSOR_UINT16:
            return 2;
        case RKNN_TENSOR_INT8:
        case RKNN_TENSOR_UINT8:
        case RKNN_TENSOR_BOOL:
            return 1;
        case RKNN_TENSOR_INT64:
            return 8;
        case RKNN_TENSOR_BFLOAT16:
            return 2;
        default:
            return sizeof(float);
    }
}
#endif

// ============================================================================
// ModelFactory Implementation
// ============================================================================

std::unique_ptr<Model> ModelFactory::create_model(ModelType type)
{
    switch (type)
    {
        case ModelType::TORCH:
            return std::make_unique<TorchModel>();
        case ModelType::ONNX:
            return std::make_unique<ONNXModel>();
        case ModelType::RKNN:
#ifdef USE_RKNN
            return std::make_unique<RknnModel>();
#else
            std::cout << LOGGER::ERROR << "RKNN backend not available in this build." << std::endl;
            return nullptr;
#endif
        default:
            return nullptr;
    }
}

ModelFactory::ModelType ModelFactory::detect_model_type(const std::string& model_path)
{
    // Extract file extension from path
    std::filesystem::path path(model_path);
    std::string extension = path.extension().string();

    // Convert to lowercase for case-insensitive comparison
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

    // Determine model type based on extension
    if (extension == ".pt" || extension == ".pth")
    {
        return ModelType::TORCH;
    }
    else if (extension == ".onnx")
    {
        return ModelType::ONNX;
    }
    else if (extension == ".rknn")
    {
#ifdef USE_RKNN
        return ModelType::RKNN;
#else
        throw std::runtime_error("RKNN backend not enabled. Build on RK3588 with librknn_api to use .rknn models.");
#endif
    }
    else
    {
        throw std::runtime_error("Unknown model file extension: " + extension + ". Supported: .pt, .pth, .onnx, .rknn");
    }
}

std::unique_ptr<Model> ModelFactory::load_model(const std::string& model_path, ModelType type)
{
    // If type is AUTO, automatically detect model type
    if (type == ModelType::AUTO)
    {
        type = detect_model_type(model_path);
    }

    // Create and load model
    auto model = create_model(type);
    if (model && model->load(model_path))
    {
        return model;
    }
    return nullptr;
}

} // namespace InferenceRuntime
