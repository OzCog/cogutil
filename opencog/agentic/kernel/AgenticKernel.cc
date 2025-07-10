/*
 * opencog/agentic/kernel/AgenticKernel.cc
 *
 * Base implementation for Agentic Grammar Kernels.
 * Provides common functionality for all kernels in the distributed network.
 *
 * Copyright (C) 2024 OpenCog Foundation
 */

#include "AgenticKernel.h"
#include <chrono>
#include <sstream>
#include <algorithm>
#include <opencog/util/Logger.h>

// GGML headers
#include "ggml_stub.h"
#define HAVE_GGML 1

namespace opencog { namespace agentic {

// =====================================================
// CognitiveData Implementation
// =====================================================

CognitiveData::~CognitiveData() {
    // Cleanup GGML resources if they exist
    if (tensor_context) {
        ggml_free(tensor_context);
        tensor_context = nullptr;
        tensor_data = nullptr; // tensor_data is freed with context
    }
}

float CognitiveData::get_attention_weight(const std::string& key) const {
    auto it = attention_weights.find(key);
    return it != attention_weights.end() ? it->second : 0.0f;
}

void CognitiveData::set_attention_weight(const std::string& key, float weight) {
    attention_weights[key] = weight;
}

bool CognitiveData::create_tensor_from_symbolic(size_t context_size) {
    if (symbolic_tree.empty()) {
        return false;
    }
    
    // Clean up existing tensor if present
    if (tensor_context) {
        ggml_free(tensor_context);
        tensor_context = nullptr;
        tensor_data = nullptr;
    }
    
    // Initialize GGML context
    struct ggml_init_params params = {
        .mem_size   = context_size,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    
    tensor_context = ggml_init(params);
    if (!tensor_context) {
        return false;
    }
    
    // Create tensor based on tree structure
    int tree_size = symbolic_tree.size();
    if (tree_size == 0) {
        return false;
    }
    
    tensor_data = ggml_new_tensor_1d(tensor_context, GGML_TYPE_F32, tree_size);
    if (!tensor_data) {
        ggml_free(tensor_context);
        tensor_context = nullptr;
        return false;
    }
    
    return update_tensor_from_symbolic();
}

bool CognitiveData::update_tensor_from_symbolic() {
    if (!tensor_data || !tensor_context || symbolic_tree.empty()) {
        return false;
    }
    
    // Fill tensor with encoded tree data
    float* data = (float*)tensor_data->data;
    auto it = symbolic_tree.begin();
    int i = 0;
    int max_elements = ggml_nelements(tensor_data);
    
    for (; i < max_elements && it != symbolic_tree.end(); ++i, ++it) {
        // Enhanced encoding: combine multiple features
        size_t str_len = it->length();
        size_t depth = symbolic_tree.depth(it);
        size_t siblings = symbolic_tree.number_of_siblings(it);
        size_t hash_val = std::hash<std::string>{}(*it);
        
        // Get attention weight for this node if available
        float attention = get_attention_weight(*it);
        
        // Multi-dimensional encoding in a single float
        // Format: attention_weight * 10000 + str_len * 100 + depth * 10 + (hash % 10)
        data[i] = attention * 10000.0f + 
                  static_cast<float>(str_len % 100) * 100.0f + 
                  static_cast<float>(depth % 10) * 10.0f + 
                  static_cast<float>(hash_val % 10);
    }
    
    // Fill remaining slots with padding
    for (; i < max_elements; ++i) {
        data[i] = 0.0f;
    }
    
    return true;
}

bool CognitiveData::sync_symbolic_from_tensor() {
    if (!tensor_data || !tensor_context) {
        return false;
    }
    
    const float* data = (const float*)tensor_data->data;
    int n_elements = ggml_nelements(tensor_data);
    
    // Clear existing tree
    symbolic_tree.clear();
    
    if (n_elements > 0) {
        // Create root
        symbolic_tree.set_head("tensor_sync_root");
        auto root = symbolic_tree.begin();
        
        // Decode tensor data back to tree structure
        for (int i = 0; i < n_elements; ++i) {
            float encoded_val = data[i];
            
            // Skip padding zeros
            if (encoded_val == 0.0f) continue;
            
            // Decode the encoding
            int combined = static_cast<int>(encoded_val);
            float attention = static_cast<float>(combined / 10000) / 10000.0f;
            int str_len = (combined % 10000) / 100;
            int depth = (combined % 100) / 10;
            int hash_remainder = combined % 10;
            
            std::ostringstream node_oss;
            node_oss << "decoded_node_" << i << "_len" << str_len 
                     << "_d" << depth << "_h" << hash_remainder;
            
            std::string node_str = node_oss.str();
            symbolic_tree.append_child(root, node_str);
            
            // Restore attention weight
            if (attention > 0.0f) {
                set_attention_weight(node_str, attention);
            }
        }
    }
    
    return true;
}

// =====================================================
// AgenticKernel Implementation
// =====================================================

AgenticKernel::AgenticKernel(const KernelConfig& config) : config_(config) {
    logger().debug("Creating agentic kernel: %s (type: %s)", 
                   config_.kernel_id.c_str(), config_.kernel_type.c_str());
}

bool AgenticKernel::should_process(const CognitiveData& input, float available_attention) const {
    float estimated_cost = estimate_processing_cost(input);
    
    // Basic attention allocation logic
    if (estimated_cost > available_attention) {
        return false;
    }
    
    if (estimated_cost > config_.max_processing_cost) {
        return false;
    }
    
    // Check activation threshold based on input attention weights
    float total_activation = 0.0f;
    for (const auto& weight : input.attention_weights) {
        total_activation += weight.second;
    }
    
    return total_activation >= config_.base_activation_threshold;
}

void AgenticKernel::register_callback(const std::string& event_type, 
                                      std::function<void(const CognitiveData&)> callback) {
    callbacks_[event_type].push_back(callback);
}

void AgenticKernel::emit_event(const std::string& event_type, const CognitiveData& data) {
    auto it = callbacks_.find(event_type);
    if (it != callbacks_.end()) {
        for (const auto& callback : it->second) {
            callback(data);
        }
    }
    
    // Also notify the kernel registry for network-wide routing
    KernelRegistry::instance().broadcast_event(event_type, data);
}

void AgenticKernel::update_from_feedback(const ProcessingResult& result, float actual_value) {
    // Default implementation: simple learning rate adjustment
    if (config_.enable_learning) {
        float prediction_error = std::abs(result.estimated_value - actual_value);
        
        // Adjust activation threshold based on performance
        if (prediction_error > 0.1f) {
            config_.base_activation_threshold *= 1.01f; // Slightly more conservative
        } else {
            config_.base_activation_threshold *= 0.99f; // Slightly more aggressive
        }
        
        // Keep threshold in reasonable bounds
        config_.base_activation_threshold = std::max(0.1f, 
            std::min(config_.base_activation_threshold, 2.0f));
    }
}

void AgenticKernel::adapt_parameters(const std::map<std::string, float>& performance_metrics) {
    // Default implementation: adjust based on common metrics
    auto efficiency_it = performance_metrics.find("efficiency");
    if (efficiency_it != performance_metrics.end()) {
        float efficiency = efficiency_it->second;
        if (efficiency < 0.5f) {
            config_.max_processing_cost *= 0.9f; // Reduce max cost if inefficient
        } else if (efficiency > 0.8f) {
            config_.max_processing_cost *= 1.1f; // Increase if very efficient
        }
    }
}

void AgenticKernel::update_stats(const ProcessingResult& result, float processing_time) {
    stats_.total_processed++;
    stats_.total_cost += result.processing_cost;
    stats_.total_value += result.estimated_value;
    
    // Update running average of processing time
    float n = static_cast<float>(stats_.total_processed);
    stats_.average_processing_time = ((n - 1) * stats_.average_processing_time + processing_time) / n;
    
    // Activation rate is the ratio of times we actually processed vs. times we were asked
    // This would need to be tracked differently in a real implementation
}

tree<std::string> AgenticKernel::parse_symbolic_input(const std::string& input) const {
    tree<std::string> result;
    
    // Simple parsing - in a real implementation this would be more sophisticated
    if (!input.empty()) {
        std::istringstream iss(input);
        try {
            iss >> result;
        } catch (const std::exception& e) {
            logger().warn("Failed to parse symbolic input: %s", e.what());
            result.set_head(input); // Fall back to single node
        }
    }
    
    return result;
}

ggml_tensor* AgenticKernel::create_tensor_from_tree(const tree<std::string>& tree_data, ggml_context* ctx) const {
    if (!ctx) return nullptr;
    
    // Convert tree to tensor representation
    int tree_size = tree_data.size();
    if (tree_size == 0) return nullptr;
    
    // Create 1D tensor to hold tree elements
    ggml_tensor* tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, tree_size);
    
    // Fill tensor with encoded tree data
    float* data = (float*)tensor->data;
    auto it = tree_data.begin();
    for (int i = 0; i < tree_size && it != tree_data.end(); ++i, ++it) {
        // Enhanced encoding: combine string length, depth, and hash
        size_t str_len = it->length();
        size_t depth = tree_data.depth(it);
        size_t hash_val = std::hash<std::string>{}(*it);
        
        // Encode as: (length * 1000 + depth * 100 + hash % 100) / 1000.0
        data[i] = static_cast<float>(str_len * 1000 + depth * 100 + (hash_val % 100)) / 1000.0f;
    }
    
    return tensor;
}

tree<std::string> AgenticKernel::extract_tree_from_tensor(const ggml_tensor* tensor) const {
    tree<std::string> result;
    
    if (!tensor || tensor->type != GGML_TYPE_F32) {
        return result;
    }
    
    // Extract tree structure from tensor data
    const float* data = (const float*)tensor->data;
    int n_elements = ggml_nelements(tensor);
    
    if (n_elements > 0) {
        std::ostringstream oss;
        oss << "tensor_root";
        result.set_head(oss.str());
        
        auto root = result.begin();
        for (int i = 0; i < n_elements; ++i) {
            // Decode the enhanced encoding
            float encoded_val = data[i];
            int combined = static_cast<int>(encoded_val * 1000);
            int str_len = combined / 1000;
            int depth = (combined % 1000) / 100;
            int hash_remainder = combined % 100;
            
            std::ostringstream node_oss;
            node_oss << "node_" << i << "_len" << str_len << "_d" << depth << "_h" << hash_remainder;
            result.append_child(root, node_oss.str());
        }
    }
    
    return result;
}

// =====================================================
// KernelRegistry Implementation
// =====================================================

KernelRegistry& KernelRegistry::instance() {
    static KernelRegistry instance_;
    return instance_;
}

void KernelRegistry::register_kernel(std::shared_ptr<AgenticKernel> kernel) {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    const std::string& kernel_id = kernel->get_kernel_id();
    kernels_[kernel_id] = kernel;
    
    logger().info("Registered agentic kernel: %s (type: %s)", 
                  kernel_id.c_str(), kernel->get_kernel_type().c_str());
}

void KernelRegistry::unregister_kernel(const std::string& kernel_id) {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    auto it = kernels_.find(kernel_id);
    if (it != kernels_.end()) {
        logger().info("Unregistered agentic kernel: %s", kernel_id.c_str());
        kernels_.erase(it);
        
        // Release any allocated attention
        auto alloc_it = attention_state_.kernel_allocations.find(kernel_id);
        if (alloc_it != attention_state_.kernel_allocations.end()) {
            attention_state_.allocated_attention -= alloc_it->second;
            attention_state_.kernel_allocations.erase(alloc_it);
        }
    }
}

std::shared_ptr<AgenticKernel> KernelRegistry::get_kernel(const std::string& kernel_id) {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    auto it = kernels_.find(kernel_id);
    return it != kernels_.end() ? it->second : nullptr;
}

std::vector<std::shared_ptr<AgenticKernel>> KernelRegistry::get_kernels_by_type(const std::string& type) {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    std::vector<std::shared_ptr<AgenticKernel>> result;
    for (const auto& pair : kernels_) {
        if (pair.second->get_kernel_type() == type) {
            result.push_back(pair.second);
        }
    }
    return result;
}

std::vector<std::string> KernelRegistry::get_all_kernel_ids() const {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    std::vector<std::string> result;
    for (const auto& pair : kernels_) {
        result.push_back(pair.first);
    }
    return result;
}

void KernelRegistry::broadcast_event(const std::string& event_type, const CognitiveData& data) {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    // Broadcast to all registered kernels
    for (const auto& pair : kernels_) {
        pair.second->emit_event(event_type, data);
    }
}

void KernelRegistry::route_data(const CognitiveData& data, const std::string& target_kernel_id) {
    auto kernel = get_kernel(target_kernel_id);
    if (kernel) {
        // Check if kernel should process this data
        float available_attention = attention_state_.total_available_attention - attention_state_.allocated_attention;
        
        if (kernel->should_process(data, available_attention)) {
            auto start_time = std::chrono::high_resolution_clock::now();
            
            ProcessingResult result = kernel->process(data);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            float processing_time = std::chrono::duration<float>(end_time - start_time).count();
            
            // Update kernel stats
            kernel->update_processing_stats(result, processing_time);
            
            // Route to suggested next kernels if any
            for (const std::string& next_kernel_id : result.suggested_next_kernels) {
                route_data(result.output_data, next_kernel_id);
            }
        }
    }
}

bool KernelRegistry::allocate_attention(const std::string& kernel_id, float amount) {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    if (attention_state_.allocated_attention + amount > attention_state_.total_available_attention) {
        return false; // Insufficient attention available
    }
    
    attention_state_.allocated_attention += amount;
    attention_state_.kernel_allocations[kernel_id] += amount;
    return true;
}

void KernelRegistry::release_attention(const std::string& kernel_id, float amount) {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    auto it = attention_state_.kernel_allocations.find(kernel_id);
    if (it != attention_state_.kernel_allocations.end()) {
        float released = std::min(amount, it->second);
        it->second -= released;
        attention_state_.allocated_attention -= released;
        
        if (it->second <= 0.0f) {
            attention_state_.kernel_allocations.erase(it);
        }
    }
}

}} // namespace opencog::agentic