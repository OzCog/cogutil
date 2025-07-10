/*
 * opencog/agentic/attention/AttentionAllocationEngine.cc
 *
 * Implementation of ECAN-inspired attention allocation with GGML tensor backend.
 */

#include "AttentionAllocationEngine.h"
#include <opencog/util/Logger.h>
#include <algorithm>
#include <numeric>
#include <cmath>

// GGML headers
#include "../kernel/ggml_stub.h"

namespace opencog { namespace agentic { namespace attention {

// =====================================================
// GGMLAttentionValue Implementation
// =====================================================

GGMLAttentionValue::GGMLAttentionValue() {
    initialize_tensor();
}

GGMLAttentionValue::~GGMLAttentionValue() {
    cleanup_tensor();
}

void GGMLAttentionValue::set_importance_values(float sti, float lti, float vlti, float confidence) {
    if (!tensor_data_) return;
    
    float* data = (float*)tensor_data_->data;
    data[0] = sti;
    data[1] = lti;
    data[2] = vlti;
    data[3] = confidence;
    
    update_timestamp();
}

AttentionValue GGMLAttentionValue::get_attention_value() const {
    AttentionValue av;
    if (!tensor_data_) return av;
    
    const float* data = (const float*)tensor_data_->data;
    av.short_term_importance = data[0];
    av.long_term_importance = data[1];
    av.very_long_term_importance = data[2];
    av.confidence = data[3];
    av.last_updated = last_updated_;
    
    return av;
}

void GGMLAttentionValue::apply_decay(float decay_rate, float time_delta) {
    if (!tensor_data_ || !decay_context_) return;
    
    // Create decay tensor
    ggml_tensor* decay_tensor = ggml_new_tensor_1d(decay_context_, GGML_TYPE_F32, 4);
    float* decay_data = (float*)decay_tensor->data;
    
    // Different decay rates for different importance types
    decay_data[0] = std::exp(-decay_rate * time_delta);           // STI
    decay_data[1] = std::exp(-decay_rate * time_delta * 0.1f);    // LTI
    decay_data[2] = std::exp(-decay_rate * time_delta * 0.01f);   // VLTI
    decay_data[3] = 1.0f;                                         // Confidence doesn't decay
    
    // Apply element-wise multiplication using GGML
    ggml_tensor* result = ggml_mul(decay_context_, tensor_data_, decay_tensor);
    
    // Copy result back to original tensor
    float* result_data = (float*)result->data;
    float* original_data = (float*)tensor_data_->data;
    for (int i = 0; i < 4; ++i) {
        original_data[i] = result_data[i];
    }
    
    update_timestamp();
}

float GGMLAttentionValue::calculate_total_importance() const {
    if (!tensor_data_) return 0.0f;
    
    const float* data = (const float*)tensor_data_->data;
    return data[0] + data[1] + data[2]; // STI + LTI + VLTI
}

bool GGMLAttentionValue::is_above_threshold(float threshold) const {
    return calculate_total_importance() >= threshold;
}

void GGMLAttentionValue::initialize_tensor() {
    // Initialize main tensor context
    struct ggml_init_params params = {
        .mem_size   = 1024 * 1024, // 1MB for attention tensors
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    
    tensor_context_ = ggml_init(params);
    decay_context_ = ggml_init(params);
    
    if (tensor_context_) {
        // Create 4-element tensor: [STI, LTI, VLTI, confidence]
        tensor_data_ = ggml_new_tensor_1d(tensor_context_, GGML_TYPE_F32, 4);
        
        if (tensor_data_) {
            // Initialize with default values
            float* data = (float*)tensor_data_->data;
            data[0] = 0.0f; // STI
            data[1] = 0.0f; // LTI
            data[2] = 0.0f; // VLTI
            data[3] = 1.0f; // confidence
        }
    }
    
    update_timestamp();
}

void GGMLAttentionValue::cleanup_tensor() {
    if (tensor_context_) {
        ggml_free(tensor_context_);
        tensor_context_ = nullptr;
        tensor_data_ = nullptr;
    }
    if (decay_context_) {
        ggml_free(decay_context_);
        decay_context_ = nullptr;
    }
}

void GGMLAttentionValue::update_timestamp() {
    last_updated_ = std::chrono::system_clock::now();
}

// =====================================================
// AttentionAllocationEngine Implementation
// =====================================================

AttentionAllocationEngine::AttentionAllocationEngine(const AllocationConfig& config)
    : config_(config) {
    
    // Initialize GGML context for attention operations
    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024, // 16MB for attention computations
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    
    attention_compute_context_ = ggml_init(params);
    
    opencog::logger().info("Initialized AttentionAllocationEngine with budget: %.2f", config_.total_attention_budget);
}

AttentionAllocationEngine::AttentionAllocationEngine() 
    : AttentionAllocationEngine(AllocationConfig()) {
}

AttentionAllocationEngine::~AttentionAllocationEngine() {
    if (attention_compute_context_) {
        ggml_free(attention_compute_context_);
    }
}

bool AttentionAllocationEngine::request_attention(const AttentionRequest& request) {
    std::lock_guard<std::mutex> lock(allocation_mutex_);
    
    // Evaluate request
    if (!evaluate_request(request)) {
        stats_.rejected_requests++;
        return false;
    }
    
    // Calculate allocation amount using GGML operations
    float allocation_amount = calculate_allocation_amount_ggml(request);
    
    if (allocation_amount <= 0 || allocation_amount > get_available_attention()) {
        stats_.rejected_requests++;
        return false;
    }
    
    // Allocate attention
    stats_.total_allocated += allocation_amount;
    active_allocations_[request.request_id] = request;
    
    // Update attention values using GGML
    update_attention_value_ggml(request.data.data_id, request.attention_value);
    
    stats_.processed_requests++;
    
    opencog::logger().debug("Allocated %.2f attention to kernel %s", allocation_amount, request.kernel_id.c_str());
    
    return true;
}

void AttentionAllocationEngine::release_attention(const std::string& kernel_id, 
                                                 const std::string& request_id,
                                                 float actual_cost, 
                                                 float actual_value) {
    std::lock_guard<std::mutex> lock(allocation_mutex_);
    
    auto it = active_allocations_.find(request_id);
    if (it != active_allocations_.end()) {
        float allocated_amount = it->second.requested_amount;
        stats_.total_released += allocated_amount;
        
        // Calculate efficiency
        float predicted_cost = it->second.estimated_processing_cost;
        float predicted_value = it->second.estimated_output_value;
        
        float cost_efficiency = predicted_cost > 0 ? std::min(predicted_cost / actual_cost, 2.0f) : 1.0f;
        float value_efficiency = predicted_value > 0 ? actual_value / predicted_value : 1.0f;
        float overall_efficiency = (cost_efficiency + value_efficiency) / 2.0f;
        
        // Update kernel performance
        update_kernel_performance(kernel_id, overall_efficiency);
        
        // Update running efficiency average
        float n = static_cast<float>(stats_.processed_requests);
        stats_.average_allocation_efficiency = ((n - 1) * stats_.average_allocation_efficiency + overall_efficiency) / n;
        
        active_allocations_.erase(it);
        
        opencog::logger().debug("Released attention for kernel %s, efficiency: %.2f", kernel_id.c_str(), overall_efficiency);
    }
}

void AttentionAllocationEngine::update_attention_values() {
    std::lock_guard<std::mutex> lock(allocation_mutex_);
    
    auto current_time = std::chrono::system_clock::now();
    
    for (auto& pair : ggml_attention_values_) {
        auto time_diff = current_time - pair.second->get_attention_value().last_updated;
        float time_delta = std::chrono::duration<float>(time_diff).count();
        
        if (time_delta > 1.0f) { // Only decay if more than 1 second has passed
            pair.second->apply_decay(config_.attention_decay_rate, time_delta);
        }
    }
    
    // Spread attention values
    if (config_.enable_attention_spreading) {
        spread_attention_values_ggml();
    }
}

void AttentionAllocationEngine::run_attention_auction() {
    std::lock_guard<std::mutex> lock(allocation_mutex_);
    
    if (pending_requests_.empty()) return;
    
    // Use GGML to compute auction results
    size_t num_requests = pending_requests_.size();
    if (num_requests == 0) return;
    
    // Create tensors for auction computation
    ggml_tensor* bid_tensor = ggml_new_tensor_1d(attention_compute_context_, GGML_TYPE_F32, num_requests);
    ggml_tensor* priority_tensor = ggml_new_tensor_1d(attention_compute_context_, GGML_TYPE_F32, num_requests);
    
    float* bid_data = (float*)bid_tensor->data;
    float* priority_data = (float*)priority_tensor->data;
    
    // Fill tensors with request data
    std::vector<AttentionRequest> requests;
    size_t i = 0;
    while (!pending_requests_.empty() && i < num_requests) {
        AttentionRequest req = pending_requests_.top();
        pending_requests_.pop();
        requests.push_back(req);
        
        bid_data[i] = req.requested_amount;
        priority_data[i] = req.priority_score();
        i++;
    }
    
    // Calculate auction winners using GGML operations
    ggml_tensor* score_tensor = ggml_mul(attention_compute_context_, bid_tensor, priority_tensor);
    
    // Process auction results
    float* scores = (float*)score_tensor->data;
    std::vector<std::pair<float, size_t>> scored_requests;
    
    for (size_t j = 0; j < requests.size(); ++j) {
        scored_requests.push_back({scores[j], j});
    }
    
    // Sort by score (descending)
    std::sort(scored_requests.begin(), scored_requests.end(), std::greater<std::pair<float, size_t>>());
    
    // Allocate to top scoring requests within budget
    float remaining_budget = get_available_attention();
    
    for (const auto& scored_request : scored_requests) {
        size_t idx = scored_request.second;
        const AttentionRequest& req = requests[idx];
        
        if (req.requested_amount <= remaining_budget) {
            // Process this request
            request_attention(req);
            remaining_budget -= req.requested_amount;
        } else {
            // Put back in queue for next round
            pending_requests_.push(req);
        }
    }
    
    opencog::logger().debug("Processed attention auction with %zu requests", requests.size());
}

float AttentionAllocationEngine::get_available_attention() const {
    return config_.total_attention_budget - stats_.total_allocated + stats_.total_released;
}

// Private implementation methods

float AttentionAllocationEngine::calculate_allocation_amount_ggml(const AttentionRequest& request) const {
    if (!attention_compute_context_) {
        return request.requested_amount; // Fallback
    }
    
    // Create computation graph for allocation calculation
    ggml_tensor* request_tensor = ggml_new_tensor_1d(attention_compute_context_, GGML_TYPE_F32, 4);
    float* data = (float*)request_tensor->data;
    
    data[0] = request.requested_amount;
    data[1] = request.estimated_processing_cost;
    data[2] = request.estimated_output_value;
    data[3] = request.attention_value.total_importance();
    
    // Apply allocation formula using GGML
    // allocation = requested * (value / cost) * importance_factor * market_factor
    
    ggml_tensor* value_cost_ratio = ggml_div(attention_compute_context_, 
                                           ggml_view_1d(attention_compute_context_, request_tensor, 1, 2 * sizeof(float)),
                                           ggml_view_1d(attention_compute_context_, request_tensor, 1, 1 * sizeof(float)));
    
    ggml_tensor* importance_factor = ggml_view_1d(attention_compute_context_, request_tensor, 1, 3 * sizeof(float));
    ggml_tensor* requested_amount = ggml_view_1d(attention_compute_context_, request_tensor, 1, 0);
    
    // Combine factors
    ggml_tensor* efficiency = ggml_mul(attention_compute_context_, value_cost_ratio, importance_factor);
    ggml_tensor* allocation = ggml_mul(attention_compute_context_, requested_amount, efficiency);
    
    // Apply market dynamics
    float market_factor = std::min(1.0f, get_available_attention() / config_.total_attention_budget);
    ggml_tensor* market_tensor = ggml_new_f32(attention_compute_context_, market_factor);
    ggml_tensor* final_allocation = ggml_mul(attention_compute_context_, allocation, market_tensor);
    
    float result = ggml_get_f32_1d(final_allocation, 0);
    
    // Clamp to reasonable bounds
    return std::min(std::max(result, config_.min_allocation_threshold), 
                   config_.total_attention_budget * 0.5f);
}

void AttentionAllocationEngine::update_attention_value_ggml(const std::string& data_id, const AttentionValue& value) {
    auto it = ggml_attention_values_.find(data_id);
    if (it == ggml_attention_values_.end()) {
        ggml_attention_values_[data_id] = std::make_unique<GGMLAttentionValue>();
    }
    
    ggml_attention_values_[data_id]->set_importance_values(
        value.short_term_importance,
        value.long_term_importance,
        value.very_long_term_importance,
        value.confidence
    );
}

void AttentionAllocationEngine::spread_attention_values_ggml() {
    // Implement attention spreading using GGML tensor operations
    if (ggml_attention_values_.size() < 2) return;
    
    size_t num_values = ggml_attention_values_.size();
    
    // Create spreading matrix
    ggml_tensor* attention_matrix = ggml_new_tensor_2d(attention_compute_context_, GGML_TYPE_F32, num_values, 4);
    ggml_tensor* spreading_weights = ggml_new_tensor_2d(attention_compute_context_, GGML_TYPE_F32, num_values, num_values);
    
    // Fill attention matrix with current values
    float* attention_data = (float*)attention_matrix->data;
    size_t i = 0;
    std::vector<std::string> data_ids;
    
    for (const auto& pair : ggml_attention_values_) {
        data_ids.push_back(pair.first);
        AttentionValue av = pair.second->get_attention_value();
        
        attention_data[i * 4 + 0] = av.short_term_importance;
        attention_data[i * 4 + 1] = av.long_term_importance;
        attention_data[i * 4 + 2] = av.very_long_term_importance;
        attention_data[i * 4 + 3] = av.confidence;
        i++;
    }
    
    // Create simple spreading weights (could be more sophisticated)
    float* weight_data = (float*)spreading_weights->data;
    for (size_t row = 0; row < num_values; ++row) {
        for (size_t col = 0; col < num_values; ++col) {
            if (row == col) {
                weight_data[row * num_values + col] = 1.0f - config_.importance_spreading_rate;
            } else {
                weight_data[row * num_values + col] = config_.importance_spreading_rate / (num_values - 1);
            }
        }
    }
    
    // Apply spreading: new_attention = spreading_weights * attention_matrix
    ggml_tensor* new_attention = ggml_mul_mat(attention_compute_context_, spreading_weights, attention_matrix);
    
    // Update attention values with spread results
    float* new_data = (float*)new_attention->data;
    for (size_t j = 0; j < num_values; ++j) {
        AttentionValue new_av;
        new_av.short_term_importance = new_data[j * 4 + 0];
        new_av.long_term_importance = new_data[j * 4 + 1];
        new_av.very_long_term_importance = new_data[j * 4 + 2];
        new_av.confidence = new_data[j * 4 + 3];
        
        update_attention_value_ggml(data_ids[j], new_av);
    }
}

bool AttentionAllocationEngine::evaluate_request(const AttentionRequest& request) {
    // Basic request validation
    if (request.requested_amount <= 0 || 
        request.requested_amount > config_.total_attention_budget) {
        return false;
    }
    
    if (request.estimated_processing_cost <= 0) {
        return false;
    }
    
    // Check if request exceeds available attention
    if (request.requested_amount > get_available_attention()) {
        return false;
    }
    
    // Check kernel credit score
    auto credit_it = kernel_credit_scores_.find(request.kernel_id);
    if (credit_it != kernel_credit_scores_.end() && credit_it->second < 0.1f) {
        return false; // Poor performing kernel
    }
    
    return true;
}

void AttentionAllocationEngine::update_kernel_performance(const std::string& kernel_id, float efficiency) {
    // Update kernel credit score based on performance
    float current_score = kernel_credit_scores_[kernel_id];
    float learning_rate = 0.1f;
    
    kernel_credit_scores_[kernel_id] = current_score * (1 - learning_rate) + efficiency * learning_rate;
    
    // Clamp to reasonable bounds
    kernel_credit_scores_[kernel_id] = std::max(0.0f, std::min(kernel_credit_scores_[kernel_id], 2.0f));
}

}}} // namespace opencog::agentic::attention