/*
 * opencog/agentic/attention/AttentionAllocationEngine.h
 *
 * ECAN-inspired attention allocation system for the distributed agentic network.
 * Manages economic allocation of cognitive resources across kernel instances.
 *
 * Copyright (C) 2024 OpenCog Foundation
 */

#ifndef _OPENCOG_ATTENTION_ALLOCATION_ENGINE_H
#define _OPENCOG_ATTENTION_ALLOCATION_ENGINE_H

#include <opencog/agentic/kernel/AgenticKernel.h>
#include <queue>
#include <chrono>

namespace opencog { namespace agentic { namespace attention {

/**
 * Attention Value - represents the economic value and priority of cognitive data
 */
struct AttentionValue {
    float short_term_importance = 0.0f;  // Immediate relevance
    float long_term_importance = 0.0f;   // Historical significance  
    float very_long_term_importance = 0.0f; // Deep conceptual importance
    float confidence = 1.0f;             // Confidence in the importance estimates
    std::chrono::system_clock::time_point last_updated;
    
    AttentionValue() : last_updated(std::chrono::system_clock::now()) {}
    
    float total_importance() const {
        return short_term_importance + long_term_importance + very_long_term_importance;
    }
    
    bool is_above_threshold(float threshold) const {
        return total_importance() >= threshold;
    }
    
    void decay(float decay_rate, float time_delta) {
        short_term_importance *= std::exp(-decay_rate * time_delta);
        // Long-term values decay more slowly
        long_term_importance *= std::exp(-decay_rate * time_delta * 0.1f);
        very_long_term_importance *= std::exp(-decay_rate * time_delta * 0.01f);
    }
};

/**
 * Attention allocation request from a kernel
 */
struct AttentionRequest {
    std::string kernel_id;
    std::string request_id;
    CognitiveData data;
    float requested_amount;
    float estimated_processing_cost;
    float estimated_output_value;
    AttentionValue attention_value;
    std::chrono::system_clock::time_point timestamp;
    
    AttentionRequest(const std::string& kid, const CognitiveData& d, float amount)
        : kernel_id(kid), data(d), requested_amount(amount),
          timestamp(std::chrono::system_clock::now()) {
        request_id = kernel_id + "_" + std::to_string(timestamp.time_since_epoch().count());
    }
    
    float priority_score() const {
        float time_bonus = 1.0f; // Could add urgency factor
        return attention_value.total_importance() * 
               (estimated_output_value / std::max(estimated_processing_cost, 0.1f)) * 
               time_bonus;
    }
};

/**
 * ECAN-inspired Economic Attention Allocation Engine
 * 
 * Manages attention as a limited resource in the cognitive economy:
 * 1. Receives attention requests from kernels
 * 2. Evaluates requests based on importance, cost, and expected value
 * 3. Allocates attention using economic principles (auction, market mechanisms)
 * 4. Tracks attention usage and adjusts allocation strategies
 * 5. Implements attention spreading and decay mechanisms
 */
class AttentionAllocationEngine {
public:
    struct AllocationConfig {
        float total_attention_budget = 1000.0f;
        float min_allocation_threshold = 1.0f;
        float attention_decay_rate = 0.01f;    // Per second
        float importance_spreading_rate = 0.1f;
        float allocation_efficiency_target = 0.8f;
        bool enable_market_dynamics = true;
        bool enable_attention_spreading = true;
        size_t max_queue_size = 1000;
    };

public:
    AttentionAllocationEngine(const AllocationConfig& config = AllocationConfig());
    ~AttentionAllocationEngine() = default;

    // Core allocation interface
    bool request_attention(const AttentionRequest& request);
    void release_attention(const std::string& kernel_id, const std::string& request_id, 
                          float actual_cost, float actual_value);
    
    // Attention value management
    void set_attention_value(const std::string& data_id, const AttentionValue& value);
    AttentionValue get_attention_value(const std::string& data_id) const;
    void update_attention_values();
    
    // Economic mechanisms
    void run_attention_auction();
    void update_attention_prices();
    void spread_attention_values();
    
    // Monitoring and statistics
    struct AllocationStats {
        float total_allocated = 0.0f;
        float total_released = 0.0f;
        float average_allocation_efficiency = 0.0f;
        size_t pending_requests = 0;
        size_t processed_requests = 0;
        size_t rejected_requests = 0;
        float attention_utilization_rate = 0.0f;
    };
    
    AllocationStats get_stats() const { return stats_; }
    void reset_stats() { stats_ = AllocationStats(); }
    
    // Configuration
    void update_config(const AllocationConfig& config) { config_ = config; }
    const AllocationConfig& get_config() const { return config_; }
    
    // Network-wide attention state
    float get_available_attention() const;
    float get_total_budget() const { return config_.total_attention_budget; }
    std::vector<AttentionRequest> get_pending_requests() const;

private:
    AllocationConfig config_;
    AllocationStats stats_;
    
    // Request management
    std::priority_queue<AttentionRequest> pending_requests_;
    std::map<std::string, AttentionRequest> active_allocations_;
    std::map<std::string, AttentionValue> attention_values_;
    
    mutable std::mutex allocation_mutex_;
    
    // Economic state
    float current_attention_price_ = 1.0f;
    std::map<std::string, float> kernel_credit_scores_;
    
    // Helper methods
    bool evaluate_request(const AttentionRequest& request);
    void process_allocation_queue();
    float calculate_allocation_amount(const AttentionRequest& request);
    void update_kernel_performance(const std::string& kernel_id, float efficiency);
    
    // Attention spreading mechanisms
    void spread_to_related_data(const std::string& data_id, const AttentionValue& source_value);
    std::vector<std::string> find_related_data(const std::string& data_id) const;
    
    // Market dynamics
    void adjust_attention_price(float demand_supply_ratio);
    float calculate_market_allocation(const AttentionRequest& request);
};

/**
 * Attention-aware cognitive data processor
 * Integrates attention allocation with data processing pipelines
 */
class AttentionAwareProcessor {
public:
    AttentionAwareProcessor(std::shared_ptr<AttentionAllocationEngine> engine);
    
    // Process data with attention allocation
    ProcessingResult process_with_attention(
        std::shared_ptr<AgenticKernel> kernel,
        const CognitiveData& input,
        float requested_attention = 0.0f);
    
    // Batch processing with attention management
    std::vector<ProcessingResult> process_batch(
        std::shared_ptr<AgenticKernel> kernel,
        const std::vector<CognitiveData>& inputs);
    
    // Attention-guided data routing
    std::vector<std::string> suggest_processing_kernels(const CognitiveData& data);

private:
    std::shared_ptr<AttentionAllocationEngine> attention_engine_;
    
    float estimate_required_attention(std::shared_ptr<AgenticKernel> kernel, 
                                     const CognitiveData& input);
    AttentionValue calculate_data_attention_value(const CognitiveData& data);
};

/**
 * Attention spreading algorithms
 */
namespace spreading {
    
    // Hebbian-like attention spreading
    void hebbian_spread(std::map<std::string, AttentionValue>& attention_map,
                       const std::string& source_id,
                       const std::vector<std::string>& target_ids,
                       float spread_rate);
    
    // Importance-based spreading
    void importance_spread(std::map<std::string, AttentionValue>& attention_map,
                          float global_spread_rate);
    
    // Network topology-aware spreading
    void topology_aware_spread(std::map<std::string, AttentionValue>& attention_map,
                              const std::map<std::string, std::vector<std::string>>& topology,
                              float spread_rate);
}

}}} // namespace opencog::agentic::attention

// Custom comparator for priority queue (higher priority first)
namespace std {
    template<>
    struct greater<opencog::agentic::attention::AttentionRequest> {
        bool operator()(const opencog::agentic::attention::AttentionRequest& a,
                       const opencog::agentic::attention::AttentionRequest& b) const {
            return a.priority_score() < b.priority_score();
        }
    };
}

#endif // _OPENCOG_ATTENTION_ALLOCATION_ENGINE_H