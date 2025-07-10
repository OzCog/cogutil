/*
 * opencog/agentic/kernel/AgenticKernel.h
 *
 * Base interface for Agentic Grammar Kernels in the distributed cognitive network.
 * Each kernel processes cognitive representations (tensors, symbols, hypergraphs)
 * and participates in the attention allocation economy.
 *
 * Copyright (C) 2024 OpenCog Foundation
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License v3 as
 * published by the Free Software Foundation and including the exceptions
 * at http://opencog.org/wiki/Licenses
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program; if not, write to:
 * Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifndef _OPENCOG_AGENTIC_KERNEL_H
#define _OPENCOG_AGENTIC_KERNEL_H

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <map>
#include <mutex>

#include <opencog/util/tree.h>

// Forward declarations for GGML integration
struct ggml_context;
struct ggml_tensor;

namespace opencog { namespace agentic {

/**
 * Base representation for cognitive data flowing through the kernel network.
 * Unifies symbolic trees, tensor representations, and metadata.
 */
struct CognitiveData {
    // Symbolic representation using opencog tree structure
    tree<std::string> symbolic_tree;
    
    // Tensor representation (GGML integration)
    ggml_tensor* tensor_data = nullptr;
    ggml_context* tensor_context = nullptr;
    
    // Metadata for attention allocation and routing
    std::map<std::string, float> attention_weights;
    std::map<std::string, std::string> metadata;
    
    // Unique identifier for tracking through the network
    std::string data_id;
    
    CognitiveData() = default;
    CognitiveData(const tree<std::string>& tree_data) : symbolic_tree(tree_data) {}
    
    ~CognitiveData();
    
    // Utility methods
    bool has_tensor() const { return tensor_data != nullptr; }
    bool has_symbolic() const { return !symbolic_tree.empty(); }
    float get_attention_weight(const std::string& key) const;
    void set_attention_weight(const std::string& key, float weight);
    
    // GGML tensor management
    bool create_tensor_from_symbolic(size_t context_size = 16 * 1024 * 1024);
    bool update_tensor_from_symbolic();
    bool sync_symbolic_from_tensor();
};

/**
 * Result of processing by an agentic kernel.
 * Contains the transformed data plus feedback for attention allocation.
 */
struct ProcessingResult {
    CognitiveData output_data;
    float processing_cost = 0.0f;
    float estimated_value = 0.0f;
    bool requires_further_processing = false;
    std::vector<std::string> suggested_next_kernels;
    
    ProcessingResult() = default;
    ProcessingResult(const CognitiveData& data) : output_data(data) {}
};

/**
 * Configuration for kernel behavior and attention allocation parameters.
 */
struct KernelConfig {
    std::string kernel_id;
    std::string kernel_type;
    float base_activation_threshold = 0.5f;
    float max_processing_cost = 100.0f;
    bool enable_learning = true;
    std::map<std::string, std::string> custom_params;
    
    KernelConfig(const std::string& id, const std::string& type) 
        : kernel_id(id), kernel_type(type) {}
};

/**
 * Base interface for all Agentic Grammar Kernels.
 * 
 * Each kernel is a microservice that:
 * 1. Processes cognitive data (symbols, tensors, hypergraphs)
 * 2. Participates in attention allocation economy
 * 3. Communicates asynchronously with other kernels
 * 4. Adapts behavior based on feedback
 */
class AgenticKernel {
public:
    AgenticKernel(const KernelConfig& config);
    virtual ~AgenticKernel() = default;

    // Core processing interface
    virtual ProcessingResult process(const CognitiveData& input) = 0;
    
    // Attention allocation interface
    virtual float estimate_processing_cost(const CognitiveData& input) const = 0;
    virtual float estimate_output_value(const CognitiveData& input) const = 0;
    virtual bool should_process(const CognitiveData& input, float available_attention) const;
    
    // Network communication interface
    virtual void register_callback(const std::string& event_type, 
                                   std::function<void(const CognitiveData&)> callback);
    virtual void emit_event(const std::string& event_type, const CognitiveData& data);
    
    // Learning and adaptation interface
    virtual void update_from_feedback(const ProcessingResult& result, float actual_value);
    virtual void adapt_parameters(const std::map<std::string, float>& performance_metrics);
    
    // Configuration and metadata
    const KernelConfig& get_config() const { return config_; }
    std::string get_kernel_id() const { return config_.kernel_id; }
    std::string get_kernel_type() const { return config_.kernel_type; }
    
    // Statistics and monitoring
    struct KernelStats {
        size_t total_processed = 0;
        float total_cost = 0.0f;
        float total_value = 0.0f;
        float average_processing_time = 0.0f;
        float activation_rate = 0.0f;
    };
    
    const KernelStats& get_stats() const { return stats_; }
    void reset_stats() { stats_ = KernelStats(); }
    
    // Public method for updating stats (used by registry)
    void update_processing_stats(const ProcessingResult& result, float processing_time) {
        update_stats(result, processing_time);
    }

protected:
    KernelConfig config_;
    KernelStats stats_;
    
    // Event callback system
    std::map<std::string, std::vector<std::function<void(const CognitiveData&)>>> callbacks_;
    
    // Helper methods for subclasses
    void update_stats(const ProcessingResult& result, float processing_time);
    tree<std::string> parse_symbolic_input(const std::string& input) const;
    ggml_tensor* create_tensor_from_tree(const tree<std::string>& tree_data, ggml_context* ctx) const;
    tree<std::string> extract_tree_from_tensor(const ggml_tensor* tensor) const;
};

/**
 * Registry for managing kernel instances in the distributed network.
 * Handles kernel discovery, lifecycle, and communication routing.
 */
class KernelRegistry {
public:
    static KernelRegistry& instance();
    
    // Kernel lifecycle management
    void register_kernel(std::shared_ptr<AgenticKernel> kernel);
    void unregister_kernel(const std::string& kernel_id);
    std::shared_ptr<AgenticKernel> get_kernel(const std::string& kernel_id);
    std::vector<std::shared_ptr<AgenticKernel>> get_kernels_by_type(const std::string& type);
    
    // Network-wide operations
    std::vector<std::string> get_all_kernel_ids() const;
    void broadcast_event(const std::string& event_type, const CognitiveData& data);
    void route_data(const CognitiveData& data, const std::string& target_kernel_id);
    
    // Attention allocation coordination
    struct NetworkAttentionState {
        float total_available_attention = 1000.0f;
        float allocated_attention = 0.0f;
        std::map<std::string, float> kernel_allocations;
    };
    
    NetworkAttentionState get_attention_state() const { return attention_state_; }
    bool allocate_attention(const std::string& kernel_id, float amount);
    void release_attention(const std::string& kernel_id, float amount);

private:
    KernelRegistry() = default;
    
    std::map<std::string, std::shared_ptr<AgenticKernel>> kernels_;
    NetworkAttentionState attention_state_;
    mutable std::mutex registry_mutex_;
};

}} // namespace opencog::agentic

#endif // _OPENCOG_AGENTIC_KERNEL_H