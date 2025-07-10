/*
 * opencog/agentic/kernel/TensorMapper.h
 *
 * TensorMapper - Maps cognitive objects to introspectable tensor shapes
 * Compatible with GGML for efficient neural-symbolic integration
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

#ifndef _OPENCOG_TENSOR_MAPPER_H
#define _OPENCOG_TENSOR_MAPPER_H

#include "AgenticKernel.h"
#include "TensorShape.h"
#include <opencog/util/tree.h>

// Forward declarations for GGML integration
struct ggml_context;
struct ggml_tensor;

namespace opencog { namespace agentic {

// Forward declarations  
using ::opencog::tree;

namespace opencog { namespace agentic {

/**
 * TensorMapper - Cognitive object to tensor mapping kernel.
 * 
 * This kernel converts symbolic cognitive representations (trees, hypergraphs,
 * links) into optimized tensor representations suitable for GGML operations
 * and neural processing. It maintains the semantic structure while enabling
 * efficient numerical computation.
 * 
 * Key Features:
 * - Prime-factorized tensor shapes for optimal memory layout
 * - Attention-aware dimension allocation
 * - Reversible mappings (tensor → symbolic and back)
 * - GGML-compatible tensor generation
 * - Adaptive mapping strategies based on data characteristics
 */
class TensorMapper : public AgenticKernel {
public:
    enum class MappingStrategy {
        STRUCTURAL,      // Preserve structural relationships in tensor layout
        SEMANTIC,        // Focus on semantic similarity in embedding space
        ATTENTION_WEIGHTED, // Prioritize high-attention elements
        PRIME_FACTORIZED,   // Optimize for prime-factorized dimensions
        GGML_OPTIMIZED,     // Optimize specifically for GGML operations
        HYBRID              // Adaptive combination of strategies
    };
    
    struct TensorMappingConfig {
        MappingStrategy strategy = MappingStrategy::HYBRID;
        size_t max_tensor_dimension = 1024; // Maximum dimension size
        bool preserve_structure = true;      // Maintain symbolic structure
        bool optimize_for_ggml = true;       // GGML-specific optimizations
        float attention_threshold = 0.1f;    // Minimum attention weight to include
        bool enable_compression = true;      // Allow lossy compression for efficiency
        std::map<std::string, float> mapping_weights; // Strategy-specific weights
    };

public:
    TensorMapper(const KernelConfig& config, const TensorMappingConfig& mapping_config);
    virtual ~TensorMapper() = default;

    // AgenticKernel interface implementation
    ProcessingResult process(const CognitiveData& input) override;
    float estimate_processing_cost(const CognitiveData& input) const override;
    float estimate_output_value(const CognitiveData& input) const override;

    // TensorMapper specific methods
    void set_mapping_strategy(MappingStrategy strategy);
    void set_attention_threshold(float threshold);
    void update_mapping_weights(const std::map<std::string, float>& weights);
    
    // Core mapping operations
    TensorShape map_symbolic_to_shape(const tree<std::string>& symbolic_data) const;
    ggml_tensor* create_tensor_from_cognitive_data(const CognitiveData& data, ggml_context* ctx) const;
    CognitiveData extract_cognitive_data_from_tensor(const ggml_tensor* tensor) const;
    
    // Advanced mapping functions  
    TensorShape optimize_shape_for_attention(const TensorShape& base_shape, 
                                            const std::map<std::string, float>& attention_weights) const;
    TensorShape apply_prime_factorization(const TensorShape& shape) const;
    
    // Bidirectional mapping validation
    bool validate_mapping_reversibility(const CognitiveData& original) const;
    float calculate_mapping_fidelity(const CognitiveData& original, const CognitiveData& reconstructed) const;
    
    // Performance optimization
    std::vector<TensorShape> suggest_tensor_optimizations(const std::vector<TensorShape>& shapes) const;
    void cache_frequent_mappings(bool enable) { enable_mapping_cache_ = enable; }
    void clear_mapping_cache() { mapping_cache_.clear(); }

private:
    TensorMappingConfig mapping_config_;
    bool enable_mapping_cache_;
    mutable std::map<std::string, TensorShape> mapping_cache_;
    
    // Core mapping algorithms
    TensorShape map_structural(const tree<std::string>& symbolic_data) const;
    TensorShape map_semantic(const tree<std::string>& symbolic_data) const;
    TensorShape map_attention_weighted(const tree<std::string>& symbolic_data, 
                                      const std::map<std::string, float>& attention_weights) const;
    TensorShape map_prime_factorized(const tree<std::string>& symbolic_data) const;
    TensorShape map_ggml_optimized(const tree<std::string>& symbolic_data) const;
    TensorShape map_hybrid(const tree<std::string>& symbolic_data, 
                          const std::map<std::string, float>& attention_weights) const;
    
    // Helper functions
    std::vector<size_t> calculate_structural_dimensions(const tree<std::string>& data) const;
    std::vector<float> encode_semantic_features(const tree<std::string>& data) const;
    std::vector<size_t> optimize_dimensions_for_ggml(const std::vector<size_t>& raw_dimensions) const;
    float calculate_attention_scaling_factor(const std::map<std::string, float>& attention_weights) const;
    
    // Validation and quality metrics
    bool is_shape_ggml_compatible(const TensorShape& shape) const;
    float calculate_memory_efficiency(const TensorShape& shape) const;
    float calculate_computation_efficiency(const TensorShape& shape) const;
    
    // Caching utilities
    std::string generate_cache_key(const tree<std::string>& data, 
                                  const std::map<std::string, float>& attention_weights) const;
    void update_cache(const std::string& key, const TensorShape& shape) const;
    bool get_from_cache(const std::string& key, TensorShape& shape) const;
};

/**
 * TensorMapperFactory - Factory for creating specialized tensor mappers.
 */
class TensorMapperFactory {
public:
    static std::shared_ptr<TensorMapper> create_structural_mapper(const std::string& kernel_id);
    static std::shared_ptr<TensorMapper> create_semantic_mapper(const std::string& kernel_id);
    static std::shared_ptr<TensorMapper> create_attention_mapper(const std::string& kernel_id);
    static std::shared_ptr<TensorMapper> create_ggml_optimized_mapper(const std::string& kernel_id);
    static std::shared_ptr<TensorMapper> create_hybrid_mapper(const std::string& kernel_id);
    
    // Specialized configurations
    static TensorMapper::TensorMappingConfig create_config_for_language_processing();
    static TensorMapper::TensorMappingConfig create_config_for_graph_analysis();
    static TensorMapper::TensorMappingConfig create_config_for_attention_networks();
    static TensorMapper::TensorMappingConfig create_config_for_reasoning_tasks();
};

/**
 * MappingQualityAnalyzer - Analyzes and optimizes tensor mapping quality.
 */
class MappingQualityAnalyzer {
public:
    struct MappingQualityReport {
        float fidelity_score;           // How well the mapping preserves information
        float efficiency_score;         // Computational efficiency of the mapping
        float memory_score;             // Memory efficiency
        float ggml_compatibility_score; // Compatibility with GGML operations
        std::vector<std::string> recommendations; // Improvement suggestions
    };
    
    static MappingQualityReport analyze_mapping(const CognitiveData& original, 
                                               const TensorShape& mapped_shape,
                                               const TensorMapper& mapper);
    
    static std::vector<TensorShape> optimize_shape_collection(const std::vector<TensorShape>& shapes);
    static TensorMapper::MappingStrategy suggest_optimal_strategy(const CognitiveData& sample_data);
    static float benchmark_mapping_performance(const TensorMapper& mapper, 
                                              const std::vector<CognitiveData>& test_data);
};

}} // namespace opencog::agentic

#endif // _OPENCOG_TENSOR_MAPPER_H