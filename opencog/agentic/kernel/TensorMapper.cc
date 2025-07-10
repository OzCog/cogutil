/*
 * opencog/agentic/kernel/TensorMapper.cc
 *
 * Implementation of TensorMapper - Maps cognitive objects to tensor shapes.
 */

#include "TensorMapper.h"
#include <opencog/util/Logger.h>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>

// GGML headers (if available)
#ifdef HAVE_GGML
#include <ggml.h>
#endif

namespace opencog { namespace agentic {

TensorMapper::TensorMapper(const KernelConfig& config, const TensorMappingConfig& mapping_config)
    : AgenticKernel(config), mapping_config_(mapping_config), enable_mapping_cache_(true) {
    
    logger().info("TensorMapper initialized: %s (strategy: %d)", 
                  config.kernel_id.c_str(), static_cast<int>(mapping_config_.strategy));
}

ProcessingResult TensorMapper::process(const CognitiveData& input) {
    ProcessingResult result;
    
    if (!input.has_symbolic()) {
        logger().warn("TensorMapper received input without symbolic data");
        return result;
    }
    
    // Map symbolic data to tensor shape
    TensorShape mapped_shape;
    std::map<std::string, float> attention_weights = input.attention_weights;
    
    switch (mapping_config_.strategy) {
        case MappingStrategy::STRUCTURAL:
            mapped_shape = map_structural(input.symbolic_tree);
            break;
        case MappingStrategy::SEMANTIC:
            mapped_shape = map_semantic(input.symbolic_tree);
            break;
        case MappingStrategy::ATTENTION_WEIGHTED:
            mapped_shape = map_attention_weighted(input.symbolic_tree, attention_weights);
            break;
        case MappingStrategy::PRIME_FACTORIZED:
            mapped_shape = map_prime_factorized(input.symbolic_tree);
            break;
        case MappingStrategy::GGML_OPTIMIZED:
            mapped_shape = map_ggml_optimized(input.symbolic_tree);
            break;
        case MappingStrategy::HYBRID:
        default:
            mapped_shape = map_hybrid(input.symbolic_tree, attention_weights);
            break;
    }
    
    // Create output cognitive data with tensor representation
    result.output_data = input; // Copy input
    result.output_data.data_id = "tensor_mapped_" + input.data_id;
    result.output_data.metadata["processing_stage"] = "tensor_mapped";
    result.output_data.metadata["tensor_shape"] = mapped_shape.to_string();
    result.output_data.metadata["mapping_strategy"] = std::to_string(static_cast<int>(mapping_config_.strategy));
    
    // Add tensor shape metadata
    auto shape_metadata = mapped_shape.to_metadata();
    for (const auto& pair : shape_metadata) {
        result.output_data.metadata["tensor_" + pair.first] = pair.second;
    }
    
    // Create GGML tensor if possible
    if (mapped_shape.is_ggml_compatible()) {
#ifdef HAVE_GGML
        // In a real implementation, we would create the actual tensor here
        result.output_data.metadata["ggml_compatible"] = "true";
        result.output_data.metadata["estimated_memory"] = std::to_string(mapped_shape.estimate_ggml_memory_size());
#endif
    }
    
    // Set processing costs and values
    result.processing_cost = static_cast<float>(mapped_shape.get_total_elements()) * 0.01f;
    result.estimated_value = static_cast<float>(mapped_shape.calculate_degrees_of_freedom()) * 0.1f;
    
    // Suggest next processing stages
    result.suggested_next_kernels.push_back("attention_allocation");
    if (mapped_shape.is_ggml_compatible()) {
        result.suggested_next_kernels.push_back("ggml_processor");
    }
    
    logger().debug("TensorMapper mapped %zu symbolic elements to tensor shape: %s", 
                   input.symbolic_tree.size(), mapped_shape.to_string().c_str());
    
    return result;
}

float TensorMapper::estimate_processing_cost(const CognitiveData& input) const {
    float base_cost = 1.0f;
    
    if (input.has_symbolic()) {
        size_t tree_size = input.symbolic_tree.size();
        base_cost += static_cast<float>(tree_size) * 0.05f;
        
        // More complex strategies cost more
        switch (mapping_config_.strategy) {
            case MappingStrategy::STRUCTURAL:
                base_cost *= 1.0f;
                break;
            case MappingStrategy::SEMANTIC:
                base_cost *= 1.5f;
                break;
            case MappingStrategy::ATTENTION_WEIGHTED:
                base_cost *= 1.3f;
                break;
            case MappingStrategy::PRIME_FACTORIZED:
                base_cost *= 2.0f;
                break;
            case MappingStrategy::GGML_OPTIMIZED:
                base_cost *= 1.8f;
                break;
            case MappingStrategy::HYBRID:
                base_cost *= 2.5f;
                break;
        }
    }
    
    return base_cost;
}

float TensorMapper::estimate_output_value(const CognitiveData& input) const {
    float base_value = 1.0f;
    
    if (input.has_symbolic()) {
        size_t tree_size = input.symbolic_tree.size();
        
        // Value increases with information content
        base_value *= std::log(static_cast<float>(tree_size) + 1.0f);
        
        // Higher value if input has attention weights (more structured)
        if (!input.attention_weights.empty()) {
            base_value *= 1.2f;
        }
        
        // Bonus for GGML compatibility potential
        if (mapping_config_.optimize_for_ggml) {
            base_value *= 1.3f;
        }
    }
    
    return base_value;
}

TensorShape TensorMapper::map_symbolic_to_shape(const tree<std::string>& symbolic_data) const {
    return map_hybrid(symbolic_data, {});
}

TensorShape TensorMapper::map_structural(const tree<std::string>& symbolic_data) const {
    if (symbolic_data.empty()) {
        return TensorShape::create_scalar();
    }
    
    std::vector<size_t> dimensions = calculate_structural_dimensions(symbolic_data);
    
    // Ensure dimensions are reasonable
    for (auto& dim : dimensions) {
        dim = std::min(dim, mapping_config_.max_tensor_dimension);
    }
    
    if (dimensions.size() == 1) {
        return TensorShape::create_vector(dimensions[0]);
    } else if (dimensions.size() == 2) {
        return TensorShape::create_matrix(dimensions[0], dimensions[1]);
    } else {
        return TensorShape(dimensions, TensorShape::TensorType::HYPERGRAPH);
    }
}

TensorShape TensorMapper::map_semantic(const tree<std::string>& symbolic_data) const {
    if (symbolic_data.empty()) {
        return TensorShape::create_scalar();
    }
    
    // For semantic mapping, we create an embedding-like structure
    size_t vocab_size = symbolic_data.size();
    size_t embedding_dim = std::min(static_cast<size_t>(std::sqrt(vocab_size) * 2), mapping_config_.max_tensor_dimension);
    
    if (embedding_dim < 2) embedding_dim = 2;
    
    return TensorShape::create_matrix(vocab_size, embedding_dim);
}

TensorShape TensorMapper::map_attention_weighted(const tree<std::string>& symbolic_data, 
                                                const std::map<std::string, float>& attention_weights) const {
    TensorShape base_shape = map_structural(symbolic_data);
    return optimize_shape_for_attention(base_shape, attention_weights);
}

TensorShape TensorMapper::map_prime_factorized(const tree<std::string>& symbolic_data) const {
    TensorShape base_shape = map_structural(symbolic_data);
    return apply_prime_factorization(base_shape);
}

TensorShape TensorMapper::map_ggml_optimized(const tree<std::string>& symbolic_data) const {
    std::vector<size_t> raw_dimensions = calculate_structural_dimensions(symbolic_data);
    std::vector<size_t> optimized_dims = optimize_dimensions_for_ggml(raw_dimensions);
    
    // Create tensor shape with optimized dimensions
    if (optimized_dims.empty()) {
        return TensorShape::create_scalar();
    } else if (optimized_dims.size() == 1) {
        return TensorShape::create_vector(optimized_dims[0]);
    } else if (optimized_dims.size() == 2) {
        return TensorShape::create_matrix(optimized_dims[0], optimized_dims[1]);
    } else {
        return TensorShape(optimized_dims, TensorShape::TensorType::TENSOR_3D);
    }
}

TensorShape TensorMapper::map_hybrid(const tree<std::string>& symbolic_data, 
                                    const std::map<std::string, float>& attention_weights) const {
    if (symbolic_data.empty()) {
        return TensorShape::create_scalar();
    }
    
    // Start with structural mapping
    TensorShape structural_shape = map_structural(symbolic_data);
    
    // Apply attention weighting if available
    TensorShape attention_shape = structural_shape;
    if (!attention_weights.empty()) {
        attention_shape = optimize_shape_for_attention(structural_shape, attention_weights);
    }
    
    // Apply prime factorization if beneficial
    TensorShape factorized_shape = apply_prime_factorization(attention_shape);
    
    // Choose the most efficient shape
    if (mapping_config_.optimize_for_ggml && factorized_shape.is_ggml_compatible()) {
        return factorized_shape;
    } else if (attention_shape.get_total_elements() < structural_shape.get_total_elements() * 1.5f) {
        return attention_shape;
    } else {
        return structural_shape;
    }
}

std::vector<size_t> TensorMapper::calculate_structural_dimensions(const tree<std::string>& data) const {
    if (data.empty()) {
        return {1};
    }
    
    size_t node_count = data.size();
    size_t max_depth = 0;
    size_t max_branching = 0;
    
    // Calculate tree statistics
    for (auto it = data.begin(); it != data.end(); ++it) {
        size_t depth = data.depth(it);
        max_depth = std::max(max_depth, depth);
        
        size_t children = data.number_of_children(it);
        max_branching = std::max(max_branching, children);
    }
    
    // Create dimensions based on tree structure
    std::vector<size_t> dimensions;
    
    if (node_count == 1) {
        dimensions = {1};
    } else if (max_depth <= 1) {
        // Flat structure - use vector
        dimensions = {node_count};
    } else if (max_depth == 2) {
        // Simple hierarchy - use matrix
        dimensions = {max_depth + 1, std::max(max_branching, size_t(1))};
    } else {
        // Complex hierarchy - use 3D tensor
        dimensions = {max_depth + 1, max_branching + 1, node_count};
    }
    
    return dimensions;
}

std::vector<size_t> TensorMapper::optimize_dimensions_for_ggml(const std::vector<size_t>& raw_dimensions) const {
    std::vector<size_t> optimized = raw_dimensions;
    
    // GGML prefers powers of 2 and multiples of common sizes
    for (auto& dim : optimized) {
        // Round up to next power of 2 for small dimensions
        if (dim <= 64) {
            size_t power_of_2 = 1;
            while (power_of_2 < dim) {
                power_of_2 *= 2;
            }
            dim = power_of_2;
        } else {
            // Round to nearest multiple of 32 for larger dimensions
            dim = ((dim + 31) / 32) * 32;
        }
        
        // Enforce maximum dimension limit
        dim = std::min(dim, mapping_config_.max_tensor_dimension);
    }
    
    // Ensure we don't exceed 4D for GGML compatibility
    while (optimized.size() > 4) {
        // Merge smallest dimensions
        auto min_it = std::min_element(optimized.begin(), optimized.end());
        size_t min_index = std::distance(optimized.begin(), min_it);
        
        if (min_index > 0) {
            optimized[min_index - 1] *= optimized[min_index];
            optimized.erase(optimized.begin() + min_index);
        } else if (optimized.size() > 1) {
            optimized[1] *= optimized[0];
            optimized.erase(optimized.begin());
        } else {
            break;
        }
    }
    
    return optimized;
}

TensorShape TensorMapper::optimize_shape_for_attention(const TensorShape& base_shape, 
                                                      const std::map<std::string, float>& attention_weights) const {
    if (attention_weights.empty()) {
        return base_shape;
    }
    
    float attention_scaling = calculate_attention_scaling_factor(attention_weights);
    
    // Scale dimensions based on attention
    std::vector<size_t> new_dimensions = base_shape.get_dimensions();
    for (auto& dim : new_dimensions) {
        dim = static_cast<size_t>(dim * attention_scaling);
        dim = std::max(dim, size_t(1));
        dim = std::min(dim, mapping_config_.max_tensor_dimension);
    }
    
    return TensorShape(new_dimensions, base_shape.get_tensor_type());
}

TensorShape TensorMapper::apply_prime_factorization(const TensorShape& shape) const {
    auto factors = shape.get_prime_factorization();
    
    if (factors.size() <= 1) {
        return shape; // Can't improve factorization
    }
    
    // Try to create balanced dimensions from prime factors
    std::vector<size_t> new_dimensions;
    size_t current_dim = 1;
    size_t target_dimensions = std::min(factors.size(), size_t(4)); // Max 4D for GGML
    size_t factors_per_dim = factors.size() / target_dimensions;
    
    for (size_t i = 0; i < factors.size(); ++i) {
        current_dim *= factors[i];
        
        if ((i + 1) % factors_per_dim == 0 || i == factors.size() - 1) {
            new_dimensions.push_back(current_dim);
            current_dim = 1;
        }
    }
    
    return TensorShape(new_dimensions, shape.get_tensor_type());
}

float TensorMapper::calculate_attention_scaling_factor(const std::map<std::string, float>& attention_weights) const {
    if (attention_weights.empty()) {
        return 1.0f;
    }
    
    float total_weight = 0.0f;
    float count = 0.0f;
    
    for (const auto& pair : attention_weights) {
        if (pair.second >= mapping_config_.attention_threshold) {
            total_weight += pair.second;
            count += 1.0f;
        }
    }
    
    if (count == 0.0f) {
        return 1.0f;
    }
    
    float average_attention = total_weight / count;
    return std::max(0.1f, std::min(average_attention * 2.0f, 3.0f)); // Scale between 0.1 and 3.0
}

// =====================================================
// TensorMapperFactory Implementation
// =====================================================

std::shared_ptr<TensorMapper> TensorMapperFactory::create_structural_mapper(const std::string& kernel_id) {
    KernelConfig config(kernel_id, "tensor_mapper");
    TensorMapper::TensorMappingConfig mapping_config;
    mapping_config.strategy = TensorMapper::MappingStrategy::STRUCTURAL;
    mapping_config.preserve_structure = true;
    
    return std::make_shared<TensorMapper>(config, mapping_config);
}

std::shared_ptr<TensorMapper> TensorMapperFactory::create_ggml_optimized_mapper(const std::string& kernel_id) {
    KernelConfig config(kernel_id, "tensor_mapper");
    TensorMapper::TensorMappingConfig mapping_config;
    mapping_config.strategy = TensorMapper::MappingStrategy::GGML_OPTIMIZED;
    mapping_config.optimize_for_ggml = true;
    mapping_config.max_tensor_dimension = 1024;
    
    return std::make_shared<TensorMapper>(config, mapping_config);
}

std::shared_ptr<TensorMapper> TensorMapperFactory::create_hybrid_mapper(const std::string& kernel_id) {
    KernelConfig config(kernel_id, "tensor_mapper");
    TensorMapper::TensorMappingConfig mapping_config;
    mapping_config.strategy = TensorMapper::MappingStrategy::HYBRID;
    mapping_config.preserve_structure = true;
    mapping_config.optimize_for_ggml = true;
    mapping_config.enable_compression = true;
    
    return std::make_shared<TensorMapper>(config, mapping_config);
}

}} // namespace opencog::agentic