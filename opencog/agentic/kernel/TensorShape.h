/*
 * opencog/agentic/kernel/TensorShape.h
 *
 * TensorShape - Dimensionality descriptor for cognitive objects
 * Provides degrees of freedom analysis and tensor shape management
 * for the agentic kernel network.
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

#ifndef _OPENCOG_TENSOR_SHAPE_H
#define _OPENCOG_TENSOR_SHAPE_H

#include <vector>
#include <string>
#include <map>
#include <memory>
#include <functional>
#include <opencog/util/tree.h>

// Forward declarations for GGML integration
struct ggml_context;
struct ggml_tensor;

namespace opencog { namespace agentic {

// Forward declaration  
using ::opencog::tree;

namespace opencog { namespace agentic {

/**
 * TensorShape - Dimensionality descriptor for cognitive representations.
 * 
 * This class provides a unified way to describe the shape and degrees of freedom
 * of cognitive objects, enabling seamless conversion between symbolic and tensor
 * representations for GGML operations.
 * 
 * Features:
 * - Multi-dimensional shape specification
 * - Degrees of freedom analysis
 * - Prime factorization for optimal tensor layouts
 * - GGML tensor type mapping
 * - Attention-aware dimensionality reduction
 */
class TensorShape {
public:
    enum class TensorType {
        SCALAR,      // 0D - single value
        VECTOR,      // 1D - sequence of values
        MATRIX,      // 2D - traditional matrix
        TENSOR_3D,   // 3D - volumes, batches
        TENSOR_4D,   // 4D - feature maps, sequences
        HYPERGRAPH,  // Variable dimensionality for symbolic structures
        ATTENTION    // Special type for attention weight matrices
    };
    
    enum class DataType {
        FLOAT32,     // Standard floating point
        FLOAT16,     // Half precision
        INT32,       // Integer values
        INT8,        // Quantized integers
        BOOLEAN,     // Binary values
        SYMBOLIC     // Non-numeric symbolic data
    };

public:
    TensorShape();
    TensorShape(const std::vector<size_t>& dimensions, TensorType type = TensorType::TENSOR_3D);
    TensorShape(size_t dim1, TensorType type = TensorType::VECTOR);
    TensorShape(size_t dim1, size_t dim2, TensorType type = TensorType::MATRIX);
    TensorShape(size_t dim1, size_t dim2, size_t dim3, TensorType type = TensorType::TENSOR_3D);
    
    ~TensorShape() = default;
    
    // Basic shape operations
    size_t get_rank() const { return dimensions_.size(); }
    size_t get_dimension(size_t index) const;
    const std::vector<size_t>& get_dimensions() const { return dimensions_; }
    size_t get_total_elements() const;
    
    TensorType get_tensor_type() const { return tensor_type_; }
    DataType get_data_type() const { return data_type_; }
    void set_data_type(DataType type) { data_type_ = type; }
    
    // Degrees of freedom analysis
    size_t calculate_degrees_of_freedom() const;
    std::vector<size_t> get_prime_factorization() const;
    bool is_prime_factorizable() const;
    
    // Shape transformations
    TensorShape reshape(const std::vector<size_t>& new_dimensions) const;
    TensorShape transpose() const;
    TensorShape reduce_dimension(size_t axis) const;
    TensorShape expand_dimension(size_t axis, size_t new_size) const;
    
    // Compatibility and operations
    bool is_compatible_with(const TensorShape& other) const;
    bool can_broadcast_to(const TensorShape& target) const;
    TensorShape broadcast_to(const TensorShape& target) const;
    
    // GGML integration
    bool is_ggml_compatible() const;
    size_t estimate_ggml_memory_size() const;
    std::string to_ggml_type_string() const;
    
    // Attention-aware operations
    TensorShape apply_attention_mask(const std::vector<float>& attention_weights) const;
    TensorShape compress_for_attention(float compression_ratio) const;
    
    // Serialization and debugging
    std::string to_string() const;
    std::map<std::string, std::string> to_metadata() const;
    static TensorShape from_metadata(const std::map<std::string, std::string>& metadata);
    
    // Factory methods for common shapes
    static TensorShape create_scalar(DataType type = DataType::FLOAT32);
    static TensorShape create_vector(size_t size, DataType type = DataType::FLOAT32);
    static TensorShape create_matrix(size_t rows, size_t cols, DataType type = DataType::FLOAT32);
    static TensorShape create_attention_matrix(size_t seq_length, size_t feature_dim);
    static TensorShape create_hypergraph_shape(size_t nodes, size_t max_edges);
    static TensorShape infer_from_symbolic_tree(const tree<std::string>& symbolic_data);
    
    // Operators
    bool operator==(const TensorShape& other) const;
    bool operator!=(const TensorShape& other) const { return !(*this == other); }
    
private:
    std::vector<size_t> dimensions_;
    TensorType tensor_type_;
    DataType data_type_;
    
    // Metadata for attention and optimization
    std::map<std::string, float> optimization_hints_;
    
    // Helper methods
    void validate_dimensions() const;
    std::vector<size_t> prime_factors(size_t n) const;
    size_t calculate_memory_footprint() const;
};

/**
 * TensorShapeAnalyzer - Advanced analysis and optimization for tensor shapes.
 * 
 * Provides static methods for analyzing collections of tensor shapes,
 * optimizing layouts for memory efficiency, and suggesting transformations
 * for better GGML performance.
 */
class TensorShapeAnalyzer {
public:
    struct ShapeAnalysis {
        size_t total_memory_footprint;
        size_t total_degrees_of_freedom;
        float memory_efficiency_score;
        float computation_efficiency_score;
        std::vector<std::string> optimization_suggestions;
    };
    
    static ShapeAnalysis analyze_shape_collection(const std::vector<TensorShape>& shapes);
    static std::vector<TensorShape> optimize_for_ggml(const std::vector<TensorShape>& shapes);
    static TensorShape suggest_optimal_shape(size_t target_elements, TensorShape::TensorType preferred_type);
    static bool is_memory_efficient_layout(const TensorShape& shape);
    static float calculate_fragmentation_score(const std::vector<TensorShape>& shapes);
};

}} // namespace opencog::agentic

#endif // _OPENCOG_TENSOR_SHAPE_H