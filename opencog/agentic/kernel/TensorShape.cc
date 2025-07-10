/*
 * opencog/agentic/kernel/TensorShape.cc
 *
 * Implementation of TensorShape - Dimensionality descriptor for cognitive objects.
 */

#include "TensorShape.h"
#include "AgenticKernel.h"
#include <opencog/util/Logger.h>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <vector>

// GGML headers
#include "ggml_stub.h"
#define HAVE_GGML 1

namespace opencog { namespace agentic {

// =====================================================
// TensorShape Implementation
// =====================================================

TensorShape::TensorShape() 
    : tensor_type_(TensorType::SCALAR), data_type_(DataType::FLOAT32) {
    dimensions_ = {1};
}

TensorShape::TensorShape(const std::vector<size_t>& dimensions, TensorType type)
    : dimensions_(dimensions), tensor_type_(type), data_type_(DataType::FLOAT32) {
    validate_dimensions();
}

TensorShape::TensorShape(size_t dim1, TensorType type)
    : dimensions_({dim1}), tensor_type_(type), data_type_(DataType::FLOAT32) {
    validate_dimensions();
}

TensorShape::TensorShape(size_t dim1, size_t dim2, TensorType type)
    : dimensions_({dim1, dim2}), tensor_type_(type), data_type_(DataType::FLOAT32) {
    validate_dimensions();
}

TensorShape::TensorShape(size_t dim1, size_t dim2, size_t dim3, TensorType type)
    : dimensions_({dim1, dim2, dim3}), tensor_type_(type), data_type_(DataType::FLOAT32) {
    validate_dimensions();
}

size_t TensorShape::get_dimension(size_t index) const {
    if (index >= dimensions_.size()) {
        return 1; // Return 1 for dimensions beyond rank (broadcasting semantics)
    }
    return dimensions_[index];
}

size_t TensorShape::get_total_elements() const {
    if (dimensions_.empty()) return 0;
    return std::accumulate(dimensions_.begin(), dimensions_.end(), 1UL, std::multiplies<size_t>());
}

size_t TensorShape::calculate_degrees_of_freedom() const {
    // Degrees of freedom = total elements - constraints
    size_t total_elements = get_total_elements();
    
    // Basic constraints based on tensor type
    size_t constraints = 0;
    switch (tensor_type_) {
        case TensorType::SCALAR:
            constraints = 0; // No constraints
            break;
        case TensorType::VECTOR:
            constraints = 1; // Length constraint
            break;
        case TensorType::MATRIX:
            constraints = 2; // Row and column constraints
            break;
        case TensorType::ATTENTION:
            // Attention matrices have normalization constraints (sum to 1)
            constraints = dimensions_.empty() ? 0 : dimensions_[0];
            break;
        default:
            constraints = get_rank(); // One constraint per dimension
            break;
    }
    
    return total_elements > constraints ? total_elements - constraints : total_elements;
}

std::vector<size_t> TensorShape::get_prime_factorization() const {
    size_t total = get_total_elements();
    return prime_factors(total);
}

bool TensorShape::is_prime_factorizable() const {
    auto factors = get_prime_factorization();
    return factors.size() > 1 || (factors.size() == 1 && factors[0] > 1);
}

TensorShape TensorShape::reshape(const std::vector<size_t>& new_dimensions) const {
    size_t current_total = get_total_elements();
    size_t new_total = std::accumulate(new_dimensions.begin(), new_dimensions.end(), 1UL, std::multiplies<size_t>());
    
    if (current_total != new_total) {
        logger().warn("TensorShape reshape: element count mismatch (%zu vs %zu)", current_total, new_total);
        return *this; // Return unchanged if incompatible
    }
    
    TensorShape result(new_dimensions);
    result.data_type_ = data_type_;
    result.optimization_hints_ = optimization_hints_;
    
    // Infer tensor type from new dimensions
    switch (new_dimensions.size()) {
        case 0:
        case 1:
            if (new_dimensions.empty() || new_dimensions[0] == 1) {
                result.tensor_type_ = TensorType::SCALAR;
            } else {
                result.tensor_type_ = TensorType::VECTOR;
            }
            break;
        case 2:
            result.tensor_type_ = TensorType::MATRIX;
            break;
        case 3:
            result.tensor_type_ = TensorType::TENSOR_3D;
            break;
        case 4:
            result.tensor_type_ = TensorType::TENSOR_4D;
            break;
        default:
            result.tensor_type_ = TensorType::HYPERGRAPH;
            break;
    }
    
    return result;
}

TensorShape TensorShape::transpose() const {
    if (dimensions_.size() < 2) {
        return *this; // Can't transpose 0D or 1D tensors meaningfully
    }
    
    std::vector<size_t> transposed_dims = dimensions_;
    std::reverse(transposed_dims.begin(), transposed_dims.end());
    
    return TensorShape(transposed_dims, tensor_type_);
}

TensorShape TensorShape::reduce_dimension(size_t axis) const {
    if (axis >= dimensions_.size()) {
        return *this;
    }
    
    std::vector<size_t> reduced_dims;
    for (size_t i = 0; i < dimensions_.size(); ++i) {
        if (i != axis) {
            reduced_dims.push_back(dimensions_[i]);
        }
    }
    
    if (reduced_dims.empty()) {
        return TensorShape(); // Scalar result
    }
    
    return TensorShape(reduced_dims);
}

bool TensorShape::is_compatible_with(const TensorShape& other) const {
    // Shapes are compatible if they can be broadcast together
    size_t max_rank = std::max(get_rank(), other.get_rank());
    
    for (size_t i = 0; i < max_rank; ++i) {
        size_t dim1 = get_dimension(get_rank() - 1 - i);
        size_t dim2 = other.get_dimension(other.get_rank() - 1 - i);
        
        if (dim1 != 1 && dim2 != 1 && dim1 != dim2) {
            return false;
        }
    }
    
    return true;
}

bool TensorShape::can_broadcast_to(const TensorShape& target) const {
    if (get_rank() > target.get_rank()) {
        return false; // Can't broadcast to lower rank
    }
    
    size_t rank_diff = target.get_rank() - get_rank();
    
    for (size_t i = 0; i < get_rank(); ++i) {
        size_t my_dim = dimensions_[i];
        size_t target_dim = target.dimensions_[rank_diff + i];
        
        if (my_dim != 1 && my_dim != target_dim) {
            return false;
        }
    }
    
    return true;
}

bool TensorShape::is_ggml_compatible() const {
    // GGML has some limitations on tensor dimensions and types
    if (get_rank() > 4) {
        return false; // GGML typically supports up to 4D tensors
    }
    
    if (get_total_elements() == 0) {
        return false; // Empty tensors not supported
    }
    
    // Check for reasonable dimension sizes
    for (size_t dim : dimensions_) {
        if (dim > 1000000) { // 1M elements per dimension is a reasonable limit
            return false;
        }
    }
    
    // Check if data type is supported by GGML
    switch (data_type_) {
        case DataType::FLOAT32:
        case DataType::FLOAT16:
        case DataType::INT32:
        case DataType::INT8:
            return true;
        case DataType::BOOLEAN:
        case DataType::SYMBOLIC:
            return false; // Not directly supported, needs conversion
    }
    
    return true;
}

size_t TensorShape::estimate_ggml_memory_size() const {
    size_t element_size = 0;
    
    switch (data_type_) {
        case DataType::FLOAT32:
            element_size = 4;
            break;
        case DataType::FLOAT16:
            element_size = 2;
            break;
        case DataType::INT32:
            element_size = 4;
            break;
        case DataType::INT8:
            element_size = 1;
            break;
        case DataType::BOOLEAN:
            element_size = 1;
            break;
        case DataType::SYMBOLIC:
            element_size = 8; // Assume pointer size for symbolic data
            break;
    }
    
    return get_total_elements() * element_size;
}

std::string TensorShape::to_ggml_type_string() const {
    switch (data_type_) {
        case DataType::FLOAT32: return "GGML_TYPE_F32";
        case DataType::FLOAT16: return "GGML_TYPE_F16";
        case DataType::INT32: return "GGML_TYPE_I32";
        case DataType::INT8: return "GGML_TYPE_I8";
        default: return "GGML_TYPE_F32"; // Default fallback
    }
}

ggml_tensor* TensorShape::create_ggml_tensor(ggml_context* ctx) const {
    if (!ctx || !is_ggml_compatible()) {
        return nullptr;
    }
    
    // Convert DataType to ggml_type
    ggml_type ggml_data_type;
    switch (data_type_) {
        case DataType::FLOAT32: ggml_data_type = GGML_TYPE_F32; break;
        case DataType::FLOAT16: ggml_data_type = GGML_TYPE_F16; break;
        case DataType::INT32: ggml_data_type = GGML_TYPE_I32; break;
        case DataType::INT8: ggml_data_type = GGML_TYPE_I8; break;
        default: ggml_data_type = GGML_TYPE_F32; break;
    }
    
    // Create tensor based on rank
    ggml_tensor* tensor = nullptr;
    switch (get_rank()) {
        case 0:
        case 1:
            tensor = ggml_new_tensor_1d(ctx, ggml_data_type, dimensions_.empty() ? 1 : dimensions_[0]);
            break;
        case 2:
            tensor = ggml_new_tensor_2d(ctx, ggml_data_type, dimensions_[0], dimensions_[1]);
            break;
        case 3:
            tensor = ggml_new_tensor_3d(ctx, ggml_data_type, dimensions_[0], dimensions_[1], dimensions_[2]);
            break;
        case 4:
            tensor = ggml_new_tensor_4d(ctx, ggml_data_type, dimensions_[0], dimensions_[1], dimensions_[2], dimensions_[3]);
            break;
        default:
            // For higher dimensions, fall back to 1D with total elements
            tensor = ggml_new_tensor_1d(ctx, ggml_data_type, get_total_elements());
            break;
    }
    
    return tensor;
}

TensorShape TensorShape::from_ggml_tensor(const ggml_tensor* tensor) {
    if (!tensor) {
        return TensorShape();
    }
    
    // Extract dimensions from GGML tensor
    std::vector<size_t> dims;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] > 1) {
            dims.push_back(tensor->ne[i]);
        }
    }
    
    if (dims.empty()) {
        dims.push_back(1); // Scalar case
    }
    
    TensorShape shape(dims);
    
    // Convert GGML type to DataType
    switch (tensor->type) {
        case GGML_TYPE_F32: shape.data_type_ = DataType::FLOAT32; break;
        case GGML_TYPE_F16: shape.data_type_ = DataType::FLOAT16; break;
        case GGML_TYPE_I32: shape.data_type_ = DataType::INT32; break;
        case GGML_TYPE_I8: shape.data_type_ = DataType::INT8; break;
        default: shape.data_type_ = DataType::FLOAT32; break;
    }
    
    // Infer tensor type from dimensions
    switch (dims.size()) {
        case 1:
            shape.tensor_type_ = (dims[0] == 1) ? TensorType::SCALAR : TensorType::VECTOR;
            break;
        case 2:
            shape.tensor_type_ = TensorType::MATRIX;
            break;
        case 3:
            shape.tensor_type_ = TensorType::TENSOR_3D;
            break;
        case 4:
            shape.tensor_type_ = TensorType::TENSOR_4D;
            break;
        default:
            shape.tensor_type_ = TensorType::HYPERGRAPH;
            break;
    }
    
    return shape;
}

std::string TensorShape::to_string() const {
    std::ostringstream oss;
    oss << "TensorShape(";
    
    for (size_t i = 0; i < dimensions_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << dimensions_[i];
    }
    
    oss << ") type=" << static_cast<int>(tensor_type_);
    oss << " dtype=" << static_cast<int>(data_type_);
    oss << " elements=" << get_total_elements();
    oss << " dof=" << calculate_degrees_of_freedom();
    
    return oss.str();
}

std::map<std::string, std::string> TensorShape::to_metadata() const {
    std::map<std::string, std::string> metadata;
    
    // Serialize dimensions
    std::ostringstream dims_oss;
    for (size_t i = 0; i < dimensions_.size(); ++i) {
        if (i > 0) dims_oss << ",";
        dims_oss << dimensions_[i];
    }
    metadata["dimensions"] = dims_oss.str();
    
    metadata["tensor_type"] = std::to_string(static_cast<int>(tensor_type_));
    metadata["data_type"] = std::to_string(static_cast<int>(data_type_));
    metadata["total_elements"] = std::to_string(get_total_elements());
    metadata["degrees_of_freedom"] = std::to_string(calculate_degrees_of_freedom());
    metadata["ggml_compatible"] = is_ggml_compatible() ? "true" : "false";
    
    return metadata;
}

TensorShape TensorShape::from_metadata(const std::map<std::string, std::string>& metadata) {
    TensorShape result;
    
    // Parse dimensions
    auto dims_it = metadata.find("dimensions");
    if (dims_it != metadata.end()) {
        std::vector<size_t> dims;
        std::istringstream iss(dims_it->second);
        std::string dim_str;
        
        while (std::getline(iss, dim_str, ',')) {
            dims.push_back(std::stoull(dim_str));
        }
        
        result.dimensions_ = dims;
    }
    
    // Parse tensor type
    auto type_it = metadata.find("tensor_type");
    if (type_it != metadata.end()) {
        result.tensor_type_ = static_cast<TensorType>(std::stoi(type_it->second));
    }
    
    // Parse data type
    auto dtype_it = metadata.find("data_type");
    if (dtype_it != metadata.end()) {
        result.data_type_ = static_cast<DataType>(std::stoi(dtype_it->second));
    }
    
    return result;
}

// Static factory methods
TensorShape TensorShape::create_scalar(DataType type) {
    TensorShape shape;
    shape.tensor_type_ = TensorType::SCALAR;
    shape.data_type_ = type;
    shape.dimensions_ = {1};
    return shape;
}

TensorShape TensorShape::create_vector(size_t size, DataType type) {
    TensorShape shape(size, TensorType::VECTOR);
    shape.data_type_ = type;
    return shape;
}

TensorShape TensorShape::create_matrix(size_t rows, size_t cols, DataType type) {
    TensorShape shape(rows, cols, TensorType::MATRIX);
    shape.data_type_ = type;
    return shape;
}

TensorShape TensorShape::create_attention_matrix(size_t seq_length, size_t feature_dim) {
    TensorShape shape(seq_length, feature_dim, TensorType::ATTENTION);
    shape.data_type_ = DataType::FLOAT32;
    return shape;
}

TensorShape TensorShape::create_hypergraph_shape(size_t nodes, size_t max_edges) {
    // Create a shape suitable for representing hypergraph structures
    std::vector<size_t> dims = {nodes, max_edges, nodes}; // Adjacency tensor representation
    TensorShape shape(dims, TensorType::HYPERGRAPH);
    shape.data_type_ = DataType::FLOAT32;
    return shape;
}

TensorShape TensorShape::infer_from_symbolic_tree(const tree<std::string>& symbolic_data) {
    if (symbolic_data.empty()) {
        return TensorShape::create_scalar();
    }
    
    size_t node_count = symbolic_data.size();
    size_t max_depth = 0;
    size_t max_width = 0;
    
    // Calculate tree statistics
    for (auto it = symbolic_data.begin(); it != symbolic_data.end(); ++it) {
        size_t depth = symbolic_data.depth(it);
        max_depth = std::max(max_depth, depth);
        
        size_t siblings = symbolic_data.number_of_siblings(it);
        max_width = std::max(max_width, siblings + 1);
    }
    
    // Create a tensor shape that can represent the tree structure
    if (node_count == 1) {
        return TensorShape::create_scalar(DataType::SYMBOLIC);
    } else if (max_depth <= 1) {
        return TensorShape::create_vector(node_count, DataType::SYMBOLIC);
    } else {
        // Multi-dimensional representation for complex trees
        std::vector<size_t> dims = {max_depth + 1, max_width, node_count};
        TensorShape shape(dims, TensorType::HYPERGRAPH);
        shape.data_type_ = DataType::SYMBOLIC;
        return shape;
    }
}

bool TensorShape::operator==(const TensorShape& other) const {
    return dimensions_ == other.dimensions_ && 
           tensor_type_ == other.tensor_type_ && 
           data_type_ == other.data_type_;
}

// Private helper methods
void TensorShape::validate_dimensions() const {
    for (size_t dim : dimensions_) {
        if (dim == 0) {
            logger().warn("TensorShape: zero dimension detected, this may cause issues");
        }
    }
}

std::vector<size_t> TensorShape::prime_factors(size_t n) const {
    std::vector<size_t> factors;
    
    // Check for factor 2
    while (n % 2 == 0) {
        factors.push_back(2);
        n = n / 2;
    }
    
    // Check for odd factors
    for (size_t i = 3; i * i <= n; i += 2) {
        while (n % i == 0) {
            factors.push_back(i);
            n = n / i;
        }
    }
    
    // If n is still greater than 2, it's a prime
    if (n > 2) {
        factors.push_back(n);
    }
    
    return factors;
}

// =====================================================
// TensorShapeAnalyzer Implementation
// =====================================================

TensorShapeAnalyzer::ShapeAnalysis TensorShapeAnalyzer::analyze_shape_collection(const std::vector<TensorShape>& shapes) {
    ShapeAnalysis analysis;
    analysis.total_memory_footprint = 0;
    analysis.total_degrees_of_freedom = 0;
    analysis.memory_efficiency_score = 0.0f;
    analysis.computation_efficiency_score = 0.0f;
    
    if (shapes.empty()) {
        return analysis;
    }
    
    size_t total_elements = 0;
    size_t ggml_compatible_count = 0;
    
    for (const auto& shape : shapes) {
        analysis.total_memory_footprint += shape.estimate_ggml_memory_size();
        analysis.total_degrees_of_freedom += shape.calculate_degrees_of_freedom();
        total_elements += shape.get_total_elements();
        
        if (shape.is_ggml_compatible()) {
            ggml_compatible_count++;
        }
    }
    
    // Calculate efficiency scores
    analysis.memory_efficiency_score = static_cast<float>(ggml_compatible_count) / shapes.size();
    analysis.computation_efficiency_score = total_elements > 0 ? 
        static_cast<float>(analysis.total_degrees_of_freedom) / total_elements : 0.0f;
    
    // Generate optimization suggestions
    if (analysis.memory_efficiency_score < 0.8f) {
        analysis.optimization_suggestions.push_back("Consider reshaping tensors for GGML compatibility");
    }
    if (analysis.computation_efficiency_score < 0.5f) {
        analysis.optimization_suggestions.push_back("Tensors may be over-constrained, consider reducing dimensionality");
    }
    if (calculate_fragmentation_score(shapes) > 0.5f) {
        analysis.optimization_suggestions.push_back("Memory layout is fragmented, consider consolidation");
    }
    
    return analysis;
}

float TensorShapeAnalyzer::calculate_fragmentation_score(const std::vector<TensorShape>& shapes) {
    if (shapes.size() < 2) return 0.0f;
    
    // Simple fragmentation metric based on size variance
    std::vector<size_t> sizes;
    for (const auto& shape : shapes) {
        sizes.push_back(shape.get_total_elements());
    }
    
    float mean = static_cast<float>(std::accumulate(sizes.begin(), sizes.end(), 0UL)) / sizes.size();
    float variance = 0.0f;
    
    for (size_t size : sizes) {
        float diff = static_cast<float>(size) - mean;
        variance += diff * diff;
    }
    variance /= sizes.size();
    
    float cv = mean > 0 ? std::sqrt(variance) / mean : 0.0f; // Coefficient of variation
    return std::min(cv, 1.0f); // Normalize to [0, 1]
}

std::vector<TensorShape> TensorShapeAnalyzer::optimize_for_ggml(const std::vector<TensorShape>& shapes) {
    std::vector<TensorShape> optimized_shapes;
    
    for (const auto& shape : shapes) {
        if (shape.is_ggml_compatible()) {
            optimized_shapes.push_back(shape);
        } else {
            // Optimize shape for GGML compatibility
            std::vector<size_t> dims = shape.get_dimensions();
            
            // Ensure dimensions are not too large
            for (size_t& dim : dims) {
                if (dim > 1000000) {
                    dim = 1000000;
                }
            }
            
            // Reduce rank if too high
            while (dims.size() > 4) {
                // Combine last two dimensions
                if (dims.size() >= 2) {
                    size_t last = dims.back();
                    dims.pop_back();
                    dims.back() *= last;
                } else {
                    break;
                }
            }
            
            TensorShape optimized(dims, shape.get_tensor_type());
            optimized.set_data_type(shape.get_data_type());
            optimized_shapes.push_back(optimized);
        }
    }
    
    return optimized_shapes;
}

TensorShape TensorShapeAnalyzer::suggest_optimal_shape(size_t target_elements, TensorShape::TensorType preferred_type) {
    if (target_elements == 0) {
        return TensorShape::create_scalar();
    }
    
    if (target_elements == 1) {
        return TensorShape::create_scalar();
    }
    
    // Find good dimensions that factor nicely
    std::vector<size_t> dims;
    
    switch (preferred_type) {
        case TensorShape::TensorType::VECTOR:
            dims = {target_elements};
            break;
            
        case TensorShape::TensorType::MATRIX: {
            // Find square or near-square dimensions
            size_t sqrt_approx = static_cast<size_t>(std::sqrt(target_elements));
            size_t dim1 = sqrt_approx;
            size_t dim2 = target_elements / dim1;
            
            if (dim1 * dim2 != target_elements) {
                dim2 = target_elements / dim1;
                if (dim1 * dim2 < target_elements) {
                    dim2++;
                }
            }
            
            dims = {dim1, dim2};
            break;
        }
        
        case TensorShape::TensorType::TENSOR_3D: {
            // Try to create balanced 3D dimensions
            size_t cube_root = static_cast<size_t>(std::cbrt(target_elements));
            dims = {cube_root, cube_root, target_elements / (cube_root * cube_root)};
            break;
        }
        
        default:
            // Default to vector
            dims = {target_elements};
            break;
    }
    
    TensorShape shape(dims, preferred_type);
    return shape;
}

bool TensorShapeAnalyzer::is_memory_efficient_layout(const TensorShape& shape) {
    // Check if shape is GGML compatible
    if (!shape.is_ggml_compatible()) {
        return false;
    }
    
    // Check if dimensions are powers of 2 or multiples of common vector sizes
    const auto& dims = shape.get_dimensions();
    for (size_t dim : dims) {
        // Check if dimension is power of 2, multiple of 32, or other efficient size
        bool efficient = (dim & (dim - 1)) == 0 ||  // Power of 2
                        (dim % 32 == 0) ||            // Multiple of 32
                        (dim % 16 == 0) ||            // Multiple of 16
                        (dim <= 8);                   // Small dimensions are generally OK
        
        if (!efficient) {
            return false;
        }
    }
    
    return true;
}

}} // namespace opencog::agentic