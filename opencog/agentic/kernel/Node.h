/*
 * opencog/agentic/kernel/Node.h
 *
 * Node - Basic cognitive entity with GGML tensor backend
 * Represents atomic cognitive objects in the hypergraph with full
 * neural-symbolic integration through GGML tensors.
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

#ifndef _OPENCOG_AGENTIC_NODE_H
#define _OPENCOG_AGENTIC_NODE_H

#include "AgenticKernel.h"
#include "TensorShape.h"
#include <string>
#include <vector>
#include <memory>
#include <map>

// Forward declarations for GGML integration
struct ggml_context;
struct ggml_tensor;

namespace opencog { namespace agentic {

/**
 * Node - Fundamental cognitive entity with GGML tensor representation.
 * 
 * Every Node in the cognitive hypergraph is backed by an actual GGML tensor
 * that encodes its semantic features, connections, and attention values.
 * This enables true neural-symbolic integration where symbolic operations
 * can be performed efficiently using tensor arithmetic.
 * 
 * Features:
 * - GGML tensor backend for all operations
 * - Semantic feature encoding in tensor space
 * - Attention-weighted tensor updates
 * - Bidirectional symbolic ↔ tensor synchronization
 * - Pattern matching through tensor similarity
 */
class Node {
public:
    enum class NodeType {
        CONCEPT,        // Conceptual node (abstract idea)
        PREDICATE,      // Predicate node (relation/function)
        WORD,          // Word/linguistic node
        NUMBER,        // Numeric constant
        VARIABLE,      // Variable node (placeholder)
        GROUNDED,      // Grounded node (external reference)
        SCHEMA,        // Procedural schema
        ATTENTION,     // Attention/focus node
        TEMPORAL,      // Time-based node
        SPATIAL,       // Spatial location node
        SYMBOLIC       // General symbolic node
    };
    
    struct NodeMetadata {
        float semantic_strength = 1.0f;    // Semantic association strength
        float attention_value = 0.0f;      // Current attention allocation
        float activation_level = 0.0f;     // Current activation
        float confidence = 1.0f;           // Confidence in node's validity
        size_t access_count = 0;           // Number of times accessed
        std::chrono::system_clock::time_point created;
        std::chrono::system_clock::time_point last_accessed;
        std::map<std::string, std::string> custom_attributes;
    };

public:
    Node(const std::string& node_id, NodeType type = NodeType::SYMBOLIC);
    Node(const std::string& node_id, const std::string& name, NodeType type = NodeType::SYMBOLIC);
    virtual ~Node();

    // Basic node operations
    const std::string& get_node_id() const { return node_id_; }
    const std::string& get_name() const { return name_; }
    NodeType get_node_type() const { return node_type_; }
    void set_name(const std::string& name);
    
    // GGML tensor backend operations
    bool create_tensor_representation(size_t feature_dimension = 128);
    bool update_tensor_from_features();
    bool sync_features_from_tensor();
    bool has_tensor_representation() const { return tensor_data_ != nullptr; }
    
    // Feature and semantic operations
    void set_semantic_feature(const std::string& feature, float value);
    float get_semantic_feature(const std::string& feature) const;
    void set_semantic_features(const std::map<std::string, float>& features);
    const std::map<std::string, float>& get_semantic_features() const { return semantic_features_; }
    
    // Attention and activation
    void set_attention_value(float attention);
    float get_attention_value() const { return metadata_.attention_value; }
    void update_activation(float activation);
    float get_activation_level() const { return metadata_.activation_level; }
    
    // Pattern matching and similarity
    float calculate_similarity(const Node& other) const;
    float calculate_tensor_similarity(const Node& other) const;
    bool matches_pattern(const Node& pattern) const;
    
    // Tensor operations
    ggml_tensor* get_tensor() const { return tensor_data_; }
    TensorShape get_tensor_shape() const;
    bool perform_tensor_operation(const std::string& operation, const std::vector<const Node*>& operands);
    
    // Metadata and statistics
    const NodeMetadata& get_metadata() const { return metadata_; }
    NodeMetadata& get_metadata() { return metadata_; }
    void update_access_statistics();
    
    // Serialization and persistence
    std::string to_string() const;
    std::map<std::string, std::string> to_metadata_map() const;
    static std::shared_ptr<Node> from_metadata_map(const std::map<std::string, std::string>& metadata);
    
    // AtomSpace compatibility
    std::string to_scheme_representation() const;
    static std::shared_ptr<Node> from_scheme_representation(const std::string& scheme_str);
    
    // Factory methods for common node types
    static std::shared_ptr<Node> create_concept_node(const std::string& name);
    static std::shared_ptr<Node> create_predicate_node(const std::string& name);
    static std::shared_ptr<Node> create_word_node(const std::string& word);
    static std::shared_ptr<Node> create_number_node(const std::string& name, float value);
    static std::shared_ptr<Node> create_variable_node(const std::string& name);
    
    // Operators
    bool operator==(const Node& other) const;
    bool operator!=(const Node& other) const { return !(*this == other); }
    bool operator<(const Node& other) const; // For sorting/ordering

protected:
    std::string node_id_;
    std::string name_;
    NodeType node_type_;
    NodeMetadata metadata_;
    
    // GGML tensor backend
    ggml_tensor* tensor_data_ = nullptr;
    ggml_context* tensor_context_ = nullptr;
    
    // Semantic features
    std::map<std::string, float> semantic_features_;
    
    // Helper methods
    std::string generate_node_id() const;
    void initialize_default_features();
    void cleanup_tensor_resources();
    
    // Tensor encoding/decoding
    void encode_features_to_tensor();
    void decode_features_from_tensor();
    std::vector<float> extract_feature_vector() const;
    void load_feature_vector(const std::vector<float>& features);
};

/**
 * NodeFactory - Factory for creating specialized nodes with optimal tensor layouts.
 */
class NodeFactory {
public:
    struct NodeConfig {
        size_t feature_dimension = 128;
        bool enable_tensor_backend = true;
        bool initialize_with_random_features = false;
        float default_attention_value = 0.0f;
        std::map<std::string, float> initial_features;
    };
    
    static std::shared_ptr<Node> create_node(const std::string& name, 
                                           Node::NodeType type,
                                           const NodeConfig& config);
    
    static std::shared_ptr<Node> create_node(const std::string& name, 
                                           Node::NodeType type);
    
    // Default configuration
    static NodeConfig default_config();
    
    // Specialized creation methods
    static std::shared_ptr<Node> create_semantic_node(const std::string& name,
                                                     const std::vector<std::string>& semantic_tags);
    static std::shared_ptr<Node> create_attention_node(const std::string& name, float initial_attention);
    static std::shared_ptr<Node> create_grounded_node(const std::string& name, const std::string& grounding);
    
    // Batch creation for efficiency
    static std::vector<std::shared_ptr<Node>> create_node_collection(
        const std::vector<std::string>& names,
        Node::NodeType type,
        const NodeConfig& config);
    
    static std::vector<std::shared_ptr<Node>> create_node_collection(
        const std::vector<std::string>& names,
        Node::NodeType type);
};

/**
 * NodeRegistry - Manages node instances and tensor resource allocation.
 */
class NodeRegistry {
public:
    static NodeRegistry& instance();
    
    // Node lifecycle management
    void register_node(std::shared_ptr<Node> node);
    void unregister_node(const std::string& node_id);
    std::shared_ptr<Node> get_node(const std::string& node_id);
    
    // Collection operations
    std::vector<std::shared_ptr<Node>> get_nodes_by_type(Node::NodeType type);
    std::vector<std::shared_ptr<Node>> find_similar_nodes(const Node& query_node, float similarity_threshold = 0.8f);
    std::vector<std::shared_ptr<Node>> find_nodes_by_feature(const std::string& feature_name, float min_value);
    
    // Tensor resource management
    size_t get_total_tensor_memory_usage() const;
    void optimize_tensor_layouts();
    void cleanup_unused_tensors();
    
    // Statistics and monitoring
    struct RegistryStats {
        size_t total_nodes = 0;
        size_t nodes_with_tensors = 0;
        size_t total_tensor_memory = 0;
        float average_feature_dimension = 0.0f;
        std::map<Node::NodeType, size_t> nodes_by_type;
    };
    
    RegistryStats get_stats() const;

private:
    NodeRegistry() = default;
    
    std::map<std::string, std::shared_ptr<Node>> nodes_;
    mutable std::mutex registry_mutex_;
};

}} // namespace opencog::agentic

#endif // _OPENCOG_AGENTIC_NODE_H