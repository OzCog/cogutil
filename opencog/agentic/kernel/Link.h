/*
 * opencog/agentic/kernel/Link.h
 *
 * Link - Relation/hyperedge primitive for cognitive hypergraph patterns
 * Enables representation of complex relationships between cognitive objects
 * in the agentic kernel network.
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

#ifndef _OPENCOG_AGENTIC_LINK_H
#define _OPENCOG_AGENTIC_LINK_H

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <set>
#include <chrono>

// Forward declarations for GGML integration
struct ggml_context;
struct ggml_tensor;

namespace opencog { namespace agentic {

/**
 * Link - Represents relations and hyperedges in cognitive structures.
 * 
 * Links are the fundamental building blocks for representing relationships
 * between cognitive objects (Nodes) in hypergraph patterns. They enable
 * the encoding of complex semantic and structural relationships that can
 * be processed by agentic kernels.
 * 
 * Features:
 * - N-ary relations (hyperedges connecting multiple nodes)
 * - Typed relationships with semantic annotations
 * - Attention weights for relevance ranking
 * - Pattern matching and traversal capabilities
 * - GGML tensor backend for neural-symbolic integration
 * - Tensor-encoded relationship strengths and attention weights
 */
class Link {
public:
    enum class LinkType {
        SIMPLE,          // Binary relation between two nodes
        INHERITANCE,     // IS-A relationship (A inherits from B)
        SIMILARITY,      // Similarity relationship with strength
        EVALUATION,      // Predicate evaluation (function application)
        IMPLICATION,     // Logical implication (if-then)
        SEQUENCE,        // Temporal or ordered sequence
        HYPEREDGE,       // General N-ary relationship
        ATTENTION,       // Attention relationship for focus allocation
        TEMPORAL,        // Time-based relationship
        CAUSAL,          // Cause-effect relationship
        SYMBOLIC         // Abstract symbolic relationship
    };
    
    struct LinkMetadata {
        float strength = 1.0f;        // Relationship strength/confidence
        float attention_weight = 0.5f; // Attention allocation weight
        size_t temporal_order = 0;     // Temporal ordering information
        std::map<std::string, std::string> annotations; // Additional metadata
        bool is_bidirectional = false; // Whether link works in both directions
        
        // Tensor-related metadata
        float tensor_similarity_cache = -1.0f; // Cached tensor similarity result
        std::chrono::system_clock::time_point created;
        std::chrono::system_clock::time_point last_accessed;
        size_t tensor_update_count = 0; // Number of tensor updates
    };

public:
    Link(LinkType type, const std::string& link_id = "");
    Link(LinkType type, const std::vector<std::string>& node_ids, const std::string& link_id = "");
    virtual ~Link() = default;
    
    // Basic link operations
    const std::string& get_link_id() const { return link_id_; }
    LinkType get_link_type() const { return link_type_; }
    const std::vector<std::string>& get_connected_nodes() const { return connected_nodes_; }
    
    void add_node(const std::string& node_id);
    void remove_node(const std::string& node_id);
    bool connects_node(const std::string& node_id) const;
    bool connects_nodes(const std::vector<std::string>& node_ids) const;
    
    // Metadata and properties
    const LinkMetadata& get_metadata() const { return metadata_; }
    LinkMetadata& get_metadata() { return metadata_; }
    void set_strength(float strength) { metadata_.strength = strength; }
    void set_attention_weight(float weight) { metadata_.attention_weight = weight; }
    void set_annotation(const std::string& key, const std::string& value);
    std::string get_annotation(const std::string& key) const;
    
    // GGML tensor backend operations
    bool create_tensor_representation(size_t feature_dimension = 64);
    bool update_tensor_from_metadata();
    bool sync_metadata_from_tensor();
    bool has_tensor_representation() const { return tensor_data_ != nullptr; }
    
    // Tensor operations for neural-symbolic integration
    ggml_tensor* get_tensor() const { return tensor_data_; }
    float calculate_tensor_strength(const std::vector<ggml_tensor*>& node_tensors) const;
    bool perform_tensor_attention_update(float attention_delta);
    void encode_relationship_to_tensor();
    
    // Hypergraph operations
    size_t get_arity() const { return connected_nodes_.size(); }
    bool is_binary() const { return get_arity() == 2; }
    bool is_unary() const { return get_arity() == 1; }
    bool is_hyperedge() const { return get_arity() > 2; }
    
    // Pattern matching
    bool matches_pattern(const Link& pattern) const;
    bool matches_type_pattern(LinkType pattern_type) const;
    float calculate_pattern_similarity(const Link& other) const;
    
    // Attention and relevance
    float calculate_relevance_score(const std::set<std::string>& focus_nodes) const;
    void update_attention_based_on_usage(float usage_factor);
    bool should_activate_for_attention(float attention_threshold) const;
    
    // Serialization and persistence
    std::string to_string() const;
    std::map<std::string, std::string> to_metadata_map() const;
    static std::shared_ptr<Link> from_metadata_map(const std::map<std::string, std::string>& metadata);
    
    // Factory methods for common link types
    static std::shared_ptr<Link> create_inheritance_link(const std::string& child, const std::string& parent);
    static std::shared_ptr<Link> create_similarity_link(const std::string& node1, const std::string& node2, float strength);
    static std::shared_ptr<Link> create_evaluation_link(const std::string& predicate, const std::vector<std::string>& arguments);
    static std::shared_ptr<Link> create_sequence_link(const std::vector<std::string>& ordered_nodes);
    static std::shared_ptr<Link> create_attention_link(const std::string& source, const std::string& target, float weight);
    
    // Operators
    bool operator==(const Link& other) const;
    bool operator!=(const Link& other) const { return !(*this == other); }
    bool operator<(const Link& other) const; // For sorting/ordering
    
protected:
    std::string link_id_;
    LinkType link_type_;
    std::vector<std::string> connected_nodes_;
    LinkMetadata metadata_;
    
    // GGML tensor backend
    ggml_tensor* tensor_data_ = nullptr;
    ggml_context* tensor_context_ = nullptr;
    
    // Relational features for tensor encoding
    std::map<std::string, float> relational_features_;
    
    // Helper methods
    std::string generate_link_id() const;
    void validate_link_structure() const;
    void initialize_default_features();
    void cleanup_tensor_resources();
    
    // Tensor encoding/decoding
    void encode_features_to_tensor();
    void decode_features_from_tensor();
    std::vector<float> extract_relational_vector() const;
    void load_relational_vector(const std::vector<float>& features);
};

}} // namespace opencog::agentic

#endif // _OPENCOG_AGENTIC_LINK_H