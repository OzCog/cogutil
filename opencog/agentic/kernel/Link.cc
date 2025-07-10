/*
 * opencog/agentic/kernel/Link.cc
 *
 * Implementation of Link - Relation/hyperedge primitive with GGML tensor backend
 */

#include "Link.h"
#include "ggml_stub.h"
#include <opencog/util/Logger.h>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>

namespace opencog { namespace agentic {

// =====================================================
// Link Implementation
// =====================================================

Link::Link(LinkType type, const std::string& link_id) 
    : link_type_(type), link_id_(link_id.empty() ? generate_link_id() : link_id) {
    
    metadata_.created = std::chrono::system_clock::now();
    metadata_.last_accessed = metadata_.created;
    
    initialize_default_features();
    
    logger().debug("Created link: %s (type: %d)", link_id_.c_str(), static_cast<int>(link_type_));
}

Link::Link(LinkType type, const std::vector<std::string>& node_ids, const std::string& link_id) 
    : link_type_(type), connected_nodes_(node_ids), 
      link_id_(link_id.empty() ? generate_link_id() : link_id) {
    
    metadata_.created = std::chrono::system_clock::now();
    metadata_.last_accessed = metadata_.created;
    
    initialize_default_features();
    validate_link_structure();
    
    logger().debug("Created link: %s with %zu nodes (type: %d)", 
                  link_id_.c_str(), connected_nodes_.size(), static_cast<int>(link_type_));
}

void Link::add_node(const std::string& node_id) {
    if (std::find(connected_nodes_.begin(), connected_nodes_.end(), node_id) == connected_nodes_.end()) {
        connected_nodes_.push_back(node_id);
        
        // Update tensor representation if it exists
        if (has_tensor_representation()) {
            update_tensor_from_metadata();
        }
    }
}

void Link::remove_node(const std::string& node_id) {
    auto it = std::find(connected_nodes_.begin(), connected_nodes_.end(), node_id);
    if (it != connected_nodes_.end()) {
        connected_nodes_.erase(it);
        
        // Update tensor representation if it exists
        if (has_tensor_representation()) {
            update_tensor_from_metadata();
        }
    }
}

bool Link::connects_node(const std::string& node_id) const {
    return std::find(connected_nodes_.begin(), connected_nodes_.end(), node_id) != connected_nodes_.end();
}

bool Link::connects_nodes(const std::vector<std::string>& node_ids) const {
    for (const auto& node_id : node_ids) {
        if (!connects_node(node_id)) {
            return false;
        }
    }
    return true;
}

void Link::set_annotation(const std::string& key, const std::string& value) {
    metadata_.annotations[key] = value;
    
    // Update relational features based on annotation
    if (key == "semantic_weight") {
        try {
            float weight = std::stof(value);
            relational_features_["annotation_semantic_weight"] = weight;
        } catch (const std::exception&) {
            relational_features_["annotation_semantic_weight"] = 0.0f;
        }
    }
    
    // Update tensor representation if it exists
    if (has_tensor_representation()) {
        update_tensor_from_metadata();
    }
}

std::string Link::get_annotation(const std::string& key) const {
    auto it = metadata_.annotations.find(key);
    return it != metadata_.annotations.end() ? it->second : "";
}

// =====================================================
// GGML Tensor Backend Operations
// =====================================================

bool Link::create_tensor_representation(size_t feature_dimension) {
    // Clean up existing tensor if present
    cleanup_tensor_resources();
    
    // Initialize GGML context
    struct ggml_init_params params = {
        .mem_size   = 32 * 1024 * 1024, // 32MB for link tensors
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    
    tensor_context_ = ggml_init(params);
    if (!tensor_context_) {
        logger().error("Failed to initialize GGML context for link: %s", link_id_.c_str());
        return false;
    }
    
    // Create relational tensor (includes arity, type, and relationship features)
    tensor_data_ = ggml_new_tensor_1d(tensor_context_, GGML_TYPE_F32, feature_dimension);
    if (!tensor_data_) {
        logger().error("Failed to create tensor for link: %s", link_id_.c_str());
        cleanup_tensor_resources();
        return false;
    }
    
    // Initialize tensor with current metadata
    return update_tensor_from_metadata();
}

bool Link::update_tensor_from_metadata() {
    if (!tensor_data_ || !tensor_context_) {
        return false;
    }
    
    float* data = (float*)tensor_data_->data;
    size_t tensor_size = ggml_nelements(tensor_data_);
    
    // Clear tensor first
    std::fill(data, data + tensor_size, 0.0f);
    
    // Encode relational features into tensor
    encode_features_to_tensor();
    
    metadata_.tensor_update_count++;
    
    return true;
}

bool Link::sync_metadata_from_tensor() {
    if (!tensor_data_ || !tensor_context_) {
        return false;
    }
    
    // Decode features from tensor
    decode_features_from_tensor();
    
    return true;
}

float Link::calculate_tensor_strength(const std::vector<ggml_tensor*>& node_tensors) const {
    if (!tensor_data_ || node_tensors.empty()) {
        return 0.0f;
    }
    
    // Calculate relationship strength based on tensor compatibility with connected nodes
    float total_compatibility = 0.0f;
    
    for (const auto& node_tensor : node_tensors) {
        if (node_tensor && node_tensor->data) {
            // Simple dot product similarity between link tensor and node tensor
            float* link_data = (float*)tensor_data_->data;
            float* node_data = (float*)node_tensor->data;
            
            size_t min_size = std::min(ggml_nelements(tensor_data_), ggml_nelements(node_tensor));
            
            float dot_product = 0.0f;
            float link_norm = 0.0f;
            float node_norm = 0.0f;
            
            for (size_t i = 0; i < min_size; ++i) {
                dot_product += link_data[i] * node_data[i];
                link_norm += link_data[i] * link_data[i];
                node_norm += node_data[i] * node_data[i];
            }
            
            // Cosine similarity
            if (link_norm > 0.0f && node_norm > 0.0f) {
                float similarity = dot_product / (std::sqrt(link_norm) * std::sqrt(node_norm));
                total_compatibility += similarity;
            }
        }
    }
    
    return node_tensors.empty() ? 0.0f : total_compatibility / node_tensors.size();
}

bool Link::perform_tensor_attention_update(float attention_delta) {
    if (!tensor_data_) {
        return false;
    }
    
    // Update attention weight
    metadata_.attention_weight = std::max(0.0f, std::min(1.0f, metadata_.attention_weight + attention_delta));
    
    // Update tensor to reflect new attention value
    return update_tensor_from_metadata();
}

void Link::encode_relationship_to_tensor() {
    if (!tensor_data_) return;
    
    float* data = (float*)tensor_data_->data;
    size_t tensor_size = ggml_nelements(tensor_data_);
    
    // Encode relationship features into specific tensor positions
    std::vector<float> relational_vector = extract_relational_vector();
    
    // Copy features to tensor (with padding/truncation as needed)
    size_t copy_size = std::min(tensor_size, relational_vector.size());
    std::copy(relational_vector.begin(), relational_vector.begin() + copy_size, data);
    
    // Pad remaining with zeros
    if (copy_size < tensor_size) {
        std::fill(data + copy_size, data + tensor_size, 0.0f);
    }
}

// =====================================================
// Pattern Matching and Similarity
// =====================================================

bool Link::matches_pattern(const Link& pattern) const {
    // Type must match
    if (link_type_ != pattern.link_type_) {
        return false;
    }
    
    // Arity must be compatible
    if (connected_nodes_.size() != pattern.connected_nodes_.size()) {
        return false;
    }
    
    // If both have tensors, use tensor-based matching
    if (has_tensor_representation() && pattern.has_tensor_representation()) {
        float similarity = calculate_pattern_similarity(pattern);
        return similarity > 0.7f; // Threshold for pattern matching
    }
    
    // Fall back to structural matching
    return connects_nodes(pattern.connected_nodes_);
}

bool Link::matches_type_pattern(LinkType pattern_type) const {
    return link_type_ == pattern_type;
}

float Link::calculate_pattern_similarity(const Link& other) const {
    if (!has_tensor_representation() || !other.has_tensor_representation()) {
        // Structural similarity
        float type_similarity = (link_type_ == other.link_type_) ? 1.0f : 0.0f;
        float arity_similarity = 1.0f - std::abs(static_cast<int>(get_arity()) - static_cast<int>(other.get_arity())) / 10.0f;
        float strength_similarity = 1.0f - std::abs(metadata_.strength - other.metadata_.strength);
        
        return (type_similarity + arity_similarity + strength_similarity) / 3.0f;
    }
    
    // Tensor-based similarity
    float* this_data = (float*)tensor_data_->data;
    float* other_data = (float*)other.tensor_data_->data;
    
    size_t min_size = std::min(ggml_nelements(tensor_data_), ggml_nelements(other.tensor_data_));
    
    float dot_product = 0.0f;
    float this_norm = 0.0f;
    float other_norm = 0.0f;
    
    for (size_t i = 0; i < min_size; ++i) {
        dot_product += this_data[i] * other_data[i];
        this_norm += this_data[i] * this_data[i];
        other_norm += other_data[i] * other_data[i];
    }
    
    // Cosine similarity
    if (this_norm > 0.0f && other_norm > 0.0f) {
        return dot_product / (std::sqrt(this_norm) * std::sqrt(other_norm));
    }
    
    return 0.0f;
}

// =====================================================
// Helper Methods
// =====================================================

std::string Link::generate_link_id() const {
    std::stringstream ss;
    ss << "link_" << static_cast<int>(link_type_) << "_" 
       << std::hash<std::string>{}(std::to_string(connected_nodes_.size()));
    return ss.str();
}

void Link::validate_link_structure() const {
    switch (link_type_) {
        case LinkType::SIMPLE:
            if (connected_nodes_.size() != 2) {
                logger().warn("Simple link should have exactly 2 nodes, has %zu", connected_nodes_.size());
            }
            break;
        case LinkType::HYPEREDGE:
            if (connected_nodes_.size() < 3) {
                logger().warn("Hyperedge should have at least 3 nodes, has %zu", connected_nodes_.size());
            }
            break;
        default:
            // Other types are flexible
            break;
    }
}

void Link::initialize_default_features() {
    // Initialize relational features based on link type and arity
    relational_features_["link_type"] = static_cast<float>(link_type_);
    relational_features_["arity"] = static_cast<float>(connected_nodes_.size());
    relational_features_["bidirectional"] = metadata_.is_bidirectional ? 1.0f : 0.0f;
    relational_features_["strength"] = metadata_.strength;
    relational_features_["attention_weight"] = metadata_.attention_weight;
    relational_features_["temporal_order"] = static_cast<float>(metadata_.temporal_order) / 1000.0f; // Normalized
    
    // Type-specific features
    switch (link_type_) {
        case LinkType::INHERITANCE:
            relational_features_["inheritance_strength"] = 0.8f;
            relational_features_["hierarchical_depth"] = 0.5f;
            break;
        case LinkType::SIMILARITY:
            relational_features_["similarity_type"] = 0.7f;
            relational_features_["semantic_distance"] = 0.3f;
            break;
        case LinkType::EVALUATION:
            relational_features_["evaluation_confidence"] = 0.9f;
            relational_features_["logical_validity"] = 0.8f;
            break;
        case LinkType::ATTENTION:
            relational_features_["attention_focus"] = metadata_.attention_weight;
            relational_features_["attention_persistence"] = 0.6f;
            break;
        default:
            relational_features_["general_relatedness"] = 0.5f;
            break;
    }
}

void Link::cleanup_tensor_resources() {
    if (tensor_context_) {
        ggml_free(tensor_context_);
        tensor_context_ = nullptr;
    }
    tensor_data_ = nullptr; // Cleaned up by context
}

void Link::encode_features_to_tensor() {
    if (!tensor_data_) return;
    
    // Extract relational vector and encode to tensor
    encode_relationship_to_tensor();
}

void Link::decode_features_from_tensor() {
    if (!tensor_data_) return;
    
    const float* data = (const float*)tensor_data_->data;
    size_t tensor_size = ggml_nelements(tensor_data_);
    
    std::vector<float> relational_vector(data, data + tensor_size);
    load_relational_vector(relational_vector);
}

std::vector<float> Link::extract_relational_vector() const {
    std::vector<float> features;
    
    // Standard relational feature order for consistency
    std::vector<std::string> feature_names = {
        "link_type", "arity", "bidirectional", "strength", "attention_weight", "temporal_order",
        "inheritance_strength", "hierarchical_depth", "similarity_type", "semantic_distance",
        "evaluation_confidence", "logical_validity", "attention_focus", "attention_persistence",
        "general_relatedness"
    };
    
    // Add standard features
    for (const std::string& name : feature_names) {
        auto it = relational_features_.find(name);
        features.push_back(it != relational_features_.end() ? it->second : 0.0f);
    }
    
    // Add metadata as features
    features.push_back(static_cast<float>(connected_nodes_.size()) / 10.0f); // Normalized arity
    features.push_back(static_cast<float>(metadata_.annotations.size()) / 10.0f); // Annotation count
    
    // Pad to minimum size
    while (features.size() < 32) { // Minimum 32-dimensional representation
        features.push_back(0.0f);
    }
    
    return features;
}

void Link::load_relational_vector(const std::vector<float>& features) {
    if (features.size() < 16) return; // Need minimum features
    
    // Standard relational feature order
    std::vector<std::string> feature_names = {
        "link_type", "arity", "bidirectional", "strength", "attention_weight", "temporal_order",
        "inheritance_strength", "hierarchical_depth", "similarity_type", "semantic_distance",
        "evaluation_confidence", "logical_validity", "attention_focus", "attention_persistence",
        "general_relatedness"
    };
    
    // Load standard features
    for (size_t i = 0; i < std::min(feature_names.size(), features.size()); ++i) {
        relational_features_[feature_names[i]] = features[i];
    }
    
    // Update metadata from tensor
    if (features.size() > 3) {
        metadata_.strength = features[3];
        metadata_.attention_weight = features[4];
        metadata_.temporal_order = static_cast<size_t>(features[5] * 1000.0f);
        metadata_.is_bidirectional = features[2] > 0.5f;
    }
}

// =====================================================
// Operators
// =====================================================

bool Link::operator==(const Link& other) const {
    return link_id_ == other.link_id_ && 
           link_type_ == other.link_type_ && 
           connected_nodes_ == other.connected_nodes_;
}

bool Link::operator<(const Link& other) const {
    if (link_type_ != other.link_type_) {
        return link_type_ < other.link_type_;
    }
    if (connected_nodes_.size() != other.connected_nodes_.size()) {
        return connected_nodes_.size() < other.connected_nodes_.size();
    }
    return link_id_ < other.link_id_;
}

// =====================================================
// Factory Methods
// =====================================================

std::shared_ptr<Link> Link::create_inheritance_link(const std::string& child, const std::string& parent) {
    std::vector<std::string> nodes = {child, parent};
    auto link = std::make_shared<Link>(LinkType::INHERITANCE, nodes);
    link->set_annotation("child", child);
    link->set_annotation("parent", parent);
    return link;
}

std::shared_ptr<Link> Link::create_similarity_link(const std::string& node1, const std::string& node2, float strength) {
    std::vector<std::string> nodes = {node1, node2};
    auto link = std::make_shared<Link>(LinkType::SIMILARITY, nodes);
    link->metadata_.strength = strength;
    link->metadata_.is_bidirectional = true;
    return link;
}

std::shared_ptr<Link> Link::create_evaluation_link(const std::string& predicate, const std::vector<std::string>& arguments) {
    std::vector<std::string> nodes = {predicate};
    nodes.insert(nodes.end(), arguments.begin(), arguments.end());
    auto link = std::make_shared<Link>(LinkType::EVALUATION, nodes);
    link->set_annotation("predicate", predicate);
    return link;
}

std::shared_ptr<Link> Link::create_sequence_link(const std::vector<std::string>& ordered_nodes) {
    auto link = std::make_shared<Link>(LinkType::SEQUENCE, ordered_nodes);
    for (size_t i = 0; i < ordered_nodes.size(); ++i) {
        link->set_annotation("order_" + std::to_string(i), std::to_string(i));
    }
    return link;
}

std::shared_ptr<Link> Link::create_attention_link(const std::string& source, const std::string& target, float weight) {
    std::vector<std::string> nodes = {source, target};
    auto link = std::make_shared<Link>(LinkType::ATTENTION, nodes);
    link->metadata_.attention_weight = weight;
    link->set_annotation("attention_type", "focus");
    return link;
}

}} // namespace opencog::agentic