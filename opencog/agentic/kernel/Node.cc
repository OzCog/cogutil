/*
 * opencog/agentic/kernel/Node.cc
 *
 * Implementation of Node - Basic cognitive entity with GGML tensor backend
 */

#include "Node.h"
#include <opencog/util/Logger.h>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>

// GGML headers
#include "ggml_stub.h"

namespace opencog { namespace agentic {

// =====================================================
// Node Implementation
// =====================================================

Node::Node(const std::string& node_id, NodeType type) 
    : node_id_(node_id), name_(node_id), node_type_(type) {
    
    metadata_.created = std::chrono::system_clock::now();
    metadata_.last_accessed = metadata_.created;
    
    initialize_default_features();
    
    logger().debug("Created node: %s (type: %d)", node_id_.c_str(), static_cast<int>(node_type_));
}

Node::Node(const std::string& node_id, const std::string& name, NodeType type)
    : node_id_(node_id), name_(name), node_type_(type) {
    
    metadata_.created = std::chrono::system_clock::now();
    metadata_.last_accessed = metadata_.created;
    
    initialize_default_features();
    
    logger().debug("Created node: %s (%s) (type: %d)", node_id_.c_str(), name_.c_str(), static_cast<int>(node_type_));
}

Node::~Node() {
    cleanup_tensor_resources();
}

void Node::set_name(const std::string& name) {
    name_ = name;
    
    // Update semantic features based on new name
    initialize_default_features();
    
    // Update tensor representation if it exists
    if (has_tensor_representation()) {
        update_tensor_from_features();
    }
}

bool Node::create_tensor_representation(size_t feature_dimension) {
    // Clean up existing tensor if present
    cleanup_tensor_resources();
    
    // Initialize GGML context
    struct ggml_init_params params = {
        .mem_size   = 64 * 1024 * 1024, // 64MB for node tensors
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    
    tensor_context_ = ggml_init(params);
    if (!tensor_context_) {
        logger().error("Failed to initialize GGML context for node: %s", node_id_.c_str());
        return false;
    }
    
    // Create feature tensor
    tensor_data_ = ggml_new_tensor_1d(tensor_context_, GGML_TYPE_F32, feature_dimension);
    if (!tensor_data_) {
        logger().error("Failed to create tensor for node: %s", node_id_.c_str());
        cleanup_tensor_resources();
        return false;
    }
    
    // Initialize tensor with current features
    return update_tensor_from_features();
}

bool Node::update_tensor_from_features() {
    if (!tensor_data_ || !tensor_context_) {
        return false;
    }
    
    float* data = (float*)tensor_data_->data;
    size_t tensor_size = ggml_nelements(tensor_data_);
    
    // Clear tensor first
    std::fill(data, data + tensor_size, 0.0f);
    
    // Encode semantic features into tensor
    encode_features_to_tensor();
    
    return true;
}

bool Node::sync_features_from_tensor() {
    if (!tensor_data_ || !tensor_context_) {
        return false;
    }
    
    // Decode features from tensor
    decode_features_from_tensor();
    
    return true;
}

void Node::set_semantic_feature(const std::string& feature, float value) {
    semantic_features_[feature] = value;
    
    // Update tensor if it exists
    if (has_tensor_representation()) {
        update_tensor_from_features();
    }
}

float Node::get_semantic_feature(const std::string& feature) const {
    auto it = semantic_features_.find(feature);
    return it != semantic_features_.end() ? it->second : 0.0f;
}

void Node::set_semantic_features(const std::map<std::string, float>& features) {
    semantic_features_ = features;
    
    // Update tensor if it exists
    if (has_tensor_representation()) {
        update_tensor_from_features();
    }
}

void Node::set_attention_value(float attention) {
    metadata_.attention_value = attention;
    
    // Update tensor representation to include attention
    if (has_tensor_representation()) {
        update_tensor_from_features();
    }
}

void Node::update_activation(float activation) {
    metadata_.activation_level = activation;
    
    // Attention spreading: higher activation increases attention
    float attention_boost = activation * 0.1f; // 10% of activation becomes attention
    metadata_.attention_value = std::min(metadata_.attention_value + attention_boost, 1.0f);
    
    if (has_tensor_representation()) {
        update_tensor_from_features();
    }
}

float Node::calculate_similarity(const Node& other) const {
    // If both have tensors, use tensor similarity
    if (has_tensor_representation() && other.has_tensor_representation()) {
        return calculate_tensor_similarity(other);
    }
    
    // Fall back to feature-based similarity
    float similarity = 0.0f;
    size_t common_features = 0;
    
    for (const auto& feature : semantic_features_) {
        auto other_it = other.semantic_features_.find(feature.first);
        if (other_it != other.semantic_features_.end()) {
            // Calculate cosine similarity for this feature
            float dot_product = feature.second * other_it->second;
            float magnitude1 = std::abs(feature.second);
            float magnitude2 = std::abs(other_it->second);
            
            if (magnitude1 > 0 && magnitude2 > 0) {
                similarity += dot_product / (magnitude1 * magnitude2);
                common_features++;
            }
        }
    }
    
    return common_features > 0 ? similarity / common_features : 0.0f;
}

float Node::calculate_tensor_similarity(const Node& other) const {
    if (!has_tensor_representation() || !other.has_tensor_representation()) {
        return 0.0f;
    }
    
    const float* data1 = (const float*)tensor_data_->data;
    const float* data2 = (const float*)other.tensor_data_->data;
    
    size_t size1 = ggml_nelements(tensor_data_);
    size_t size2 = ggml_nelements(other.tensor_data_);
    size_t min_size = std::min(size1, size2);
    
    if (min_size == 0) return 0.0f;
    
    // Calculate cosine similarity
    float dot_product = 0.0f;
    float magnitude1 = 0.0f;
    float magnitude2 = 0.0f;
    
    for (size_t i = 0; i < min_size; ++i) {
        dot_product += data1[i] * data2[i];
        magnitude1 += data1[i] * data1[i];
        magnitude2 += data2[i] * data2[i];
    }
    
    magnitude1 = std::sqrt(magnitude1);
    magnitude2 = std::sqrt(magnitude2);
    
    if (magnitude1 > 0 && magnitude2 > 0) {
        return dot_product / (magnitude1 * magnitude2);
    }
    
    return 0.0f;
}

bool Node::matches_pattern(const Node& pattern) const {
    // Pattern matching based on node type and key features
    if (node_type_ != pattern.node_type_) {
        return false;
    }
    
    // Check if pattern features are satisfied
    for (const auto& pattern_feature : pattern.semantic_features_) {
        float my_value = get_semantic_feature(pattern_feature.first);
        float pattern_value = pattern_feature.second;
        
        // Allow some tolerance in matching
        if (std::abs(my_value - pattern_value) > 0.1f) {
            return false;
        }
    }
    
    return true;
}

TensorShape Node::get_tensor_shape() const {
    if (!has_tensor_representation()) {
        return TensorShape();
    }
    
    return TensorShape::from_ggml_tensor(tensor_data_);
}

void Node::update_access_statistics() {
    metadata_.access_count++;
    metadata_.last_accessed = std::chrono::system_clock::now();
    
    // Increase attention based on access frequency
    float attention_boost = 0.01f; // Small boost per access
    metadata_.attention_value = std::min(metadata_.attention_value + attention_boost, 1.0f);
    
    if (has_tensor_representation()) {
        update_tensor_from_features();
    }
}

std::string Node::to_string() const {
    std::ostringstream oss;
    oss << "Node(" << node_id_ << " : " << name_ << " : " << static_cast<int>(node_type_) << ")";
    oss << " [attention=" << metadata_.attention_value;
    oss << ", activation=" << metadata_.activation_level;
    oss << ", features=" << semantic_features_.size();
    if (has_tensor_representation()) {
        oss << ", tensor=" << ggml_nelements(tensor_data_) << "D";
    }
    oss << "]";
    return oss.str();
}

std::string Node::to_scheme_representation() const {
    std::ostringstream oss;
    
    // Generate Scheme representation based on node type
    switch (node_type_) {
        case NodeType::CONCEPT:
            oss << "(ConceptNode \"" << name_ << "\")";
            break;
        case NodeType::PREDICATE:
            oss << "(PredicateNode \"" << name_ << "\")";
            break;
        case NodeType::WORD:
            oss << "(WordNode \"" << name_ << "\")";
            break;
        case NodeType::NUMBER:
            oss << "(NumberNode \"" << name_ << "\")";
            break;
        case NodeType::VARIABLE:
            oss << "(VariableNode \"" << name_ << "\")";
            break;
        default:
            oss << "(Node \"" << name_ << "\")";
            break;
    }
    
    return oss.str();
}

// Factory methods
std::shared_ptr<Node> Node::create_concept_node(const std::string& name) {
    std::string node_id = "concept_" + name + "_" + std::to_string(std::hash<std::string>{}(name));
    return std::make_shared<Node>(node_id, name, NodeType::CONCEPT);
}

std::shared_ptr<Node> Node::create_predicate_node(const std::string& name) {
    std::string node_id = "predicate_" + name + "_" + std::to_string(std::hash<std::string>{}(name));
    return std::make_shared<Node>(node_id, name, NodeType::PREDICATE);
}

std::shared_ptr<Node> Node::create_word_node(const std::string& word) {
    std::string node_id = "word_" + word + "_" + std::to_string(std::hash<std::string>{}(word));
    return std::make_shared<Node>(node_id, word, NodeType::WORD);
}

std::shared_ptr<Node> Node::create_number_node(const std::string& name, float value) {
    std::string node_id = "number_" + name + "_" + std::to_string(std::hash<std::string>{}(name));
    auto node = std::make_shared<Node>(node_id, name, NodeType::NUMBER);
    node->set_semantic_feature("numeric_value", value);
    return node;
}

bool Node::operator==(const Node& other) const {
    return node_id_ == other.node_id_ && 
           name_ == other.name_ && 
           node_type_ == other.node_type_;
}

bool Node::operator<(const Node& other) const {
    if (node_type_ != other.node_type_) {
        return node_type_ < other.node_type_;
    }
    return node_id_ < other.node_id_;
}

// Private helper methods
std::string Node::generate_node_id() const {
    std::ostringstream oss;
    oss << "node_" << static_cast<int>(node_type_) << "_" << std::hash<std::string>{}(name_);
    return oss.str();
}

void Node::initialize_default_features() {
    // Clear existing features
    semantic_features_.clear();
    
    // Add type-specific default features
    semantic_features_["node_type"] = static_cast<float>(node_type_);
    semantic_features_["name_length"] = static_cast<float>(name_.length());
    semantic_features_["name_hash"] = static_cast<float>(std::hash<std::string>{}(name_) % 1000) / 1000.0f;
    
    // Add type-specific features
    switch (node_type_) {
        case NodeType::CONCEPT:
            semantic_features_["conceptual_depth"] = 0.5f;
            semantic_features_["abstractness"] = 0.7f;
            break;
        case NodeType::PREDICATE:
            semantic_features_["relational_arity"] = 2.0f;
            semantic_features_["logical_strength"] = 0.8f;
            break;
        case NodeType::WORD:
            semantic_features_["linguistic_frequency"] = 0.3f;
            semantic_features_["phonetic_complexity"] = static_cast<float>(name_.length()) / 20.0f;
            break;
        case NodeType::NUMBER:
            semantic_features_["numeric_type"] = 1.0f;
            semantic_features_["mathematical_significance"] = 0.5f;
            break;
        default:
            semantic_features_["generality"] = 0.5f;
            break;
    }
}

void Node::cleanup_tensor_resources() {
    if (tensor_context_) {
        ggml_free(tensor_context_);
        tensor_context_ = nullptr;
        tensor_data_ = nullptr; // freed with context
    }
}

void Node::encode_features_to_tensor() {
    if (!tensor_data_) return;
    
    float* data = (float*)tensor_data_->data;
    size_t tensor_size = ggml_nelements(tensor_data_);
    
    // Extract feature vector
    std::vector<float> feature_vector = extract_feature_vector();
    
    // Copy features to tensor (with padding/truncation as needed)
    size_t copy_size = std::min(tensor_size, feature_vector.size());
    std::copy(feature_vector.begin(), feature_vector.begin() + copy_size, data);
    
    // Pad remaining with zeros
    if (copy_size < tensor_size) {
        std::fill(data + copy_size, data + tensor_size, 0.0f);
    }
}

void Node::decode_features_from_tensor() {
    if (!tensor_data_) return;
    
    const float* data = (const float*)tensor_data_->data;
    size_t tensor_size = ggml_nelements(tensor_data_);
    
    std::vector<float> feature_vector(data, data + tensor_size);
    load_feature_vector(feature_vector);
}

std::vector<float> Node::extract_feature_vector() const {
    std::vector<float> features;
    
    // Standard feature order for consistency
    std::vector<std::string> feature_names = {
        "node_type", "name_length", "name_hash",
        "conceptual_depth", "abstractness", "relational_arity", 
        "logical_strength", "linguistic_frequency", "phonetic_complexity",
        "numeric_type", "mathematical_significance", "generality"
    };
    
    // Add standard features
    for (const std::string& name : feature_names) {
        features.push_back(get_semantic_feature(name));
    }
    
    // Add metadata as features
    features.push_back(metadata_.attention_value);
    features.push_back(metadata_.activation_level);
    features.push_back(metadata_.confidence);
    features.push_back(static_cast<float>(metadata_.access_count) / 1000.0f); // Normalized
    
    // Add custom features (up to some limit)
    size_t custom_limit = 20;
    size_t custom_count = 0;
    for (const auto& custom_feature : semantic_features_) {
        bool is_standard = std::find(feature_names.begin(), feature_names.end(), custom_feature.first) != feature_names.end();
        if (!is_standard && custom_count < custom_limit) {
            features.push_back(custom_feature.second);
            custom_count++;
        }
    }
    
    // Pad to minimum size
    while (features.size() < 32) { // Minimum 32-dimensional representation
        features.push_back(0.0f);
    }
    
    return features;
}

void Node::load_feature_vector(const std::vector<float>& features) {
    if (features.size() < 16) return; // Need minimum features
    
    // Standard feature order
    std::vector<std::string> feature_names = {
        "node_type", "name_length", "name_hash",
        "conceptual_depth", "abstractness", "relational_arity", 
        "logical_strength", "linguistic_frequency", "phonetic_complexity",
        "numeric_type", "mathematical_significance", "generality"
    };
    
    // Load standard features
    for (size_t i = 0; i < std::min(feature_names.size(), features.size()); ++i) {
        semantic_features_[feature_names[i]] = features[i];
    }
    
    // Load metadata if available
    if (features.size() > 12) {
        metadata_.attention_value = features[12];
        metadata_.activation_level = features[13];
        metadata_.confidence = features[14];
        metadata_.access_count = static_cast<size_t>(features[15] * 1000.0f);
    }
}

// =====================================================
// NodeFactory Implementation
// =====================================================

std::shared_ptr<Node> NodeFactory::create_node(const std::string& name, 
                                              Node::NodeType type,
                                              const NodeConfig& config) {
    std::string node_id = "factory_" + name + "_" + std::to_string(std::hash<std::string>{}(name));
    auto node = std::make_shared<Node>(node_id, name, type);
    
    // Apply configuration
    if (!config.initial_features.empty()) {
        node->set_semantic_features(config.initial_features);
    }
    
    if (config.default_attention_value > 0) {
        node->set_attention_value(config.default_attention_value);
    }
    
    if (config.enable_tensor_backend) {
        node->create_tensor_representation(config.feature_dimension);
    }
    
    if (config.initialize_with_random_features) {
        // Add some random features for diversity
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        
        for (int i = 0; i < 10; ++i) {
            std::string feature_name = "random_feature_" + std::to_string(i);
            node->set_semantic_feature(feature_name, dist(gen));
        }
    }
    
    return node;
}

std::shared_ptr<Node> NodeFactory::create_node(const std::string& name, 
                                              Node::NodeType type) {
    return create_node(name, type, default_config());
}

NodeFactory::NodeConfig NodeFactory::default_config() {
    NodeConfig config;
    config.feature_dimension = 128;
    config.enable_tensor_backend = true;
    config.initialize_with_random_features = false;
    config.default_attention_value = 0.0f;
    return config;
}

std::vector<std::shared_ptr<Node>> NodeFactory::create_node_collection(
    const std::vector<std::string>& names,
    Node::NodeType type,
    const NodeConfig& config) {
    
    std::vector<std::shared_ptr<Node>> nodes;
    nodes.reserve(names.size());
    
    for (const std::string& name : names) {
        nodes.push_back(create_node(name, type, config));
    }
    
    return nodes;
}

std::vector<std::shared_ptr<Node>> NodeFactory::create_node_collection(
    const std::vector<std::string>& names,
    Node::NodeType type) {
    
    return create_node_collection(names, type, default_config());
}

// =====================================================
// NodeRegistry Implementation
// =====================================================

NodeRegistry& NodeRegistry::instance() {
    static NodeRegistry instance_;
    return instance_;
}

void NodeRegistry::register_node(std::shared_ptr<Node> node) {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    const std::string& node_id = node->get_node_id();
    nodes_[node_id] = node;
    
    logger().debug("Registered node: %s", node_id.c_str());
}

void NodeRegistry::unregister_node(const std::string& node_id) {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    auto it = nodes_.find(node_id);
    if (it != nodes_.end()) {
        logger().debug("Unregistered node: %s", node_id.c_str());
        nodes_.erase(it);
    }
}

std::shared_ptr<Node> NodeRegistry::get_node(const std::string& node_id) {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    auto it = nodes_.find(node_id);
    if (it != nodes_.end()) {
        it->second->update_access_statistics();
        return it->second;
    }
    return nullptr;
}

std::vector<std::shared_ptr<Node>> NodeRegistry::get_nodes_by_type(Node::NodeType type) {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    std::vector<std::shared_ptr<Node>> result;
    for (const auto& pair : nodes_) {
        if (pair.second->get_node_type() == type) {
            result.push_back(pair.second);
        }
    }
    return result;
}

std::vector<std::shared_ptr<Node>> NodeRegistry::find_similar_nodes(const Node& query_node, float similarity_threshold) {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    std::vector<std::shared_ptr<Node>> result;
    for (const auto& pair : nodes_) {
        float similarity = query_node.calculate_similarity(*pair.second);
        if (similarity >= similarity_threshold) {
            result.push_back(pair.second);
        }
    }
    
    // Sort by similarity (descending)
    std::sort(result.begin(), result.end(), 
              [&query_node](const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b) {
                  return query_node.calculate_similarity(*a) > query_node.calculate_similarity(*b);
              });
    
    return result;
}

NodeRegistry::RegistryStats NodeRegistry::get_stats() const {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    RegistryStats stats;
    stats.total_nodes = nodes_.size();
    
    for (const auto& pair : nodes_) {
        const auto& node = pair.second;
        
        if (node->has_tensor_representation()) {
            stats.nodes_with_tensors++;
            // Estimate memory usage (simplified)
            stats.total_tensor_memory += node->get_tensor_shape().estimate_ggml_memory_size();
        }
        
        stats.nodes_by_type[node->get_node_type()]++;
    }
    
    return stats;
}

}} // namespace opencog::agentic