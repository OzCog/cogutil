/*
 * opencog/agentic/grammar/TokenParsingAgent.cc
 *
 * Implementation of the Token Parsing Agent for the agentic kernel network.
 */

#include "TokenParsingAgent.h"
#include <opencog/util/Logger.h>
#include <sstream>
#include <algorithm>
#include <cctype>

namespace opencog { namespace agentic { namespace grammar {

TokenParsingAgent::TokenParsingAgent(const KernelConfig& config, const TokenParsingConfig& parsing_config)
    : AgenticKernel(config), parsing_config_(parsing_config) {
    
    logger().info("TokenParsingAgent initialized: %s (strategy: %d)", 
                  config.kernel_id.c_str(), static_cast<int>(parsing_config_.strategy));
}

ProcessingResult TokenParsingAgent::process(const CognitiveData& input) {
    ProcessingResult result;
    
    // Extract input text from the cognitive data
    std::string input_text;
    if (input.has_symbolic() && !input.symbolic_tree.empty()) {
        // Convert tree to string representation
        std::ostringstream oss;
        oss << input.symbolic_tree;
        input_text = oss.str();
    } else {
        // Check metadata for raw text input
        auto text_it = input.metadata.find("raw_text");
        if (text_it != input.metadata.end()) {
            input_text = text_it->second;
        }
    }
    
    if (input_text.empty()) {
        logger().warn("TokenParsingAgent received empty input");
        return result;
    }
    
    // Tokenize based on configured strategy
    std::vector<std::string> tokens;
    switch (parsing_config_.strategy) {
        case TokenizationStrategy::WHITESPACE:
            tokens = tokenize_whitespace(input_text);
            break;
        case TokenizationStrategy::REGEX_BASED:
            tokens = tokenize_regex(input_text);
            break;
        case TokenizationStrategy::GRAMMAR_RULES:
            tokens = tokenize_grammar(input_text);
            break;
        case TokenizationStrategy::HYBRID:
            tokens = tokenize_hybrid(input_text);
            break;
    }
    
    // Build hierarchical tree structure
    tree<std::string> token_tree = build_token_tree(tokens);
    
    // Create output cognitive data
    result.output_data.symbolic_tree = token_tree;
    result.output_data.data_id = "parsed_" + input.data_id;
    result.output_data.metadata = input.metadata;
    result.output_data.metadata["processing_stage"] = "tokenized";
    result.output_data.metadata["token_count"] = std::to_string(tokens.size());
    
    // Assign attention weights based on token importance
    assign_attention_weights(result.output_data);
    
    // Set processing costs and value estimates
    result.processing_cost = static_cast<float>(tokens.size()) * 0.1f; // Simple cost model
    result.estimated_value = static_cast<float>(tokens.size()) * 0.5f; // Value based on information content
    
    // Suggest next processing stages
    result.suggested_next_kernels.push_back("grammar_kernel");
    result.suggested_next_kernels.push_back("tensor_mapper");
    
    logger().debug("TokenParsingAgent processed %zu tokens from input", tokens.size());
    return result;
}

float TokenParsingAgent::estimate_processing_cost(const CognitiveData& input) const {
    // Estimate cost based on input size
    float base_cost = 1.0f;
    
    if (input.has_symbolic()) {
        base_cost += static_cast<float>(input.symbolic_tree.size()) * 0.1f;
    }
    
    auto text_it = input.metadata.find("raw_text");
    if (text_it != input.metadata.end()) {
        base_cost += static_cast<float>(text_it->second.length()) * 0.01f;
    }
    
    // Different strategies have different costs
    switch (parsing_config_.strategy) {
        case TokenizationStrategy::WHITESPACE:
            return base_cost * 1.0f;
        case TokenizationStrategy::REGEX_BASED:
            return base_cost * 1.5f;
        case TokenizationStrategy::GRAMMAR_RULES:
            return base_cost * 2.0f;
        case TokenizationStrategy::HYBRID:
            return base_cost * 2.5f;
    }
    
    return base_cost;
}

float TokenParsingAgent::estimate_output_value(const CognitiveData& input) const {
    // Value estimation based on information content
    float base_value = 1.0f;
    
    auto text_it = input.metadata.find("raw_text");
    if (text_it != input.metadata.end()) {
        // Simple heuristic: longer text usually contains more information
        float length_factor = std::min(static_cast<float>(text_it->second.length()) / 100.0f, 10.0f);
        base_value *= length_factor;
    }
    
    // More sophisticated strategies potentially provide more value
    switch (parsing_config_.strategy) {
        case TokenizationStrategy::GRAMMAR_RULES:
        case TokenizationStrategy::HYBRID:
            base_value *= 1.5f;
            break;
        default:
            break;
    }
    
    return base_value;
}

void TokenParsingAgent::set_tokenization_strategy(TokenizationStrategy strategy) {
    parsing_config_.strategy = strategy;
    logger().info("TokenParsingAgent strategy changed to: %d", static_cast<int>(strategy));
}

void TokenParsingAgent::add_regex_pattern(const std::regex& pattern) {
    parsing_config_.regex_patterns.push_back(pattern);
}

void TokenParsingAgent::set_token_weight(const std::string& token, float weight) {
    parsing_config_.token_weights[token] = weight;
}

std::vector<std::string> TokenParsingAgent::tokenize_whitespace(const std::string& input) const {
    std::vector<std::string> tokens;
    
    // Use the existing opencog tokenize function
    std::vector<std::string> temp_tokens;
    std::string delims;
    for (const auto& delim : parsing_config_.delimiters) {
        delims += delim;
    }
    
    tokenize(input, std::back_inserter(temp_tokens), delims);
    
    // Filter out empty tokens and normalize
    for (const auto& token : temp_tokens) {
        std::string normalized = normalize_token(token);
        if (!normalized.empty()) {
            tokens.push_back(normalized);
        }
    }
    
    return tokens;
}

std::vector<std::string> TokenParsingAgent::tokenize_regex(const std::string& input) const {
    std::vector<std::string> tokens;
    
    if (parsing_config_.regex_patterns.empty()) {
        // Fall back to whitespace tokenization
        return tokenize_whitespace(input);
    }
    
    std::string remaining = input;
    
    for (const auto& pattern : parsing_config_.regex_patterns) {
        std::sregex_iterator iter(remaining.begin(), remaining.end(), pattern);
        std::sregex_iterator end;
        
        for (; iter != end; ++iter) {
            std::string match = iter->str();
            std::string normalized = normalize_token(match);
            if (!normalized.empty()) {
                tokens.push_back(normalized);
            }
        }
    }
    
    return tokens;
}

std::vector<std::string> TokenParsingAgent::tokenize_grammar(const std::string& input) const {
    // Simplified grammar-based tokenization
    // In a full implementation, this would use a proper parser
    std::vector<std::string> tokens;
    
    // For now, implement a simple bracket-aware tokenization
    std::string current_token;
    int bracket_depth = 0;
    
    for (char c : input) {
        if (c == '(' || c == '[' || c == '{') {
            if (!current_token.empty() && bracket_depth == 0) {
                tokens.push_back(normalize_token(current_token));
                current_token.clear();
            }
            current_token += c;
            bracket_depth++;
        } else if (c == ')' || c == ']' || c == '}') {
            current_token += c;
            bracket_depth--;
            if (bracket_depth == 0) {
                tokens.push_back(normalize_token(current_token));
                current_token.clear();
            }
        } else if (std::isspace(c) && bracket_depth == 0) {
            if (!current_token.empty()) {
                tokens.push_back(normalize_token(current_token));
                current_token.clear();
            }
        } else {
            current_token += c;
        }
    }
    
    if (!current_token.empty()) {
        tokens.push_back(normalize_token(current_token));
    }
    
    return tokens;
}

std::vector<std::string> TokenParsingAgent::tokenize_hybrid(const std::string& input) const {
    // Combine multiple strategies
    auto whitespace_tokens = tokenize_whitespace(input);
    auto grammar_tokens = tokenize_grammar(input);
    
    // Merge and deduplicate
    std::vector<std::string> combined_tokens = whitespace_tokens;
    for (const auto& token : grammar_tokens) {
        if (std::find(combined_tokens.begin(), combined_tokens.end(), token) == combined_tokens.end()) {
            combined_tokens.push_back(token);
        }
    }
    
    return combined_tokens;
}

tree<std::string> TokenParsingAgent::build_token_tree(const std::vector<std::string>& tokens) const {
    tree<std::string> token_tree;
    
    if (tokens.empty()) {
        return token_tree;
    }
    
    // Create root node
    token_tree.set_head("token_root");
    auto root = token_tree.begin();
    
    if (parsing_config_.preserve_structure) {
        // Try to preserve hierarchical structure for structured tokens
        for (const auto& token : tokens) {
            if (is_structural_token(token)) {
                // Parse structured tokens (like "(func arg1 arg2)")
                tree<std::string> sub_tree = parse_symbolic_input(token);
                if (!sub_tree.empty()) {
                    token_tree.append_child(root, sub_tree.begin());
                } else {
                    token_tree.append_child(root, token);
                }
            } else {
                token_tree.append_child(root, token);
            }
        }
    } else {
        // Simple flat structure
        for (const auto& token : tokens) {
            token_tree.append_child(root, token);
        }
    }
    
    return token_tree;
}

void TokenParsingAgent::assign_attention_weights(CognitiveData& data) const {
    // Assign attention weights based on token importance
    for (auto it = data.symbolic_tree.begin(); it != data.symbolic_tree.end(); ++it) {
        float importance = calculate_token_importance(*it);
        data.set_attention_weight(*it, importance);
    }
    
    // Set overall attention weights for different aspects
    data.set_attention_weight("structural_importance", 0.8f);
    data.set_attention_weight("content_importance", 0.6f);
    data.set_attention_weight("processing_priority", 0.7f);
}

float TokenParsingAgent::calculate_token_importance(const std::string& token) const {
    // Check predefined weights
    auto weight_it = parsing_config_.token_weights.find(token);
    if (weight_it != parsing_config_.token_weights.end()) {
        return weight_it->second;
    }
    
    // Default importance calculation
    float importance = 0.5f; // Base importance
    
    // Longer tokens tend to be more informative
    importance += std::min(static_cast<float>(token.length()) / 20.0f, 0.3f);
    
    // Structural tokens are important
    if (is_structural_token(token)) {
        importance += 0.2f;
    }
    
    // Tokens with special characters might be keywords
    if (token.find_first_of("_-.:") != std::string::npos) {
        importance += 0.1f;
    }
    
    return std::min(importance, 1.0f);
}

bool TokenParsingAgent::is_structural_token(const std::string& token) const {
    return (token.front() == '(' && token.back() == ')') ||
           (token.front() == '[' && token.back() == ']') ||
           (token.front() == '{' && token.back() == '}');
}

std::string TokenParsingAgent::normalize_token(const std::string& token) const {
    std::string normalized = token;
    
    // Remove leading/trailing whitespace
    normalized.erase(0, normalized.find_first_not_of(" \t\n\r"));
    normalized.erase(normalized.find_last_not_of(" \t\n\r") + 1);
    
    return normalized;
}

// =====================================================
// TokenParsingAgentFactory Implementation
// =====================================================

std::shared_ptr<TokenParsingAgent> TokenParsingAgentFactory::create_simple_parser(const std::string& kernel_id) {
    KernelConfig config(kernel_id, "token_parser");
    TokenParsingAgent::TokenParsingConfig parsing_config;
    parsing_config.strategy = TokenParsingAgent::TokenizationStrategy::WHITESPACE;
    
    return std::make_shared<TokenParsingAgent>(config, parsing_config);
}

std::shared_ptr<TokenParsingAgent> TokenParsingAgentFactory::create_regex_parser(
    const std::string& kernel_id, const std::vector<std::regex>& patterns) {
    
    KernelConfig config(kernel_id, "token_parser");
    TokenParsingAgent::TokenParsingConfig parsing_config;
    parsing_config.strategy = TokenParsingAgent::TokenizationStrategy::REGEX_BASED;
    parsing_config.regex_patterns = patterns;
    
    return std::make_shared<TokenParsingAgent>(config, parsing_config);
}

std::shared_ptr<TokenParsingAgent> TokenParsingAgentFactory::create_grammar_parser(
    const std::string& kernel_id, const std::vector<std::string>& grammar_rules) {
    
    KernelConfig config(kernel_id, "token_parser");
    TokenParsingAgent::TokenParsingConfig parsing_config;
    parsing_config.strategy = TokenParsingAgent::TokenizationStrategy::GRAMMAR_RULES;
    
    return std::make_shared<TokenParsingAgent>(config, parsing_config);
}

std::shared_ptr<TokenParsingAgent> TokenParsingAgentFactory::create_hybrid_parser(const std::string& kernel_id) {
    KernelConfig config(kernel_id, "token_parser");
    TokenParsingAgent::TokenParsingConfig parsing_config;
    parsing_config.strategy = TokenParsingAgent::TokenizationStrategy::HYBRID;
    parsing_config.preserve_structure = true;
    
    return std::make_shared<TokenParsingAgent>(config, parsing_config);
}

}}} // namespace opencog::agentic::grammar