/*
 * opencog/agentic/grammar/TokenParsingAgent.h
 *
 * Token Parsing Agent - converts input data streams into symbolic tokens
 * for processing by the agentic kernel network.
 *
 * Copyright (C) 2024 OpenCog Foundation
 */

#ifndef _OPENCOG_TOKEN_PARSING_AGENT_H
#define _OPENCOG_TOKEN_PARSING_AGENT_H

#include <opencog/agentic/kernel/AgenticKernel.h>
#include <opencog/util/misc.h>
#include <regex>

namespace opencog { namespace agentic { namespace grammar {

/**
 * Token Parsing Agent - First stage in the cognitive processing pipeline.
 * 
 * Converts raw input text/data into structured symbolic representations
 * that can be processed by other agentic kernels in the network.
 * 
 * Features:
 * - Multiple tokenization strategies (whitespace, regex, grammar-based)
 * - Hierarchical token structure using tree representations
 * - Attention weight assignment based on token importance
 * - Integration with GGML for tensor-based token embeddings
 */
class TokenParsingAgent : public AgenticKernel {
public:
    enum class TokenizationStrategy {
        WHITESPACE,     // Simple whitespace splitting
        REGEX_BASED,    // Regex pattern matching
        GRAMMAR_RULES,  // Grammar-rule based parsing
        HYBRID          // Adaptive combination of strategies
    };
    
    struct TokenParsingConfig {
        TokenizationStrategy strategy = TokenizationStrategy::WHITESPACE;
        std::vector<std::string> delimiters = {" ", "\t", "\n"};
        std::vector<std::regex> regex_patterns;
        std::map<std::string, float> token_weights; // Pre-defined importance weights
        bool create_tensor_embeddings = true;
        bool preserve_structure = true;
    };

public:
    TokenParsingAgent(const KernelConfig& config, const TokenParsingConfig& parsing_config);
    virtual ~TokenParsingAgent() = default;

    // AgenticKernel interface implementation
    ProcessingResult process(const CognitiveData& input) override;
    float estimate_processing_cost(const CognitiveData& input) const override;
    float estimate_output_value(const CognitiveData& input) const override;

    // Token parsing specific methods
    void set_tokenization_strategy(TokenizationStrategy strategy);
    void add_regex_pattern(const std::regex& pattern);
    void set_token_weight(const std::string& token, float weight);
    void update_grammar_rules(const std::vector<std::string>& rules);

private:
    TokenParsingConfig parsing_config_;
    
    // Tokenization methods
    std::vector<std::string> tokenize_whitespace(const std::string& input) const;
    std::vector<std::string> tokenize_regex(const std::string& input) const;
    std::vector<std::string> tokenize_grammar(const std::string& input) const;
    std::vector<std::string> tokenize_hybrid(const std::string& input) const;
    
    // Tree structure building
    tree<std::string> build_token_tree(const std::vector<std::string>& tokens) const;
    void assign_attention_weights(CognitiveData& data) const;
    
    // Helper methods
    float calculate_token_importance(const std::string& token) const;
    bool is_structural_token(const std::string& token) const;
    std::string normalize_token(const std::string& token) const;
};

/**
 * Factory for creating pre-configured token parsing agents.
 */
class TokenParsingAgentFactory {
public:
    static std::shared_ptr<TokenParsingAgent> create_simple_parser(const std::string& kernel_id);
    static std::shared_ptr<TokenParsingAgent> create_regex_parser(const std::string& kernel_id, 
                                                                  const std::vector<std::regex>& patterns);
    static std::shared_ptr<TokenParsingAgent> create_grammar_parser(const std::string& kernel_id,
                                                                    const std::vector<std::string>& grammar_rules);
    static std::shared_ptr<TokenParsingAgent> create_hybrid_parser(const std::string& kernel_id);
};

}}} // namespace opencog::agentic::grammar

#endif // _OPENCOG_TOKEN_PARSING_AGENT_H