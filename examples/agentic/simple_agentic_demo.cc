/*
 * examples/agentic/simple_agentic_demo.cc
 *
 * Demonstration of the Orchestral Architect system:
 * Shows basic operation of agentic kernels, attention allocation,
 * and distributed cognitive processing.
 *
 * Copyright (C) 2024 OpenCog Foundation
 */

#include <iostream>
#include <string>
#include <memory>
#include <vector>

#include <opencog/agentic/kernel/AgenticKernel.h>
#include <opencog/agentic/grammar/TokenParsingAgent.h>
#include <opencog/agentic/attention/AttentionAllocationEngine.h>
#include <opencog/util/Logger.h>

using namespace opencog::agentic;
using namespace opencog::agentic::grammar;
using namespace opencog::agentic::attention;

/**
 * Simple Echo Kernel - demonstrates basic kernel functionality
 */
class EchoKernel : public AgenticKernel {
public:
    EchoKernel(const std::string& kernel_id) 
        : AgenticKernel(KernelConfig(kernel_id, "echo_kernel")) {
    }

    ProcessingResult process(const CognitiveData& input) override {
        ProcessingResult result(input);
        
        // Add echo prefix to the symbolic tree
        if (input.has_symbolic()) {
            result.output_data.symbolic_tree = input.symbolic_tree;
            
            // Add "echo:" prefix to root
            if (!result.output_data.symbolic_tree.empty()) {
                auto root = result.output_data.symbolic_tree.begin();
                *root = "echo:" + *root;
            }
        }
        
        // Set metadata
        result.output_data.metadata = input.metadata;
        result.output_data.metadata["processed_by"] = get_kernel_id();
        result.output_data.data_id = "echo_" + input.data_id;
        
        // Simple cost/value model
        result.processing_cost = 2.0f;
        result.estimated_value = 3.0f;
        
        return result;
    }

    float estimate_processing_cost(const CognitiveData& input) const override {
        return 2.0f + (input.has_symbolic() ? input.symbolic_tree.size() * 0.1f : 0.0f);
    }

    float estimate_output_value(const CognitiveData& input) const override {
        return 3.0f + (input.has_symbolic() ? input.symbolic_tree.size() * 0.2f : 0.0f);
    }
};

/**
 * Grammar Analysis Kernel - analyzes token structure
 */
class GrammarAnalysisKernel : public AgenticKernel {
public:
    GrammarAnalysisKernel(const std::string& kernel_id)
        : AgenticKernel(KernelConfig(kernel_id, "grammar_analysis")) {
    }

    ProcessingResult process(const CognitiveData& input) override {
        ProcessingResult result(input);
        
        if (input.has_symbolic()) {
            // Analyze grammar structure
            size_t depth = analyze_tree_depth(input.symbolic_tree);
            size_t complexity = analyze_complexity(input.symbolic_tree);
            
            // Create analysis tree
            opencog::tree<std::string> analysis_tree;
            analysis_tree.set_head("grammar_analysis");
            auto root = analysis_tree.begin();
            
            analysis_tree.append_child(root, "depth:" + std::to_string(depth));
            analysis_tree.append_child(root, "complexity:" + std::to_string(complexity));
            analysis_tree.append_child(root, "tokens:" + std::to_string(input.symbolic_tree.size()));
            
            // Add original tree as subtree
            auto original_node = analysis_tree.append_child(root, "original");
            analysis_tree.append_child(original_node, input.symbolic_tree.begin());
            
            result.output_data.symbolic_tree = analysis_tree;
        }
        
        result.output_data.metadata = input.metadata;
        result.output_data.metadata["analysis_by"] = get_kernel_id();
        result.output_data.data_id = "analyzed_" + input.data_id;
        
        result.processing_cost = 5.0f;
        result.estimated_value = 8.0f;
        result.suggested_next_kernels.push_back("reasoning_kernel");
        
        return result;
    }

    float estimate_processing_cost(const CognitiveData& input) const override {
        return 5.0f + (input.has_symbolic() ? input.symbolic_tree.size() * 0.3f : 0.0f);
    }

    float estimate_output_value(const CognitiveData& input) const override {
        return 8.0f + (input.has_symbolic() ? input.symbolic_tree.size() * 0.4f : 0.0f);
    }

private:
    size_t analyze_tree_depth(const opencog::tree<std::string>& tree) const {
        if (tree.empty()) return 0;
        return tree.max_depth(tree.begin());
    }
    
    size_t analyze_complexity(const opencog::tree<std::string>& tree) const {
        if (tree.empty()) return 0;
        // Simple complexity measure: branching factor * depth
        return tree.max_branching(tree.begin()) * analyze_tree_depth(tree);
    }
};

/**
 * Demonstration of the Orchestral Architect system
 */
void demonstrate_agentic_system() {
    std::cout << "\n=== Orchestral Architect Demo ===" << std::endl;
    
    // 1. Create and register kernels
    std::cout << "\n1. Creating Agentic Kernels..." << std::endl;
    
    auto token_parser = TokenParsingAgentFactory::create_hybrid_parser("parser_001");
    auto echo_kernel = std::make_shared<EchoKernel>("echo_001");
    auto grammar_analyzer = std::make_shared<GrammarAnalysisKernel>("grammar_001");
    
    KernelRegistry& registry = KernelRegistry::instance();
    registry.register_kernel(token_parser);
    registry.register_kernel(echo_kernel);
    registry.register_kernel(grammar_analyzer);
    
    std::cout << "Registered kernels: " << registry.get_all_kernel_ids().size() << std::endl;
    
    // 2. Create attention allocation engine
    std::cout << "\n2. Initializing Attention Allocation Engine..." << std::endl;
    
    AttentionAllocationEngine::AllocationConfig attention_config;
    attention_config.total_attention_budget = 500.0f;
    auto attention_engine = std::make_shared<AttentionAllocationEngine>(attention_config);
    auto attention_processor = std::make_shared<AttentionAwareProcessor>(attention_engine);
    
    std::cout << "Attention budget: " << attention_engine->get_total_budget() << std::endl;
    
    // 3. Process sample data through the network
    std::cout << "\n3. Processing Cognitive Data..." << std::endl;
    
    // Create sample input
    CognitiveData input_data;
    input_data.data_id = "demo_input_001";
    input_data.metadata["raw_text"] = "(thinking (about (recursive (cognitive (networks))))";
    input_data.metadata["source"] = "demo";
    
    std::cout << "Input: " << input_data.metadata["raw_text"] << std::endl;
    
    // Stage 1: Token parsing
    std::cout << "\n--- Stage 1: Token Parsing ---" << std::endl;
    auto parse_result = attention_processor->process_with_attention(token_parser, input_data, 50.0f);
    
    std::cout << "Parsed tokens: " << parse_result.output_data.metadata["token_count"] << std::endl;
    std::cout << "Processing cost: " << parse_result.processing_cost << std::endl;
    std::cout << "Estimated value: " << parse_result.estimated_value << std::endl;
    
    if (parse_result.output_data.has_symbolic()) {
        std::cout << "Token tree: " << parse_result.output_data.symbolic_tree << std::endl;
    }
    
    // Stage 2: Echo processing
    std::cout << "\n--- Stage 2: Echo Processing ---" << std::endl;
    auto echo_result = attention_processor->process_with_attention(echo_kernel, parse_result.output_data, 30.0f);
    
    std::cout << "Echo result ID: " << echo_result.output_data.data_id << std::endl;
    if (echo_result.output_data.has_symbolic()) {
        std::cout << "Echo tree: " << echo_result.output_data.symbolic_tree << std::endl;
    }
    
    // Stage 3: Grammar analysis
    std::cout << "\n--- Stage 3: Grammar Analysis ---" << std::endl;
    auto analysis_result = attention_processor->process_with_attention(grammar_analyzer, parse_result.output_data, 80.0f);
    
    std::cout << "Analysis result ID: " << analysis_result.output_data.data_id << std::endl;
    if (analysis_result.output_data.has_symbolic()) {
        std::cout << "Analysis tree: " << analysis_result.output_data.symbolic_tree << std::endl;
    }
    
    // 4. Show attention allocation statistics
    std::cout << "\n4. Attention Allocation Statistics..." << std::endl;
    auto attention_stats = attention_engine->get_stats();
    std::cout << "Total allocated attention: " << attention_stats.total_allocated << std::endl;
    std::cout << "Processed requests: " << attention_stats.processed_requests << std::endl;
    std::cout << "Available attention: " << attention_engine->get_available_attention() << std::endl;
    
    // 5. Show kernel statistics
    std::cout << "\n5. Kernel Performance Statistics..." << std::endl;
    for (const auto& kernel_id : registry.get_all_kernel_ids()) {
        auto kernel = registry.get_kernel(kernel_id);
        if (kernel) {
            const auto& stats = kernel->get_stats();
            std::cout << "Kernel " << kernel_id << ":" << std::endl;
            std::cout << "  Total processed: " << stats.total_processed << std::endl;
            std::cout << "  Total cost: " << stats.total_cost << std::endl;
            std::cout << "  Total value: " << stats.total_value << std::endl;
            std::cout << "  Avg processing time: " << stats.average_processing_time << "s" << std::endl;
        }
    }
    
    std::cout << "\n=== Demo Complete ===" << std::endl;
}

/**
 * Show the system architecture
 */
void show_architecture() {
    std::cout << "\n=== Orchestral Architect: System Architecture ===" << std::endl;
    std::cout << "\nDistributed Agentic Cognitive Grammar Network:" << std::endl;
    std::cout << "┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐" << std::endl;
    std::cout << "│ Token Parser    │───▶│ Grammar Analyzer │───▶│ Reasoning Engine│" << std::endl;
    std::cout << "│ Agent           │    │ Kernel           │    │ (Future)        │" << std::endl;
    std::cout << "└─────────────────┘    └──────────────────┘    └─────────────────┘" << std::endl;
    std::cout << "        │                        │                        │" << std::endl;
    std::cout << "        ▼                        ▼                        ▼" << std::endl;
    std::cout << "┌─────────────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│              Attention Allocation Engine                         │" << std::endl;
    std::cout << "│  • Economic resource allocation                                  │" << std::endl;
    std::cout << "│  • ECAN-inspired attention spreading                            │" << std::endl;
    std::cout << "│  • Dynamic priority management                                  │" << std::endl;
    std::cout << "└─────────────────────────────────────────────────────────────────┘" << std::endl;
    std::cout << "        │                        │                        │" << std::endl;
    std::cout << "        ▼                        ▼                        ▼" << std::endl;
    std::cout << "┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐" << std::endl;
    std::cout << "│ Tensor Mapper   │    │ Hypergraph       │    │ Distributed     │" << std::endl;
    std::cout << "│ (GGML Bridge)   │    │ Encoder          │    │ Memory Store    │" << std::endl;
    std::cout << "└─────────────────┘    └──────────────────┘    └─────────────────┘" << std::endl;
}

int main() {
    // Initialize logging
    opencog::logger().set_level(opencog::Logger::INFO);
    opencog::logger().set_print_to_stdout_flag(true);
    
    try {
        show_architecture();
        demonstrate_agentic_system();
        
        std::cout << "\n=== Orchestral Architect Demo Successfully Completed ===" << std::endl;
        std::cout << "\nThis demonstrates the foundational components of the distributed" << std::endl;
        std::cout << "agentic cognitive grammar system. Each kernel processes cognitive" << std::endl;
        std::cout << "representations and participates in the attention economy." << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in demo: " << e.what() << std::endl;
        return 1;
    }
}