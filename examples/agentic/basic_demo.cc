/*
 * examples/agentic/basic_demo.cc
 *
 * Basic demonstration of the agentic kernel system without dependencies
 * on complex attention allocation - focuses on core functionality.
 */

#include <iostream>
#include <string>
#include <memory>

#include <opencog/agentic/kernel/AgenticKernel.h>
#include <opencog/agentic/grammar/TokenParsingAgent.h>
#include <opencog/util/Logger.h>

using namespace opencog::agentic;
using namespace opencog::agentic::grammar;

/**
 * Simple demonstration without attention allocation complexity
 */
void basic_kernel_demo() {
    std::cout << "\n=== Basic Agentic Kernel Demo ===" << std::endl;
    
    // 1. Create a token parsing agent
    std::cout << "\n1. Creating Token Parsing Agent..." << std::endl;
    auto parser = TokenParsingAgentFactory::create_simple_parser("basic_parser");
    
    // 2. Create sample input data
    std::cout << "\n2. Preparing input data..." << std::endl;
    CognitiveData input;
    input.data_id = "demo_001";
    input.metadata["raw_text"] = "hello world this is a simple test";
    input.metadata["source"] = "basic_demo";
    
    std::cout << "Input text: " << input.metadata["raw_text"] << std::endl;
    
    // 3. Process the data
    std::cout << "\n3. Processing with Token Parser..." << std::endl;
    ProcessingResult result = parser->process(input);
    
    // 4. Display results
    std::cout << "\n4. Results:" << std::endl;
    std::cout << "Output ID: " << result.output_data.data_id << std::endl;
    std::cout << "Processing cost: " << result.processing_cost << std::endl;
    std::cout << "Estimated value: " << result.estimated_value << std::endl;
    std::cout << "Token count: " << result.output_data.metadata["token_count"] << std::endl;
    
    if (result.output_data.has_symbolic()) {
        std::cout << "Parsed tree structure:" << std::endl;
        std::cout << result.output_data.symbolic_tree << std::endl;
    }
    
    // 5. Show attention weights
    std::cout << "\n5. Attention Weights:" << std::endl;
    for (auto it = result.output_data.symbolic_tree.begin(); 
         it != result.output_data.symbolic_tree.end(); ++it) {
        float weight = result.output_data.get_attention_weight(*it);
        if (weight > 0.0f) {
            std::cout << "  '" << *it << "' -> " << weight << std::endl;
        }
    }
    
    // 6. Test different tokenization strategies
    std::cout << "\n6. Testing Different Strategies..." << std::endl;
    
    // Grammar-based parsing
    auto grammar_parser = TokenParsingAgentFactory::create_grammar_parser("grammar_parser", {});
    input.metadata["raw_text"] = "(func arg1 (nested arg2) arg3)";
    std::cout << "Grammar input: " << input.metadata["raw_text"] << std::endl;
    
    auto grammar_result = grammar_parser->process(input);
    std::cout << "Grammar result tokens: " << grammar_result.output_data.metadata["token_count"] << std::endl;
    if (grammar_result.output_data.has_symbolic()) {
        std::cout << "Grammar tree: " << grammar_result.output_data.symbolic_tree << std::endl;
    }
    
    std::cout << "\n=== Basic Demo Complete ===" << std::endl;
}

/**
 * Demonstrate kernel registry functionality
 */
void registry_demo() {
    std::cout << "\n=== Kernel Registry Demo ===" << std::endl;
    
    // Create multiple kernels
    auto parser1 = TokenParsingAgentFactory::create_simple_parser("parser_001");
    auto parser2 = TokenParsingAgentFactory::create_hybrid_parser("parser_002");
    
    // Register them
    KernelRegistry& registry = KernelRegistry::instance();
    registry.register_kernel(parser1);
    registry.register_kernel(parser2);
    
    std::cout << "Registered kernels: " << registry.get_all_kernel_ids().size() << std::endl;
    for (const auto& id : registry.get_all_kernel_ids()) {
        auto kernel = registry.get_kernel(id);
        std::cout << "  - " << id << " (type: " << kernel->get_kernel_type() << ")" << std::endl;
    }
    
    // Test kernel lookup by type
    auto parsers = registry.get_kernels_by_type("token_parser");
    std::cout << "Found " << parsers.size() << " token parsers" << std::endl;
    
    std::cout << "=== Registry Demo Complete ===" << std::endl;
}

/**
 * Show the architecture in ASCII art
 */
void show_ascii_architecture() {
    std::cout << "\n=== Orchestral Architect ASCII Architecture ===" << std::endl;
    std::cout << R"(
╔══════════════════════════════════════════════════════════════════╗
║                    DISTRIBUTED AGENTIC NETWORK                   ║
╚══════════════════════════════════════════════════════════════════╝

    Input Stream                 Cognitive Processing Pipeline
         │                              │
         ▼                              ▼
    ┌─────────┐    ┌─────────────┐    ┌──────────────┐    ┌────────────┐
    │ Raw     │───▶│ Token       │───▶│ Grammar      │───▶│ Reasoning  │
    │ Data    │    │ Parsing     │    │ Analysis     │    │ Engine     │
    │         │    │ Agent       │    │ Kernel       │    │ (Future)   │
    └─────────┘    └─────────────┘    └──────────────┘    └────────────┘
                          │                   │                  │
                          ▼                   ▼                  ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │              ATTENTION ALLOCATION ENGINE                         │
    │  ◆ Economic resource management                                  │
    │  ◆ Priority-based processing                                     │
    │  ◆ Performance feedback loops                                    │
    │  ◆ Dynamic kernel adaptation                                     │
    └─────────────────────────────────────────────────────────────────┘
                          │                   │                  │
                          ▼                   ▼                  ▼
    ┌─────────────┐    ┌──────────────┐    ┌─────────────────────────┐
    │ Tensor      │    │ Hypergraph   │    │ Distributed Memory      │
    │ Mapper      │    │ Encoder      │    │ Store                   │
    │ (GGML)      │    │              │    │                         │
    └─────────────┘    └──────────────┘    └─────────────────────────┘

Key Features:
• Neural-Symbolic Integration: Seamless tree ↔ tensor conversion
• Economic Attention: Resource allocation based on cost/value analysis  
• Event-Driven Communication: Asynchronous kernel interaction
• Recursive Architecture: Self-reflecting and adaptive system
• Modular Design: Each kernel is independently deployable
)" << std::endl;
}

int main() {
    // Initialize logging
    opencog::logger().set_level(opencog::Logger::INFO);
    opencog::logger().set_print_to_stdout_flag(true);
    
    try {
        show_ascii_architecture();
        basic_kernel_demo();
        registry_demo();
        
        std::cout << "\n╔══════════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout <<   "║  ORCHESTRAL ARCHITECT: Foundation Successfully Demonstrated     ║" << std::endl;
        std::cout <<   "╚══════════════════════════════════════════════════════════════════╝" << std::endl;
        std::cout << "\nThis demonstrates the core components of the distributed agentic" << std::endl;
        std::cout << "cognitive grammar system. Each kernel processes symbolic and tensor" << std::endl;
        std::cout << "representations while participating in the attention economy." << std::endl;
        std::cout << "\nNext steps: Integrate with AtomSpace, PLN reasoning, and MOSES learning." << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Demo error: " << e.what() << std::endl;
        return 1;
    }
}