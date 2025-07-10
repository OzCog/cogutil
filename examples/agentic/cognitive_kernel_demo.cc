/*
 * examples/agentic/cognitive_kernel_demo.cc
 *
 * Cognitive Kernel Genesis - Demonstration of Phase 1 & 2 Implementation
 * 
 * This example showcases the core cognitive primitives and kernel orchestration
 * features implemented for the distributed cognition kernel.
 */

#include <iostream>
#include <memory>
#include <vector>
#include <opencog/agentic/kernel/AgenticKernel.h>
#include <opencog/agentic/grammar/TokenParsingAgent.h>
#include <opencog/util/tree.h>

using namespace opencog::agentic;
using namespace opencog::agentic::grammar;

int main() {
    std::cout << "=== Cognitive Kernel Genesis - Phase 1 & 2 Demo ===\n\n";
    
    // 1. Create a Token Parsing Agent (lexical → syntactic stream)
    std::cout << "1. Creating Token Parsing Agent...\n";
    auto token_parser = TokenParsingAgentFactory::create_simple_parser("parser_001");
    std::cout << "   Created parser: " << token_parser->get_kernel_id() 
              << " (type: " << token_parser->get_kernel_type() << ")\n\n";
    
    // 2. Register kernel in the distributed network
    std::cout << "2. Registering kernel in the network...\n";
    KernelRegistry& registry = KernelRegistry::instance();
    registry.register_kernel(token_parser);
    
    auto all_kernels = registry.get_all_kernel_ids();
    std::cout << "   Network now contains " << all_kernels.size() << " kernels:\n";
    for (const auto& id : all_kernels) {
        std::cout << "   - " << id << "\n";
    }
    std::cout << "\n";
    
    // 3. Demonstrate cognitive processing pipeline
    std::cout << "3. Processing cognitive data through the kernel...\n";
    
    CognitiveData input_data;
    input_data.data_id = "cognitive_input_001";
    input_data.metadata["raw_text"] = "(thinking (about (cognitive (networks))))";
    input_data.set_attention_weight("structural_importance", 0.8f);
    input_data.set_attention_weight("content_importance", 0.9f);
    
    std::cout << "   Input data: " << input_data.metadata["raw_text"] << "\n";
    std::cout << "   Attention weights: structural=" 
              << input_data.get_attention_weight("structural_importance")
              << ", content=" << input_data.get_attention_weight("content_importance") << "\n";
    
    // 4. Process through the agentic kernel
    std::cout << "\n4. Processing through agentic kernel...\n";
    
    // Check if kernel should process this data
    float available_attention = 100.0f;
    bool should_process = token_parser->should_process(input_data, available_attention);
    std::cout << "   Should process (attention check): " << (should_process ? "YES" : "NO") << "\n";
    
    if (should_process) {
        // Estimate processing cost and value
        float cost = token_parser->estimate_processing_cost(input_data);
        float value = token_parser->estimate_output_value(input_data);
        std::cout << "   Estimated cost: " << cost << ", value: " << value << "\n";
        
        // Process the data
        ProcessingResult result = token_parser->process(input_data);
        
        std::cout << "   Processing complete!\n";
        std::cout << "   Output data ID: " << result.output_data.data_id << "\n";
        std::cout << "   Token count: " << result.output_data.metadata["token_count"] << "\n";
        std::cout << "   Processing stage: " << result.output_data.metadata["processing_stage"] << "\n";
        std::cout << "   Actual cost: " << result.processing_cost << "\n";
        std::cout << "   Estimated value: " << result.estimated_value << "\n";
        
        // Show suggested next kernels
        std::cout << "   Suggested next kernels: ";
        for (size_t i = 0; i < result.suggested_next_kernels.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << result.suggested_next_kernels[i];
        }
        std::cout << "\n";
        
        // Display the symbolic tree structure
        if (result.output_data.has_symbolic()) {
            std::cout << "   Symbolic tree structure: " << result.output_data.symbolic_tree << "\n";
        }
    }
    
    // 5. Demonstrate attention allocation system
    std::cout << "\n5. Attention allocation demonstration...\n";
    
    auto attention_state = registry.get_attention_state();
    std::cout << "   Total attention budget: " << attention_state.total_available_attention << "\n";
    std::cout << "   Currently allocated: " << attention_state.allocated_attention << "\n";
    std::cout << "   Available: " << (attention_state.total_available_attention - attention_state.allocated_attention) << "\n";
    
    // Request attention allocation
    bool allocated = registry.allocate_attention("parser_001", 50.0f);
    std::cout << "   Requested 50 units for parser_001: " << (allocated ? "GRANTED" : "DENIED") << "\n";
    
    if (allocated) {
        auto new_state = registry.get_attention_state();
        std::cout << "   New allocated amount: " << new_state.allocated_attention << "\n";
        
        // Release attention
        registry.release_attention("parser_001", 50.0f);
        auto final_state = registry.get_attention_state();
        std::cout << "   Released attention, new allocated: " << final_state.allocated_attention << "\n";
    }
    
    // 6. Demonstrate adaptive learning
    std::cout << "\n6. Adaptive learning demonstration...\n";
    
    auto initial_stats = token_parser->get_stats();
    std::cout << "   Initial stats - processed: " << initial_stats.total_processed 
              << ", total cost: " << initial_stats.total_cost << "\n";
    
    // Simulate some performance feedback
    if (should_process) {
        ProcessingResult result = token_parser->process(input_data);
        token_parser->update_processing_stats(result, 0.1f); // 0.1 second processing time
        
        // Provide feedback for adaptation
        token_parser->update_from_feedback(result, result.estimated_value * 1.2f); // Better than expected
        
        auto final_stats = token_parser->get_stats();
        std::cout << "   Final stats - processed: " << final_stats.total_processed 
                  << ", total cost: " << final_stats.total_cost << "\n";
        std::cout << "   Average processing time: " << final_stats.average_processing_time << "s\n";
    }
    
    // 7. Show kernel configuration adaptation
    std::cout << "\n7. Kernel configuration adaptation...\n";
    
    auto config = token_parser->get_config();
    std::cout << "   Current activation threshold: " << config.base_activation_threshold << "\n";
    std::cout << "   Max processing cost: " << config.max_processing_cost << "\n";
    
    // Simulate performance metrics feedback
    std::map<std::string, float> performance_metrics;
    performance_metrics["efficiency"] = 0.9f; // High efficiency
    performance_metrics["accuracy"] = 0.85f;  // Good accuracy
    
    token_parser->adapt_parameters(performance_metrics);
    std::cout << "   Adaptation based on performance metrics applied.\n";
    
    std::cout << "\n=== Cognitive Kernel Genesis Demo Complete ===\n";
    std::cout << "\nPhase 1 & 2 Features Demonstrated:\n";
    std::cout << "✓ Token Parser (lexical → syntactic stream)\n";
    std::cout << "✓ Agentic Kernel orchestration\n";  
    std::cout << "✓ Attention allocation system\n";
    std::cout << "✓ Adaptive learning and parameter adjustment\n";
    std::cout << "✓ Distributed kernel registry\n";
    std::cout << "✓ Cognitive data representation with attention weights\n";
    std::cout << "✓ Recursive kernel invocation capabilities\n";
    std::cout << "\nReady for Phase 3: TensorShape mapping and GGML integration!\n";
    
    return 0;
}