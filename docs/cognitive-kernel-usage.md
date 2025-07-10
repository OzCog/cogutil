# Cognitive Kernel Genesis - Usage Guide

## Overview

The Cognitive Kernel Genesis implementation provides a distributed cognition framework where each component is an "Agentic Grammar Kernel" that processes cognitive representations (tensors, symbols, hypergraphs) within a dynamic, extensible network.

## Quick Start

### Building the System

```bash
cd cogutil
mkdir build && cd build
cmake ..
make agentic-kernel agentic-grammar
```

### Running the Demo

```bash
cd build/examples/agentic
./cognitive_kernel_demo
```

## Core Components

### 1. Agentic Kernels

Base class for all processing nodes in the cognitive network:

```cpp
#include <opencog/agentic/kernel/AgenticKernel.h>

// Create kernel configuration
KernelConfig config("my_kernel", "custom_processor");
config.base_activation_threshold = 0.7f;

// Implement custom kernel
class MyKernel : public AgenticKernel {
public:
    MyKernel(const KernelConfig& config) : AgenticKernel(config) {}
    
    ProcessingResult process(const CognitiveData& input) override {
        // Your processing logic here
        ProcessingResult result;
        result.output_data = input; // Transform as needed
        result.processing_cost = 1.0f;
        result.estimated_value = 2.0f;
        return result;
    }
    
    float estimate_processing_cost(const CognitiveData& input) const override {
        return static_cast<float>(input.symbolic_tree.size()) * 0.1f;
    }
    
    float estimate_output_value(const CognitiveData& input) const override {
        return 1.0f; // Your value estimation logic
    }
};
```

### 2. Token Parsing Agent

Converts raw input into structured symbolic representations:

```cpp
#include <opencog/agentic/grammar/TokenParsingAgent.h>

// Create different types of parsers
auto simple_parser = TokenParsingAgentFactory::create_simple_parser("parser_001");
auto grammar_parser = TokenParsingAgentFactory::create_grammar_parser("parser_002", {});
auto hybrid_parser = TokenParsingAgentFactory::create_hybrid_parser("parser_003");

// Process input
CognitiveData input;
input.metadata["raw_text"] = "(thinking (about (cognitive (networks))))";
ProcessingResult result = parser->process(input);
```

### 3. Kernel Registry

Network-wide kernel management:

```cpp
KernelRegistry& registry = KernelRegistry::instance();

// Register kernels
registry.register_kernel(my_kernel);

// Route data between kernels
registry.route_data(cognitive_data, "target_kernel_id");

// Get kernels by type
auto parsers = registry.get_kernels_by_type("token_parser");
```

### 4. Attention Allocation

Economic resource management:

```cpp
#include <opencog/agentic/attention/AttentionAllocationEngine.h>

AttentionAllocationEngine attention_engine;

// Request attention for processing
AttentionRequest request("kernel_id", cognitive_data, 50.0f);
bool allocated = attention_engine.request_attention(request);

// Release attention after processing
attention_engine.release_attention("kernel_id", "request_id", actual_cost, actual_value);
```

### 5. Cognitive Data Structure

Unified representation for all data:

```cpp
#include <opencog/agentic/kernel/AgenticKernel.h>

CognitiveData data;
data.data_id = "unique_identifier";
data.metadata["raw_text"] = "input text";

// Set attention weights
data.set_attention_weight("importance", 0.8f);
data.set_attention_weight("urgency", 0.6f);

// Create symbolic tree
tree<std::string> symbolic_tree;
symbolic_tree.set_head("root");
// ... build tree structure
data.symbolic_tree = symbolic_tree;
```

## Advanced Features

### TensorShape Primitives

For neural-symbolic integration:

```cpp
#include <opencog/agentic/kernel/TensorShape.h>

// Create tensor shapes
TensorShape scalar = TensorShape::create_scalar();
TensorShape vector = TensorShape::create_vector(128);
TensorShape matrix = TensorShape::create_matrix(64, 128);

// Analyze degrees of freedom
size_t dof = shape.calculate_degrees_of_freedom();
auto factors = shape.get_prime_factorization();

// GGML compatibility
bool ggml_compatible = shape.is_ggml_compatible();
size_t memory_size = shape.estimate_ggml_memory_size();
```

### Link Primitives for Hypergraphs

Represent complex relationships:

```cpp
#include <opencog/agentic/kernel/Link.h>

// Create different types of links
auto inheritance = Link::create_inheritance_link("child_node", "parent_node");
auto similarity = Link::create_similarity_link("node1", "node2", 0.8f);
auto evaluation = Link::create_evaluation_link("predicate", {"arg1", "arg2"});

// Set attention weights
link->set_attention_weight(0.7f);
link->set_strength(0.9f);

// Pattern matching
bool matches = link1->matches_pattern(*link2);
float similarity = link1->calculate_pattern_similarity(*link2);
```

### Event-Driven Communication

Asynchronous kernel coordination:

```cpp
// Register for events
kernel->register_callback("data_ready", [](const CognitiveData& data) {
    // Handle incoming data
    std::cout << "Received data: " << data.data_id << std::endl;
});

// Emit events
kernel->emit_event("processing_complete", output_data);
```

### Adaptive Learning

Kernels adapt based on performance:

```cpp
// Update based on actual performance
kernel->update_from_feedback(result, actual_value);

// Adapt parameters based on metrics
std::map<std::string, float> metrics;
metrics["efficiency"] = 0.85f;
metrics["accuracy"] = 0.92f;
kernel->adapt_parameters(metrics);
```

## Integration Patterns

### Processing Pipeline

```cpp
// Create processing pipeline
auto tokenizer = TokenParsingAgentFactory::create_grammar_parser("tokenizer", {});
auto analyzer = std::make_shared<MyAnalyzer>(config);
auto mapper = TensorMapperFactory::create_ggml_optimized_mapper("mapper");

// Register all kernels
KernelRegistry& registry = KernelRegistry::instance();
registry.register_kernel(tokenizer);
registry.register_kernel(analyzer);
registry.register_kernel(mapper);

// Process data through pipeline
CognitiveData input;
input.metadata["raw_text"] = "complex cognitive expression";

auto result1 = tokenizer->process(input);
auto result2 = analyzer->process(result1.output_data);
auto result3 = mapper->process(result2.output_data);
```

### Attention-Aware Processing

```cpp
AttentionAllocationEngine attention_engine;
AttentionAwareProcessor processor(std::make_shared<AttentionAllocationEngine>(attention_engine));

// Process with automatic attention management
auto result = processor.process_with_attention(kernel, input_data, 50.0f);

// Batch processing with attention
std::vector<CognitiveData> batch_inputs = {...};
auto batch_results = processor.process_batch(kernel, batch_inputs);
```

## Performance Optimization

### Memory Efficiency

- Use GGML-optimized tensor shapes for large datasets
- Enable attention-based pruning for irrelevant data
- Cache frequently used mappings in TensorMapper

### Processing Efficiency

- Adjust kernel activation thresholds based on workload
- Use attention allocation to prioritize important data
- Enable adaptive learning for automatic parameter tuning

### Network Efficiency

- Group related kernels for locality
- Use event-driven communication to minimize overhead
- Monitor attention allocation statistics for bottlenecks

## Future Extensions

The system is designed for extensibility:

1. **New Kernel Types**: Implement custom AgenticKernel subclasses
2. **Attention Strategies**: Create custom AttentionAllocationEngine implementations
3. **Integration Modules**: Add connections to external systems (ROS, databases, APIs)
4. **GGML Operations**: Implement actual tensor operations using GGML
5. **AtomSpace Integration**: Connect to OpenCog's knowledge representation

## Troubleshooting

### Build Issues

- Ensure Boost libraries are installed: `apt-get install libboost-all-dev`
- Check CMake version: requires 3.12+
- For GGML integration, ensure headers are in include path

### Runtime Issues

- Check attention allocation if kernels aren't processing
- Verify kernel registration in registry
- Monitor attention weights for data prioritization
- Enable debug logging for detailed processing information

## Contributing

Key areas for contribution:

1. **New Kernel Implementations**: Domain-specific processors
2. **Attention Algorithms**: Advanced resource allocation strategies
3. **Integration Modules**: External system connections
4. **Performance Optimization**: GGML tensor operations
5. **Testing**: Comprehensive test coverage

The architecture enables emergent synergy where every component contributes to the collective cognitive capability of the system.