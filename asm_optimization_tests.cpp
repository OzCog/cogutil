/*
 * Unit tests for assembly optimizations in cogutil
 * 
 * This tests the correctness of assembly-optimized functions
 * to ensure they produce identical results to standard implementations
 */

#include <iostream>
#include <cassert>
#include <vector>
#include <random>
#include <cstring>
#include <set>

#include <opencog/util/asm_optimizations.h>
#include <opencog/util/asm_atomics.h>
#include <opencog/util/numeric.h>

using namespace opencog;
using namespace opencog::asm_opt;
using namespace opencog::asm_atomics;

class AssemblyOptimizationTests {
public:
    void testIntegerLog2() {
        std::cout << "Testing integer_log2 optimizations..." << std::endl;
        
        // Test known values
        assert(integer_log2(1) == 0);
        assert(integer_log2(2) == 1);
        assert(integer_log2(4) == 2);
        assert(integer_log2(8) == 3);
        assert(integer_log2(16) == 4);
        assert(integer_log2(1024) == 10);
        assert(integer_log2(1UL << 32) == 32);
        
        // Test various powers of 2
        for (int i = 0; i < 63; ++i) {
            size_t val = 1UL << i;
            assert(integer_log2(val) == i);
        }
        
        // Test edge cases
        assert(integer_log2(0) == 0);
        assert(integer_log2(3) == 1);  // Should give floor(log2(3)) = 1
        assert(integer_log2(7) == 2);  // Should give floor(log2(7)) = 2
        
        std::cout << "✓ integer_log2 tests passed" << std::endl;
    }
    
    void testNextPowerOfTwo() {
        std::cout << "Testing next_power_of_two optimizations..." << std::endl;
        
        // Test known values
        assert(next_power_of_two(1) == 1);
        assert(next_power_of_two(2) == 2);
        assert(next_power_of_two(3) == 4);
        assert(next_power_of_two(4) == 4);
        assert(next_power_of_two(5) == 8);
        assert(next_power_of_two(8) == 8);
        assert(next_power_of_two(9) == 16);
        assert(next_power_of_two(1023) == 1024);
        assert(next_power_of_two(1024) == 1024);
        assert(next_power_of_two(1025) == 2048);
        
        // Test that results are always powers of 2
        std::mt19937 gen(42);
        std::uniform_int_distribution<size_t> dis(1, 1UL << 30);
        
        for (int i = 0; i < 1000; ++i) {
            size_t val = dis(gen);
            size_t result = next_power_of_two(val);
            
            // Verify result is power of 2
            assert((result & (result - 1)) == 0);
            
            // Verify result >= val
            assert(result >= val);
            
            // Verify result/2 < val (unless val is already power of 2)
            if (val > 1 && (val & (val - 1)) != 0) {
                assert(result / 2 < val);
            }
        }
        
        std::cout << "✓ next_power_of_two tests passed" << std::endl;
    }
    
    void testHashFunctions() {
        std::cout << "Testing hash function optimizations..." << std::endl;
        
        // Test hash functions produce reasonable distribution
        std::vector<uint64_t> hashes;
        for (uint64_t i = 0; i < 10000; ++i) {
            hashes.push_back(fast_hash64(i));
        }
        
        // Check for some basic properties:
        // 1. No obvious patterns in low bits
        size_t even_count = 0, odd_count = 0;
        for (uint64_t hash : hashes) {
            if (hash & 1) odd_count++;
            else even_count++;
        }
        
        // Should be roughly 50/50 split
        assert(even_count > 4000 && even_count < 6000);
        assert(odd_count > 4000 && odd_count < 6000);
        
        // 2. Different inputs should produce different outputs (mostly)
        std::set<uint64_t> unique_hashes(hashes.begin(), hashes.end());
        assert(unique_hashes.size() > 9900);  // Allow for some collisions
        
        // Test string hashing
        std::vector<std::string> test_strings = {
            "hello", "world", "opencog", "assembly", "optimization",
            "performance", "hash", "function", "test", "benchmark"
        };
        
        std::set<uint64_t> string_hashes;
        for (const auto& str : test_strings) {
            uint64_t hash = fast_string_hash(str.c_str(), str.length());
            string_hashes.insert(hash);
        }
        
        // All strings should hash to different values
        assert(string_hashes.size() == test_strings.size());
        
        std::cout << "✓ hash function tests passed" << std::endl;
    }
    
    void testAtomicOperations() {
        std::cout << "Testing atomic operation optimizations..." << std::endl;
        
        fast_atomic_counter counter(0);
        
        // Test basic increment/decrement
        assert(counter.load() == 0);
        assert(counter.increment() == 1);
        assert(counter.load() == 1);
        assert(counter.fetch_increment() == 1);
        assert(counter.load() == 2);
        
        assert(counter.decrement() == 1);
        assert(counter.load() == 1);
        assert(counter.fetch_decrement() == 1);
        assert(counter.load() == 0);
        
        // Test atomic add
        assert(counter.fetch_add(100) == 0);
        assert(counter.load() == 100);
        
        // Test store/reset
        counter.store(42);
        assert(counter.load() == 42);
        
        counter.reset();
        assert(counter.load() == 0);
        
        std::cout << "✓ atomic operation tests passed" << std::endl;
    }
    
    void testLockFreeBuffer() {
        std::cout << "Testing lock-free ring buffer..." << std::endl;
        
        lockfree_ring_buffer<int, 8> buffer;
        
        // Test basic operations
        assert(buffer.empty());
        assert(!buffer.full());
        assert(buffer.size() == 0);
        
        // Test push/pop
        assert(buffer.push(1));
        assert(buffer.push(2));
        assert(buffer.push(3));
        
        assert(!buffer.empty());
        assert(buffer.size() == 3);
        
        int value;
        assert(buffer.pop(value));
        assert(value == 1);
        
        assert(buffer.pop(value));
        assert(value == 2);
        
        assert(buffer.size() == 1);
        
        // Fill buffer to capacity
        for (int i = 10; i < 16; ++i) {  // Buffer capacity is 7 (8-1), we already have 1 item
            bool pushed = buffer.push(i);
            if (!pushed) break;  // Stop when buffer is full
        }
        
        assert(buffer.full());
        assert(!buffer.push(99));  // Should fail when full
        
        // Empty the buffer
        while (!buffer.empty()) {
            assert(buffer.pop(value));
        }
        
        assert(buffer.empty());
        assert(!buffer.pop(value));  // Should fail when empty
        
        std::cout << "✓ lock-free ring buffer tests passed" << std::endl;
    }
    
    void runAllTests() {
        std::cout << "Running Assembly Optimization Correctness Tests" << std::endl;
        std::cout << "===============================================" << std::endl;
        
        testIntegerLog2();
        testNextPowerOfTwo();
        testHashFunctions();
        testAtomicOperations();
        testLockFreeBuffer();
        
        std::cout << "\n🎉 All tests passed! Assembly optimizations are working correctly." << std::endl;
    }
};

int main() {
    AssemblyOptimizationTests tests;
    tests.runAllTests();
    return 0;
}