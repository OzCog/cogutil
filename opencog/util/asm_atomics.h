/*
 * opencog/util/asm_atomics.h
 *
 * High-performance assembly-optimized atomic operations
 *
 * Copyright (C) 2024 OpenCog Foundation
 * All Rights Reserved
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

#ifndef _OPENCOG_ASM_ATOMICS_H
#define _OPENCOG_ASM_ATOMICS_H

#include <atomic>
#include <cstdint>

namespace opencog {
namespace asm_atomics {

/** \addtogroup grp_cogutil
 *  @{
 */

#ifdef __x86_64__

/**
 * High-performance atomic operations using assembly for x86_64
 * These provide better performance than std::atomic in many cases
 * by using specific CPU instructions and memory orderings
 */

/// Fast atomic increment with acquire-release semantics
template<typename T>
inline T fast_fetch_add(std::atomic<T>* ptr, T value) {
    static_assert(sizeof(T) <= 8, "Type too large for atomic operations");
    
    if constexpr (sizeof(T) == 8) {
        T result;
        __asm__ volatile (
            "lock xaddq %0, %1"
            : "=r" (result), "+m" (*ptr)
            : "0" (value)
            : "memory", "cc"
        );
        return result;
    } else if constexpr (sizeof(T) == 4) {
        T result;
        __asm__ volatile (
            "lock xaddl %0, %1"
            : "=r" (result), "+m" (*ptr)
            : "0" (value)
            : "memory", "cc"
        );
        return result;
    } else {
        return ptr->fetch_add(value, std::memory_order_acq_rel);
    }
}

/// Fast atomic increment
template<typename T>
inline T fast_fetch_increment(std::atomic<T>* ptr) {
    return fast_fetch_add(ptr, T(1));
}

/// Fast atomic decrement
template<typename T>
inline T fast_fetch_decrement(std::atomic<T>* ptr) {
    return fast_fetch_add(ptr, T(-1));
}

/// High-performance atomic load with acquire semantics
template<typename T>
inline T fast_load_acquire(const std::atomic<T>* ptr) {
    static_assert(sizeof(T) <= 8, "Type too large for atomic operations");
    
    if constexpr (sizeof(T) == 8) {
        T result;
        __asm__ volatile (
            "movq %1, %0"
            : "=r" (result)
            : "m" (*ptr)
            : "memory"
        );
        return result;
    } else if constexpr (sizeof(T) == 4) {
        T result;
        __asm__ volatile (
            "movl %1, %0"
            : "=r" (result)
            : "m" (*ptr)
            : "memory"
        );
        return result;
    } else {
        return ptr->load(std::memory_order_acquire);
    }
}

/// High-performance atomic store with release semantics
template<typename T>
inline void fast_store_release(std::atomic<T>* ptr, T value) {
    static_assert(sizeof(T) <= 8, "Type too large for atomic operations");
    
    if constexpr (sizeof(T) == 8) {
        __asm__ volatile (
            "movq %1, %0"
            : "=m" (*ptr)
            : "r" (value)
            : "memory"
        );
    } else if constexpr (sizeof(T) == 4) {
        __asm__ volatile (
            "movl %1, %0"
            : "=m" (*ptr)
            : "r" (value)
            : "memory"
        );
    } else {
        ptr->store(value, std::memory_order_release);
    }
}

/// High-performance compare-and-swap with acquire-release semantics
template<typename T>
inline bool fast_compare_exchange_weak(std::atomic<T>* ptr, T& expected, T desired) {
    static_assert(sizeof(T) <= 8, "Type too large for atomic operations");
    
    if constexpr (sizeof(T) == 8) {
        T prev;
        bool success;
        __asm__ volatile (
            "lock cmpxchgq %3, %1\n\t"
            "sete %0"
            : "=q" (success), "+m" (*ptr), "=a" (prev)
            : "r" (desired), "2" (expected)
            : "memory", "cc"
        );
        expected = prev;
        return success;
    } else if constexpr (sizeof(T) == 4) {
        T prev;
        bool success;
        __asm__ volatile (
            "lock cmpxchgl %3, %1\n\t"
            "sete %0"
            : "=q" (success), "+m" (*ptr), "=a" (prev)
            : "r" (desired), "2" (expected)
            : "memory", "cc"
        );
        expected = prev;
        return success;
    } else {
        return ptr->compare_exchange_weak(expected, desired, std::memory_order_acq_rel);
    }
}

/// Optimized atomic counter with performance tracking
class fast_atomic_counter {
private:
    alignas(64) std::atomic<uint64_t> _count;  // Cache line aligned
    
public:
    fast_atomic_counter(uint64_t initial = 0) : _count(initial) {}
    
    /// Increment and return previous value
    uint64_t fetch_increment() {
        return fast_fetch_increment(&_count);
    }
    
    /// Increment and return new value
    uint64_t increment() {
        return fast_fetch_increment(&_count) + 1;
    }
    
    /// Decrement and return previous value
    uint64_t fetch_decrement() {
        return fast_fetch_decrement(&_count);
    }
    
    /// Decrement and return new value
    uint64_t decrement() {
        return fast_fetch_decrement(&_count) - 1;
    }
    
    /// Add value and return previous value
    uint64_t fetch_add(uint64_t value) {
        return fast_fetch_add(&_count, value);
    }
    
    /// Load current value
    uint64_t load() const {
        return fast_load_acquire(&_count);
    }
    
    /// Store new value
    void store(uint64_t value) {
        fast_store_release(&_count, value);
    }
    
    /// Reset to zero
    void reset() {
        store(0);
    }
    
    /// Conversion operator for compatibility
    operator uint64_t() const {
        return load();
    }
};

#else

// Fallback implementations for non-x86_64 architectures
template<typename T>
inline T fast_fetch_add(std::atomic<T>* ptr, T value) {
    return ptr->fetch_add(value, std::memory_order_acq_rel);
}

template<typename T>
inline T fast_fetch_increment(std::atomic<T>* ptr) {
    return ptr->fetch_add(T(1), std::memory_order_acq_rel);
}

template<typename T>
inline T fast_fetch_decrement(std::atomic<T>* ptr) {
    return ptr->fetch_sub(T(1), std::memory_order_acq_rel);
}

template<typename T>
inline T fast_load_acquire(const std::atomic<T>* ptr) {
    return ptr->load(std::memory_order_acquire);
}

template<typename T>
inline void fast_store_release(std::atomic<T>* ptr, T value) {
    ptr->store(value, std::memory_order_release);
}

template<typename T>
inline bool fast_compare_exchange_weak(std::atomic<T>* ptr, T& expected, T desired) {
    return ptr->compare_exchange_weak(expected, desired, std::memory_order_acq_rel);
}

class fast_atomic_counter {
private:
    std::atomic<uint64_t> _count;
    
public:
    fast_atomic_counter(uint64_t initial = 0) : _count(initial) {}
    
    uint64_t fetch_increment() {
        return _count.fetch_add(1, std::memory_order_acq_rel);
    }
    
    uint64_t increment() {
        return _count.fetch_add(1, std::memory_order_acq_rel) + 1;
    }
    
    uint64_t fetch_decrement() {
        return _count.fetch_sub(1, std::memory_order_acq_rel);
    }
    
    uint64_t decrement() {
        return _count.fetch_sub(1, std::memory_order_acq_rel) - 1;
    }
    
    uint64_t fetch_add(uint64_t value) {
        return _count.fetch_add(value, std::memory_order_acq_rel);
    }
    
    uint64_t load() const {
        return _count.load(std::memory_order_acquire);
    }
    
    void store(uint64_t value) {
        _count.store(value, std::memory_order_release);
    }
    
    void reset() {
        store(0);
    }
    
    operator uint64_t() const {
        return load();
    }
};

#endif

/**
 * Lock-free atomic ring buffer for high-performance single-producer-single-consumer scenarios
 */
template<typename T, size_t N>
class lockfree_ring_buffer {
private:
    static_assert((N & (N - 1)) == 0, "Size must be power of 2");
    static constexpr size_t MASK = N - 1;
    
    alignas(64) std::atomic<size_t> _head{0};  // Cache line aligned
    alignas(64) std::atomic<size_t> _tail{0};  // Cache line aligned
    alignas(64) T _buffer[N];                  // Cache line aligned
    
public:
    /// Try to push an element (returns false if full)
    bool push(const T& item) {
        const size_t current_head = fast_load_acquire(&_head);
        const size_t next_head = (current_head + 1) & MASK;
        
        if (next_head == fast_load_acquire(&_tail)) {
            return false;  // Buffer is full
        }
        
        _buffer[current_head] = item;
        fast_store_release(&_head, next_head);
        return true;
    }
    
    /// Try to pop an element (returns false if empty)
    bool pop(T& item) {
        const size_t current_tail = fast_load_acquire(&_tail);
        
        if (current_tail == fast_load_acquire(&_head)) {
            return false;  // Buffer is empty
        }
        
        item = _buffer[current_tail];
        fast_store_release(&_tail, (current_tail + 1) & MASK);
        return true;
    }
    
    /// Check if buffer is empty
    bool empty() const {
        return fast_load_acquire(&_head) == fast_load_acquire(&_tail);
    }
    
    /// Check if buffer is full
    bool full() const {
        const size_t head = fast_load_acquire(&_head);
        const size_t tail = fast_load_acquire(&_tail);
        return ((head + 1) & MASK) == tail;
    }
    
    /// Get current size (approximate)
    size_t size() const {
        const size_t head = fast_load_acquire(&_head);
        const size_t tail = fast_load_acquire(&_tail);
        return (head - tail) & MASK;
    }
    
    /// Get capacity
    constexpr size_t capacity() const {
        return N - 1;  // One slot is reserved to distinguish full from empty
    }
};

/** @}*/

} // namespace asm_atomics
} // namespace opencog

#endif // _OPENCOG_ASM_ATOMICS_H