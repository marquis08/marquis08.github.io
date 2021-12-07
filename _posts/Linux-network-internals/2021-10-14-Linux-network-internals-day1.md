---
date: 2021-10-14 02:47
title: "Understanding Linux network internals - day 1"
categories: Linux Network
tags: Linux Network
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

> This study is based on the book *Understanding Linux network internals*.

# Basic Terminology
- Eight-bit: *octets*, *byte*
- vector = array
- layers of TCP/IP network stack
  - L2: link layer, Ethernet
  - L3: network layer, IP version 4 or 6
  - L4: transport layer, UDP, TCP, or ICMP
    - numbers are based on 7-layer OSI model
- ingress = input
- egress = output
- receiving a data unit = RX
- transmitting a data unit = TX
- data unit = frame, packet, segment, and message depending on the layer whre it is used(Chapter 13 for details)
- BH: Bottom half
- IRQ: Interrupt

# Common Coding Patterns
- Each networking is one of the citizens inside the kernel
  - Must make proper and fair use of memory, CPU, and all other shared resources
- Most features interact with other kernel components depending on the feature.
  - try to follow similar mechanisms to implement similar functionalities(no need to reinvent the wheel every time).
- Some requirements are common to several kernel components
  - the need to allocate several instances of the same data structure type
  - the need to keep track of references to an instance of a data structure to avoid unsafe memory deallocations
- subsystem (kernel component): 
  - a collection of files that implement a major set of features.
    - such as IP or routing
  - to be maintained by the same people and to change in lockstep
    - lockstep: ???????

## Memory Caches
- The kernel uses kmalloc and kfree functions to allocate and free a memory block
  - kmalloc: allocate a memory block
  - kfree: free a memory block
- A special memory cache
  - Associated kernel component initialization routine allocate this when allocation and deallocation happen frequently.
  - When a memory block is freed, it is actually returned to the same cache from which it was allocated.
    - 무슨 말인지 정확히 모르겠음 ??????????????
  - ex. fib_hash_init

### Network data structures memory caches
- Socket buffer descriptors
  - allocated by `skb_init`in `net/core/sk_buff.c`
  - allocate `sk_buff` buffer descriptors
    - What is a descriptor ??????????????
  - `sk_buff`: the highest number of allocations and deallocations in the networking subsystem.
- Neighboring protocol mappings
  - to allocate the data structures that store L3-to-L2 address mappings.
- Routing Tables
  - uses two memory caches for two of the data structures that define routes.
- `kmem_cache_alloc` or `/proc`
  - limits on the number of instances

## Caching and Hash Tables
- L3-to-L2 mappings (such as the ARPcache used by IPv4)
- Cache lookup routines often take an input parameter that says whether a cache miss should or should not create a new element and add it to the cache. Other lookup routines simply add missing elements all the time.
  - cache miss????
- Standard way
  - put inputs in a list
  - Travesing takes too long.
    - use the hash key to look up
  - it is always important to minimize the number of inputs that hash to the same value.
- Increase the size of the hash table &rarr; decrease the average length of the collision lists and improves the average lookup time.
- To reduce the damage of Denial of Service (DoS) add a random component (regularly changed) to the key used to distribute elements in the cache’s buckets.
  - cache bucket ????????????????????????

## Reference Counts
- When a piece of code tries to access a data structure that has already been freed, the kernel is not very happy, and the user is rarely happy with the kernel’s reaction.
- to avoid those nasty problems
- to make garbage collection mechanisms easier and more effective
- For any data structure type that requires a reference count, the kernel component that owns the structure usually exports two functions that can be used to increment and decrement the reference count.
  - `xxx_hold`, `xxx_release(xxx_put)`
- The use of the reference count is a simple but effective mechanism to avoid freeing still-referenced data structures.
- Problems:
  - If you release a reference to a data structure but forget to call the xxx_release function, the kernel will never allow the data structure to be freed. This leads to gradual memory exhaustion.
  - If you take a reference to a data structure but forget to call xxx_hold, and at some later point you happen to be the only  reference holder, the structure will be prematurely freed because you are not accounted for. This case definitely can be more catastrophic than the previous one; your next attempt to access the structure can corrupt other data or cause a kernel panic that brings down the whole system instantly.
  

### Reference count Increament Cases
- There is a close relationship between two data structure types. In this case, one of the two often maintains a pointer initialized to the address of the second one.
- A timer is started whose handler is going to access the data structure. When the timer is fired, the reference count on the structure is incremented, because the last thing you want is for the data structure to be freed before the timer expires.
- A successful lookup on a list or a hash table returns a pointer to the matching element. In most cases, the returned result is used by the caller to carry out some task. Because of that, it is common for a lookup routine to increase the reference count on the matching element, and let the caller release it when necessary.

## Garbage Collection
- Most kernel subsystems implement some sort of garbage collection to reclaim the memory held by unused or stale data structure instances.

### Asynchronous GC
The conditions that make a data structure eligible for deletion depend on the features and logic of the subsystem, but a common criterion is the presence of **a null reference count**.

### Synchronous GC
The criteria used to select the data structures eligible for deletion are not necessarily the same ones used by asynchronous cleanup.

## Function Pointers and Virtual Function Tables (VFTs)
Function pointers are a convenient way to write clean C code while getting some of the benefits of the object-oriented languages.
A key advantage to using function pointers is that they can be initialized differently depending on various criteria and the role played by the object. Function pointers are used extensively in the networking code.

### Drawback of Function Pointers
Function pointers have one main drawback: they make browsing the source code a little harder. While going through a given code path, you may end up focusing on a function pointer call. In such cases, before proceeding down the code path, you need to find out how the function pointer has been initialized.

### VFT
A set of function pointers grouped into a data structure are often referred to as a virtual function table (VFT). When a VFT is used as the interface between two major subsystems, such as the L3 and L4 protocol layers, or when the VFT is simply exported as an interface to a generic kernel component (set of objects), the number of function pointers in it may swell to include many different pointers that accommodate a wide range of protocols or other features.

## goto Statements
Deprecated but still use in linux kernel.  
The use of goto statements can reduce the readability of the code, and make debugging harder, because at any position following a goto you can no longer derive unequivocally the conditions that led the execution to that point.

### goto in Networking
carefully placed goto statements can make it easier to jump to code that handles undesired or peculiar events. In kernel programming, and particularly in networking, such events are very common, so goto becomes a convenient tool.

## Vector Definitions
In some cases, the definition of a data structure includes an optional block at the end.

## Conditional Directives (#ifdef and family)
- To check whether a given feature is supported by the kernel
- Configuration tools such as make xconfig determine whether the feature is compiled in, not supported at all, or loadable as a module.

## Compile-Time Optimization for Condition Checks
An example of the optimization made possible by the likely and unlikely macros is in handling options in the IPheader. The use of  Poptions is limited to very specific cases, and the kernel can safely assume that most IPpackets do not carry IPoptions.
When the kernel forwards an IPpacket, it needs to take care of options according to the rules.

## Mutual Exclusion
Locking is used extensively in the networking code, and you are likely to see it come up as an issue under every topic in this book.
Each mutual exclusion mechanism is the best choice for particular circumstances.
- Spin locks
  - only one thread of execution at a time.
  - Because of the waste caused by looping, spin locks are used only on multiprocessor systems, and generally are used only when the developer expects the lock to be held for short intervals.
  - Also because of the waste caused to other threads, a thread of execution must not sleep while holding a spin lock.
- Read-write spin locks
  - When the uses of a given lock can be clearly classified as read-only and readwrite, the use of read-write spin locks is preferred. The difference between spin locks and read-write spin locks is that in the latter, multiple readers can hold the lock at the same time.
- Read-Copy-Update (RCU)
  - RCU is one of the latest mechanisms made available in Linux to provide mutual exclusion.
  - An example where RCU is used in the networking code is the routing subsystem. Lookups are more frequent than updates on the cache and the routine that implements the routing cache lookup does not block in the middle of the search.

## Conversions Between Host and Network Order
- Data structures spanning more than one byte can be stored in memory with two different formats: Little Endian and Big Endian. The first format stores the least significant byte at the lowest memory address, and the second does the opposite. The format used by an operating system such as Linux depends on the processor in use. For example, Intel processors follow the Little Endian model, and Motorola processors use the Big Endian model.
- Suppose our Linux box receives an IPpacket from a remote host. Because it does not know which format, Little Endian or Big Endian, was used by the remote host to initialize the protocol headers, how will it read the header? For this reason, each protocol family must define what “endianness” it uses. The TCP/IP stack, for example, follows the Big Endian model.
- every time the kernel needs to read, save, or compare a field of the IP header that spans more than one byte, it must first convert it  from network byte order to host byte order or vice versa.


## Catching Bugs
- The kernel uses the BUG_ON and BUG_TRAP macros to catch cases where such conditions are not met. When the input condition
to BUG_TRAP is false, the kernel prints a warning message. BUG_ON instead prints an error message and panics.

## Statistics
It is a good habit for a feature to collect statistics about the occurrence of specific conditions, such as cache lookup successes and failures, memory allocation successes and failures, etc.

## Measuring Time
- The kernel often needs to measure how much time has passed since a given moment. 
- For example, a routine that carries on a CPU-intensive task often releases the CPU after a given amount of time. It will continue its job when it is rescheduled for execution. This is especially important in kernel code, even though the kernel supports kernel preemption. A common example in the networking code is given by the routines that implement garbage collection.


# User-Space Tools
- `iputils`
  - Besides the perennial command ping, iputils includes arping (used to generate ARP requests), the Network Router Discovery daemon rdisc, and others.
    - ARP: ????????????
- `net-tools`
  - This is a suite of networking tools, where you can find the well-known `ifconfig`, `route`, `netstat`, and `arp`, but also `ipmaddr`, `iptunnel`, `ether-wake`, `netplugd`, etc.
- `IPROUTE2`
  - This is the new-generation networking configuration suite.
  - Through an omnibus command named `ip`, the suite can be used to configure IPaddresses and routing along with all of its advanced features, neighboring protocols, etc.


# Browsing the Source Code
It is time to say goodbye to grep and invest 15 minutes in learning how to use the aforementioned tools
One that I would like to suggest to those that do not know it already is `cscope`, which you can download from http://cscope.sourceforge.net. It is a simple yet powerful tool for searching, for example, where a function or variable is defined, where it is called, etc.

## Dead Code

# When a Feature Is Offered as a Patch




# Appendix
## Reference