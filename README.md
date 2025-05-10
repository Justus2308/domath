# DOmath

A performance-oriented Zig library for linear algebra (right now only vectors).

## Motivation

Traditionally math libraries operate on the assumption that vectors are processed one at a time. This hasn't really changed with the dawn of SIMD computing. It has been used to make significant performance gains, but most of the time still only at the scope of one vector at a time. This was maybe fine when SIMD registers were still pretty small, but nowadays hardware supporting AVX-512 variants is common and even wider registers are already being specd for Arm (SVE2) and Risc-V. These registers can fit way more than 4 floats (128 bits), and vectors rarely get wider than that. On top of that, some common vector operation like the cross product require lots of internal shuffling and reduce stages which is not something SIMD excels at.

DOmath is built to utilize modern SIMD instructions to the max by operating on many vectors at once and on every dimension individually, if possible. It is built around an optimal data layout rather than an arbitrary abstraction, hence <ins>D</ins>ata <ins>O</ins>riented <ins>math</ins>.
Of course this requires storing vectors struct-of-array instead of array-of-structs style. Luckily Zig makes working with this kind of data layout easy ([`std.MultiArrayList`](https://github.com/ziglang/zig/blob/master/lib/std/multi_array_list.zig)).

## Implementation

Zig vectors don't really have a well-defined memory layout at the moment and their alignment requirements are at times... surprising, so this library aims to not expose the user to any raw vectors but rather to work on plain arrays of data and only use vectors for internal operations. Zig guarantees that arrays coerce to vectors and vice versa ([docs](https://ziglang.org/documentation/master/#Vectors)). This makes working with data that has to have a certain layout easy (e.g. GPU buffers).

All parameters are marked `noalias` to enable the compiler to do load/store elimination when chaining operations. This means that the buffers supplied to a function cannot overlap with each other.
It is thus recommended to use this library in conjunction with immutable data structures (which also makes multi threading easier).

## Usage

### Add this project to yours as a dependency:

1. Run this command:

```sh
zig fetch --save git+https://github.com/Justus2308/domath.git
```

2. Import the module into your `build.zig` file

```zig
const domath_dependency = b.dependency("domath", .{
    .target = target,
    .optimize = optimize,
});
exe.root_module.addImport("domath", domath_dependency.module("domath"));
```

### Use as a module in your code:

The API is still a little contrived, so instead of an artificially constructed 'perfect' example here is some 

```zig
const std = @import("std");
const domath = @import("domath");

```