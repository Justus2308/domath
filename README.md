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

Designing the API was/is quite challenging since C-like languages just really aren't designed for operating on multiple elements at once.

I've tried to make using this library as convenient as possible by making interacting with `std.MultiArrayList` easy and providing an `Accumulator` that manages intermediary buffers for you when chaining operations.

```zig
const std = @import("std");
const domath = @import("domath");

const v3 = domath.v3f32;

var in_list = std.MultiArrayList(struct { a: f32, b: f32, i: usize, c: f32 }).empty;
const factors = v3.splatScalar(2);

// fill list with your data...

var out_list = std.MultiArrayList(struct { x: f32, y: f32, z: f32 }).empty;
try out_list.ensureTotalCapacity(std.heap.page_allocator, in_list.len);

const in_slice = in_list.slice();
const out_slice = out_list.slice();

var offset: usize = 0;
while (offset < in_list.len) : (offset += v3.vectors_per_op) {
    const in = v3.fromMultiArrayList(in_slice, .{ .a, .b, .c }, offset);
    const out = v3.fromMultiArrayList(out_slice, .{ .x, .y, .z }, offset);

    var accu = v3.Accumulator.begin(.normalize, &in, .{});
    accu.cont(.scale, .{&factors});
    accu.end(.invert, .{}, &out);
}

// deal with remaining elements...

// out_list(x, y, z) now contains in_list(a, b, c) but normalized, scaled by factors and inverted.

```

If you want to manage your intermediary buffers yourself, there are also convenience functions/types available for working with those:

- `Slicable` can hold exactly one batch of vector elements
- `slices()` turns a pointer to a `Slicable` into an array of pointers to appropriate sub-slices
- `Scalars` and `Bools` can hold exactly one batch of scalars/bools

Conversion from one vector type to another is possible with `cast()` and `swizzle()`.