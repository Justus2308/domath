# DOmath

A performance-oriented Zig library for linear algebra (right now only vectors).

## Motivation

Traditionally math libraries operate on the assumption that vectors are processed one at a time. This hasn't really changed with the arrival of SIMD computing. It has been used to make significant performance gains, but most of the time still only at the scope of one vector at a time. This was maybe fine when SIMD registers were still pretty small, but nowadays hardware supporting AVX-512 variants is common and even wider registers are already being specd for Arm (SVE2) and Risc-V. These registers can fit way more than 4 floats (128 bits), and vectors rarely get wider than that. On top of that, some common vector operation like the cross product require lots of internal shuffling and reduce stages which is not something SIMD excels at.

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

Designing the API was/is quite challenging since C-like languages just aren't built for operating on multiple elements at once.

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
out_list.len = in_list.len;

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

## Benchmarks

This library comes with some basic benchmarks against popular linear algebra libraries written in Zig using [zBench](https://github.com/hendriknielaender/zBench):

- [zalgebra](https://github.com/kooparse/zalgebra/)
- [zm](https://github.com/griush/zm)
- [zlm](https://github.com/ziglibs/zlm) (sadly still on Zig 0.13.0, but [someone forked it](https://github.com/nukkeldev/zlm))

On a M1 MacBook Pro (128-bit SIMD registers):

```
benchmark              runs     total time     time/run (avg ± σ)     (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
domath: normalize4     943      5s             5.302ms ± 381.416us    (4.982ms ... 9.005ms)        5.263ms    7.155ms    7.477ms
zalgebra: normalize4   895      4.958s         5.54ms ± 117.667us     (5.324ms ... 6.459ms)        5.591ms    5.993ms    6.194ms
zm: normalize4         923      5.016s         5.434ms ± 137.931us    (5.197ms ... 6.57ms)         5.48ms     6.005ms    6.205ms
zlm: normalize4        829      4.987s         6.016ms ± 105.882us    (5.826ms ... 6.75ms)         6.056ms    6.389ms    6.439ms
benchmark              runs     total time     time/run (avg ± σ)     (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
domath: scaleByLen3    1351     4.981s         3.687ms ± 90.062us     (3.535ms ... 4.488ms)        3.72ms     3.998ms    4.173ms
zalgebra: scaleByLen3  963      4.914s         5.103ms ± 125.376us    (4.875ms ... 5.656ms)        5.167ms    5.552ms    5.619ms
zm: scaleByLen3        984      4.995s         5.076ms ± 131.125us    (4.855ms ... 5.63ms)         5.16ms     5.512ms    5.554ms
zlm: scaleByLen3       1298     5.03s          3.875ms ± 92.845us     (3.731ms ... 4.636ms)        3.906ms    4.182ms    4.373ms
benchmark              runs     total time     time/run (avg ± σ)     (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
domath: multiplyAddNeg 788      4.994s         6.337ms ± 167.236us    (6.096ms ... 7.555ms)        6.396ms    6.983ms    7.094ms
zalgebra: multiplyAddN 817      4.949s         6.058ms ± 141.624us    (5.8ms ... 6.974ms)          6.142ms    6.543ms    6.6ms
zm: multiplyAddNegate2 818      5.046s         6.169ms ± 164.996us    (5.853ms ... 7.236ms)        6.234ms    6.77ms     6.943ms
zlm: multiplyAddNegate 834      4.898s         5.873ms ± 116.736us    (5.61ms ... 6.864ms)         5.944ms    6.222ms    6.317ms
benchmark              runs     total time     time/run (avg ± σ)     (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
domath: crossLerp3     469      4.908s         10.466ms ± 181.24us    (10.159ms ... 11.31ms)       10.534ms   11.1ms     11.176ms
zalgebra: crossLerp3   383      4.942s         12.904ms ± 271.734us   (12.395ms ... 14.599ms)      12.978ms   14.044ms   14.243ms
zm: crossLerp3         388      4.989s         12.859ms ± 303.344us   (12.413ms ... 14.441ms)      13.016ms   13.876ms   14.362ms
zlm: crossLerp3        543      4.956s         9.128ms ± 223.023us    (8.771ms ... 10.092ms)       9.32ms     9.652ms    9.885ms
```

zalgebra and zm store `@Vector`s directly in memory so I suspect that their bad results for `scaleByLen3` come from the fact that `@alignOf(@Vector(3, f32)) == 16` on this machine (so more memory to load).

On a Ryzen 3600X (256-bit SIMD registers):

```
benchmark              runs     total time     time/run (avg ± σ)     (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
domath: normalize4     649      5.032s         7.754ms ± 482.818us    (7.261ms ... 10.495ms)       7.846ms    9.769ms    9.967ms
zalgebra: normalize4   399      4.966s         12.447ms ± 397.971us   (12.037ms ... 13.877ms)      12.618ms   13.82ms    13.865ms
zm: normalize4         431      4.989s         11.577ms ± 437.076us   (11.1ms ... 13.527ms)        11.696ms   12.786ms   12.987ms
zlm: normalize4        346      4.986s         14.413ms ± 448.655us   (13.978ms ... 15.892ms)      14.66ms    15.731ms   15.889ms
benchmark              runs     total time     time/run (avg ± σ)     (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
domath: scaleByLen3    916      4.984s         5.441ms ± 288.734us    (5.136ms ... 7.114ms)        5.482ms    6.641ms    6.741ms
zalgebra: scaleByLen3  489      4.987s         10.2ms ± 441.075us     (9.816ms ... 12.444ms)       10.217ms   11.948ms   11.987ms
zm: scaleByLen3        486      4.977s         10.242ms ± 426.136us   (9.827ms ... 12.394ms)       10.408ms   11.887ms   11.943ms
zlm: scaleByLen3       687      4.983s         7.253ms ± 327.754us    (6.947ms ... 8.866ms)        7.245ms    8.535ms    8.772ms
benchmark              runs     total time     time/run (avg ± σ)     (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
domath: multiplyAddNeg 248      4.986s         20.107ms ± 483.768us   (19.413ms ... 21.489ms)      20.573ms   21.319ms   21.43ms
zalgebra: multiplyAddN 575      4.914s         8.546ms ± 435.49us     (8.187ms ... 11.601ms)       8.638ms    10.577ms   10.922ms
zm: multiplyAddNegate2 578      4.931s         8.531ms ± 413.565us    (8.193ms ... 12.235ms)       8.557ms    10.175ms   10.981ms
zlm: multiplyAddNegate 603      5.022s         8.328ms ± 435.142us    (7.862ms ... 10.792ms)       8.427ms    10.126ms   10.344ms
benchmark              runs     total time     time/run (avg ± σ)     (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
domath: crossLerp3     142      4.997s         35.197ms ± 672.953us   (33.459ms ... 37.408ms)      35.617ms   36.632ms   37.408ms
zalgebra: crossLerp3   245      4.904s         20.018ms ± 1.021ms     (18.962ms ... 26.338ms)      20.411ms   24.686ms   25.288ms
zm: crossLerp3         253      4.966s         19.63ms ± 922.393us    (18.655ms ... 24.743ms)      19.975ms   24.022ms   24.238ms
zlm: crossLerp3        364      5.009s         13.761ms ± 671.617us   (12.921ms ... 17.183ms)      14.101ms   16.246ms   16.546ms
```

I suspect that the bad results on `multiplyAddNegate2` and `crossLerp3` come from the fact that my approach will produce a separate multiply and add instead of a fused-multiply-add on x64.

Sadly I don't have access to a machine with AVX512 support or something similiar.

For now, my approach seems to perform better for certain workloads and is at least competitive for the rest of them, with the exception of fused-multiply-add operations.

I will add more benchmarks in the future.