# DOmath

A performance-oriented Zig library for linear algebra (right now only vectors).

## Motivation

Traditionally math libraries operate on the assumption that vectors are processed one at a time. This hasn't really changed with the arrival of SIMD computing. It has been used to make significant performance gains, but most of the time still only at the scope of one vector at a time. This was maybe fine when SIMD registers were still pretty small, but nowadays hardware supporting AVX-512 variants is common and even wider registers are already being specd for Arm (SVE2) and Risc-V. These registers can fit way more than 4 floats (128 bits), and vectors rarely get wider than that.

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

Further usage examples are in [`bench/domath.zig`](bench/domath.zig) and the tests in [`src/vector.zig`](src/vector.zig).

## Benchmarks

This library comes with some basic benchmarks against popular linear algebra libraries written in Zig using [zBench](https://github.com/hendriknielaender/zBench):

- [zalgebra](https://github.com/kooparse/zalgebra/)
- [zm](https://github.com/griush/zm)
- [zlm](https://github.com/ziglibs/zlm) (sadly still on Zig 0.13.0, but [someone forked it](https://github.com/nukkeldev/zlm))

```
zig build bench --release=fast
```

On a M1 MacBook Pro (128-bit SIMD registers):

```
benchmark              runs     total time     time/run (avg ± σ)     (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
domath: normalize4     3357     4.972s         1.481ms ± 45.508us     (1.365ms ... 1.654ms)        1.5ms      1.54ms     1.558ms
zalgebra: normalize4   3031     4.999s         1.649ms ± 12.1us       (1.64ms ... 1.849ms)         1.649ms    1.706ms    1.723ms
zm: normalize4         3187     4.998s         1.568ms ± 10.548us     (1.563ms ... 1.77ms)         1.566ms    1.608ms    1.626ms
zlm: normalize4        2657     4.984s         1.876ms ± 15.664us     (1.861ms ... 2.054ms)        1.883ms    1.941ms    1.978ms
benchmark              runs     total time     time/run (avg ± σ)     (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
domath: scaleByLen3    4942     4.996s         1.011ms ± 6.328us      (1ms ... 1.142ms)            1.011ms    1.036ms    1.053ms
zalgebra: scaleByLen3  3384     4.982s         1.472ms ± 28.244us     (1.429ms ... 1.831ms)        1.497ms    1.526ms    1.536ms
zm: scaleByLen3        3370     5.036s         1.494ms ± 28.167us     (1.432ms ... 1.594ms)        1.523ms    1.53ms     1.541ms
zlm: scaleByLen3       4424     5.001s         1.13ms ± 15.566us      (1.112ms ... 1.246ms)        1.14ms     1.168ms    1.181ms
benchmark              runs     total time     time/run (avg ± σ)     (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
domath: multiplyAddNeg 5045     5.004s         992.062us ± 15.583us   (921.75us ... 1.114ms)       997.458us  1.028ms    1.041ms
zalgebra: multiplyAddN 5355     4.999s         933.645us ± 17.389us   (916.458us ... 1.362ms)      941.958us  980.959us  996.166us
zm: multiplyAddNegate2 5372     4.993s         929.514us ± 13.952us   (916.25us ... 1.267ms)       934.208us  971.292us  984.584us
zlm: multiplyAddNegate 5618     5.004s         890.838us ± 14.141us   (877.209us ... 1.005ms)      895.541us  934.208us  949.166us
benchmark              runs     total time     time/run (avg ± σ)     (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
domath: cross3         5045     5.014s         993.998us ± 17.183us   (974.375us ... 1.148ms)      999.667us  1.062ms    1.086ms
zalgebra: cross3       3633     4.992s         1.374ms ± 19.711us     (1.322ms ... 1.446ms)        1.382ms    1.41ms     1.419ms
zm: cross3             3635     4.998s         1.375ms ± 19.411us     (1.32ms ... 1.444ms)         1.383ms    1.412ms    1.421ms
zlm: cross3            5430     5.003s         921.455us ± 11.811us   (887.917us ... 984.167us)    930us      949.708us  956.542us
```

zalgebra and zm store `@Vector`s directly in memory so I suspect that their bad results for `scaleByLen3` and `cross3` come from the fact that `@alignOf(@Vector(3, f32)) == 16` on this machine (so more memory to load/more cache misses).

On a Ryzen 3600X (256-bit SIMD registers, outdated benchmark with lots of alloc overhead):

```
benchmark              runs     total time     time/run (avg ± σ)     (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
domath: normalize4     636      4.884s         7.68ms ± 1.083ms       (6.821ms ... 13.074ms)       7.843ms    11.305ms   11.698ms
zalgebra: normalize4   409      4.973s         12.16ms ± 580.89us     (11.541ms ... 15.237ms)      12.434ms   13.992ms   14.449ms
zm: normalize4         445      5.011s         11.261ms ± 660.987us   (10.598ms ... 14.738ms)      11.499ms   13.859ms   14.127ms
zlm: normalize4        338      4.979s         14.73ms ± 964.525us    (13.602ms ... 18.841ms)      15.181ms   18.193ms   18.779ms
benchmark              runs     total time     time/run (avg ± σ)     (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
domath: scaleByLen3    901      4.9s           5.439ms ± 568.74us     (4.892ms ... 9.588ms)        5.562ms    7.543ms    8.145ms
zalgebra: scaleByLen3  480      5.228s         10.893ms ± 1.321ms     (9.367ms ... 17.327ms)       11.575ms   14.743ms   15.589ms
zm: scaleByLen3        494      5.055s         10.233ms ± 753.231us   (9.479ms ... 14.374ms)       10.509ms   12.807ms   13.026ms
zlm: scaleByLen3       676      5.053s         7.474ms ± 690.681us    (6.805ms ... 10.909ms)       7.802ms    9.938ms    10.289ms
benchmark              runs     total time     time/run (avg ± σ)     (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
domath: multiplyAddNeg 246      5.009s         20.365ms ± 863.47us    (19.256ms ... 24.944ms)      20.785ms   23.164ms   23.414ms
zalgebra: multiplyAddN 561      4.959s         8.84ms ± 1.152ms       (7.698ms ... 13.969ms)       9.019ms    12.893ms   13.668ms
zm: multiplyAddNegate2 564      5.215s         9.246ms ± 1.315ms      (7.924ms ... 15.909ms)       9.697ms    13.865ms   15.286ms
zlm: multiplyAddNegate 589      5.09s          8.641ms ± 1.203ms      (7.574ms ... 14.647ms)       8.909ms    13.384ms   13.629ms
benchmark              runs     total time     time/run (avg ± σ)     (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
domath: cross3         230      5.003s         21.753ms ± 1.076ms     (20.812ms ... 27.636ms)      22.076ms   26.756ms   26.921ms
zalgebra: cross3       358      5.526s         15.437ms ± 2.554ms     (12.725ms ... 27.239ms)      16.352ms   22.89ms    24.535ms
zm: cross3             345      5.202s         15.081ms ± 2.203ms     (12.818ms ... 23.941ms)      16.133ms   22.407ms   23.514ms
zlm: cross3            489      4.834s         9.886ms ± 1.014ms      (8.944ms ... 15.833ms)       10.106ms   13.526ms   13.89ms
```

Sadly I don't have access to a machine with AVX512 support or something similiar. I suspect that my approach could perform quite a lot better on one of those as the amount of instructions it compiles to with AVX512/AVX10 enabled shrinks dramatically ([godbolt](https://godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXACx8BBAKoBnTAAUAHpwAMvAFYTStJg1AAvPMFJL6yAngGVG6AMKpaAVxYM9DgDJ4GmADl3ACNMYhAAJgBmUgAHVAVCWwZnNw89eMSbAV9/IJZQ8OiLTCtshiECJmICVPdPLhKy5MrqglzAkLDImIUqmrr0xr62jvzCnoBKC1RXYmR2DjQGPoBqegZVgFIogBFVqJ2AIS2NAEFltZExaoVtve2AViOoiK3H3apX49OLgTWAPKxABqmGQ932AAFQdYSBBXqRVl8IpMfudfr9MKp4jUkZtkMQEgoohBfqtyasGKgxHgmHcAG4AfT8IFWACpLgQnkcNu9dhz/lzrgZiApSGSKVSaXTVgB3Zmedmc7m8j4ClZC0QisUS8lS2i0u6zAis9Vrd48xh8tnC27i86TVb01B4dDbADsJ3OFNWyvp6lWQJhEKdCotGj5ADo2Wizj6/QBPVlBsEhpl%2BC1cKMxqJeuMUv3GZMg1M7fbphgWt4faOx3W%2BwVygMp8FluVh54Rms5vPxxuypOBkuth7yjPPLPd2N9jVyotD4NtseV57V3a13MY70U43h7MhqD0hPsueOgC0qwg9OMJ4Hk1Rm%2B35N3E/3bcPN7ZTfPl/9t%2BM97TjurgEFWb4PIeqi3gmP5XseX6yqogGPmcWzursvwcNMtCcI8vCeBwWikKgnAAFpmKsCizPMmDbNEPCkCahFYdMADWIBRFEkacTxvF8QAbPonCSPhmi8CRHC8AoIAaIxYnTHAsBIGgLCxHQYTkJQKlqfQ4TGBoUiyTQtAEGE0kQMEYmkMEfjVEm3C8DZzDEAmALBNoYJMQxKlsIIAIMLQ9lEVgwSuMAjhiLQ0kOaQWAsIYwDiMxsV4MQnl4PSmDRURWJgiBiwMX4pk4clBrBMQdnOFgVkEMQeAsJwDGZcQwQJJguyYPFRgGkY8l8AYwAKMCeCYLKQKMI1vD8IIwrsFIMiCIoKjqMluiNAYvWmOYZXSZA0yoLE5TRWeAKrAASqUmB0pgABidJcme/TAJgXKqAAHPxjL8ZIZ6yn46CoLKdxniwyCxK4qzGAwzVROJzV1Vgu0QNMljpXYEAOIMDSkD4fidAU3SNJkSQCFjGQJCTDBjF04TDJd1gtP0tQuPUeiowzAitDU1ME7TFhM2TwxMzzExcCj1ELBI2G4aJyUSZDFEaJGUiRhol64IQJB0VEYu8ExWj3qQ7GcdxfFm5xgklSJpANVwGiyQRRESVJMlycxCkwIgKCoKp6lkBQEDaX7ID6YZfB0KZoqUJZyVOXZk3WbZLluR51gJz5jAEP5gVWSFYURbQUUJ3FCVJcFqXpZl2W8LlyD5QnRWlFZZUVS5VWLERtX1QnzWtUoHVdYlfigO7/VMINw2jeNBEMdNwhanN0hz0tahWboET6AlIdmPoeDBEj%2B2Hckx2nRd9DXXdayPdUz2vR9X0/X9DAA0Dqwg2DENQ811uoPDrpZfAFG9Nyj2GfoLHGz8RaEziBTco4DiblCgXzdm5QubMzSNjFBjNRh43GNAkYAwWZDH5jgvINMpYzDmJLMWQkOB4VII7cSnAFbAFWErFWasIAayIMQbWus3YGzYhxLi5tza0Otrbe2DCrLOwsK7fWLFaERFlk7Tget5LTGaokOwkggA%3D%3D%3D)).

For now, my approach seems to perform better for certain workloads and is at least competitive for the rest of them, with some pretty glaring exceptions.

I will add more benchmarks in the future.