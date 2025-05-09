const std = @import("std");
const testing = std.testing;
const assert = std.debug.assert;

/// Assumes cache line aligned batches
pub fn calcBatchSize(comptime Element: type) comptime_int {
    verifyElemType(Element);
    return @max(
        std.simd.suggestVectorLength(Element) orelse 1,
        @divExact(std.atomic.cache_line, @sizeOf(Element)),
    );
}

fn verifyElemType(comptime Element: type) void {
    // bools are not supported, most ops dont make sense for them.
    switch (@typeInfo(Element)) {
        .int => |info| if (info.bits == 0) @compileError("needs to be able to store 1"),
        .optional => |info| if (@typeInfo(info.child) != .pointer) @compileError("only optional pointers are allowed"),
        .pointer => |info| if (info.is_allowzero == false) @compileError("needs to be able to store null"),
        .float => {},
        else => @compileError("unsupported element type"),
    }
}

/// Create a namespace for vector math functions specialized on `len` and `Element` to live in.
/// Supports floats, integers and nullable pointers.
/// Expects batches of `vectors_per_op` items (depends on build target).
/// Note that all `out` parameters are marked `noalias` to facilitate load/store optimizations
/// when chaining vector operations.
pub fn namespace(comptime len: comptime_int, comptime Element: type) type {
    return namespaceWithConfig(len, Element, .{});
}

pub const Config = struct {
    batch_size: ?usize = null,
};

/// Create a namespace for vector math functions specialized on `len` and `Element` to live in.
/// Supports floats, integers and nullable pointers.
/// Expects batches of `vectors_per_op` items (depends on build target/config).
/// Note that all parameters are marked `noalias` to facilitate load/store optimizations
/// when chaining vector operations.
pub fn namespaceWithConfig(comptime len: comptime_int, comptime Element: type, comptime config: Config) type {
    verifyElemType(Element);
    return struct {
        pub const vectors_per_op = config.batch_size orelse @as(usize, calcBatchSize(Element));
        const OpVec = @Vector(vectors_per_op, Element);

        pub const Scalars = [vectors_per_op]Element;
        pub const Bools = [vectors_per_op]bool;

        fn scalarFromRawValue(comptime raw: comptime_int) Element {
            return switch (@typeInfo(Element)) {
                .int, .float => raw,
                .optional, .pointer => @ptrFromInt(raw),
                else => unreachable,
            };
        }

        const zero_val = scalarFromRawValue(0);
        const one_val = scalarFromRawValue(1);

        const zeroes_vec: OpVec = @splat(zero_val);
        const ones_vec: OpVec = @splat(one_val);

        pub fn zeroesScalar() Scalars {
            return splatScalar(zero_val);
        }
        pub fn onesScalar() Scalars {
            return splatScalar(one_val);
        }

        pub fn zeroes(noalias out: *const [len]*Scalars) void {
            splat(zero_val, out);
        }
        pub fn ones(noalias out: *const [len]*Scalars) void {
            splat(one_val, out);
        }

        pub fn splatScalar(in: Element) Scalars {
            return @splat(in);
        }
        pub fn splat(in: Element, noalias out: *const [len]*Scalars) void {
            for (out) |vec_out| {
                vec_out.* = @splat(in);
            }
        }

        pub fn asIn(slice: []const Element, offset: usize) *const Scalars {
            return @ptrCast(slice[offset..][0..vectors_per_op]);
        }
        pub fn asOut(slice: []Element, offset: usize) *Scalars {
            return @ptrCast(slice[offset..][0..vectors_per_op]);
        }

        pub fn add(
            noalias v_in: *const [len]*const Scalars,
            noalias w_in: *const [len]*const Scalars,
            noalias out: *const [len]*Scalars,
        ) void {
            for (v_in, w_in, out) |vec1_in, vec2_in, vec_out| {
                const opv1: OpVec = vec1_in.*;
                const opv2: OpVec = vec2_in.*;
                vec_out.* = (opv1 + opv2);
            }
        }
        pub fn sub(
            noalias v_in: *const [len]*const Scalars,
            noalias w_in: *const [len]*const Scalars,
            noalias out: *const [len]*Scalars,
        ) void {
            for (v_in, w_in, out) |vec1_in, vec2_in, vec_out| {
                const opv1: OpVec = vec1_in.*;
                const opv2: OpVec = vec2_in.*;
                vec_out.* = (opv1 - opv2);
            }
        }
        pub fn mul(
            noalias v_in: *const [len]*const Scalars,
            noalias w_in: *const [len]*const Scalars,
            noalias out: *const [len]*Scalars,
        ) void {
            for (v_in, w_in, out) |vec1_in, vec2_in, vec_out| {
                const opv1: OpVec = vec1_in.*;
                const opv2: OpVec = vec2_in.*;
                vec_out.* = (opv1 * opv2);
            }
        }
        pub fn div(
            noalias v_in: *const [len]*const Scalars,
            noalias w_in: *const [len]*const Scalars,
            noalias out: *const [len]*Scalars,
        ) void {
            for (v_in, w_in, out) |vec1_in, vec2_in, vec_out| {
                const opv1: OpVec = vec1_in.*;
                const opv2: OpVec = vec2_in.*;
                vec_out.* = (opv1 / opv2);
            }
        }
        /// Guarantees that division by zero results in zero.
        pub fn divAllowZero(
            noalias v_in: *const [len]*const Scalars,
            noalias w_in: *const [len]*const Scalars,
            noalias out: *const [len]*Scalars,
        ) void {
            for (v_in, w_in, out) |vec1_in, vec2_in, vec_out| {
                const opv1: OpVec = vec1_in.*;
                const opv2: OpVec = vec2_in.*;
                const res = (opv1 / opv2);
                vec_out.* = @select(Element, (opv2 == zeroes_vec), zeroes_vec, res);
            }
        }
        pub fn mod(
            noalias v_in: *const [len]*const Scalars,
            noalias w_in: *const [len]*const Scalars,
            noalias out: *const [len]*Scalars,
        ) void {
            for (v_in, w_in, out) |vec1_in, vec2_in, vec_out| {
                const opv1: OpVec = vec1_in.*;
                const opv2: OpVec = vec2_in.*;
                vec_out.* = (opv1 % opv2);
            }
        }

        pub fn lengths(noalias in: *const [len]*const Scalars, noalias out: *Scalars) void {
            const lens_sqrd = innerLengthsSqrd(in);
            const res = @sqrt(lens_sqrd);
            out.* = res;
        }
        pub fn lengthsSqrd(noalias in: *const [len]*const Scalars, noalias out: *Scalars) void {
            const res = innerLengthsSqrd(in);
            out.* = res;
        }
        inline fn innerLengthsSqrd(noalias in: *const [len]*const Scalars) OpVec {
            var res = zeroes_vec;
            for (in) |vec| {
                const opvec: OpVec = vec.*;
                const sqrd = (opvec * opvec);
                res += sqrd;
            }
            return res;
        }

        pub fn normalize(noalias in: *const [len]*const Scalars, noalias out: *const [len]*Scalars) void {
            const lens: OpVec = @sqrt(innerLengthsSqrd(in));
            const ilens = (ones_vec / lens);
            for (in, out) |vec_in, vec_out| {
                const opvec: OpVec = vec_in.*;
                vec_out.* = @select(Element, (lens == zeroes_vec), zeroes_vec, (opvec * ilens));
            }
        }

        pub fn scale(
            noalias v_in: *const [len]*const Scalars,
            noalias s_in: *const Scalars,
            noalias out: *const [len]*Scalars,
        ) void {
            const factors: OpVec = s_in.*;
            for (v_in, out) |vec_in, vec_out| {
                var opvec: OpVec = vec_in.*;
                opvec *= factors;
                vec_out.* = opvec;
            }
        }

        pub fn dots(
            noalias v_in: *const [len]*const Scalars,
            noalias w_in: *const [len]*const Scalars,
            noalias out: *Scalars,
        ) void {
            var res = zeroes_vec;
            for (v_in, w_in) |vec1, vec2| {
                const opv1: OpVec = vec1.*;
                const opv2: OpVec = vec2.*;
                res += (opv1 * opv2);
            }
            out.* = res;
        }

        pub const cross = switch (len) {
            2 => cross2,
            3 => cross3,
            else => crossUnimpl,
        };
        fn cross2(
            noalias v_in: *const [len]*const Scalars,
            noalias w_in: *const [len]*const Scalars,
            noalias out: *Scalars,
        ) void {
            const prod1 = (@as(OpVec, v_in[0].*) * @as(OpVec, w_in[1].*));
            const prod2 = (@as(OpVec, v_in[1].*) * @as(OpVec, w_in[0].*));
            out.* = (prod1 - prod2);
        }

        fn cross3(
            noalias v_in: *const [len]*const Scalars,
            noalias w_in: *const [len]*const Scalars,
            noalias out: *const [len]*Scalars,
        ) void {
            const vx: OpVec = v_in[0].*;
            const vy: OpVec = v_in[1].*;
            const vz: OpVec = v_in[2].*;

            const wx: OpVec = w_in[0].*;
            const wy: OpVec = w_in[1].*;
            const wz: OpVec = w_in[2].*;

            out[0].* = ((vy * wz) - (vz * wy));
            out[1].* = ((vz * wx) - (vx * wz));
            out[2].* = ((vx * wy) - (vy * wx));
        }

        fn crossUnimpl(
            v_in: *const [len]*const Scalars,
            w_in: *const [len]*const Scalars,
            out: anytype,
        ) noreturn {
            _ = .{ v_in, w_in, out };
            @panic("cross product is only implemented for dim=2-3");
        }

        pub fn distances(
            noalias v_in: *const [len]*const Scalars,
            noalias w_in: *const [len]*const Scalars,
            noalias out: *Scalars,
        ) void {
            const dist_sqrd = innerDistancesSqrd(v_in, w_in);
            const res = @sqrt(dist_sqrd);
            out.* = res;
        }
        pub fn distancesSqrd(
            noalias v_in: *const [len]*const Scalars,
            noalias w_in: *const [len]*const Scalars,
            noalias out: *Scalars,
        ) void {
            const res = innerDistancesSqrd(v_in, w_in);
            out.* = res;
        }
        inline fn innerDistancesSqrd(
            noalias v_in: *const [len]*const Scalars,
            noalias w_in: *const [len]*const Scalars,
        ) OpVec {
            var res = zeroes_vec;
            for (v_in, w_in) |vec1, vec2| {
                const opv1: OpVec = vec1.*;
                const opv2: OpVec = vec2.*;
                const tmp = (opv1 - opv2);
                res += (tmp * tmp);
            }
            return res;
        }

        pub fn invert(noalias in: *const [len]*const Scalars, noalias out: *const [len]*Scalars) void {
            const simd_ones = ones_vec;
            for (in, out) |vec_in, vec_out| {
                const opvec: OpVec = vec_in.*;
                vec_out.* = (simd_ones / opvec);
            }
        }

        pub const negate = switch (@typeInfo(Element)) {
            .float, .int => negateImpl,
            else => negateUnimpl,
        };
        fn negateImpl(noalias in: *const [len]*const Scalars, noalias out: *const [len]*Scalars) void {
            for (in, out) |vec_in, vec_out| {
                const opvec: OpVec = vec_in.*;
                vec_out.* = -opvec;
            }
        }
        fn negateUnimpl(in: *const [len]*const Scalars, out: *const [len]*Scalars) void {
            _ = .{ in, out };
            @panic("cannot negate element type");
        }

        pub fn lerp(
            noalias v_in: *const [len]*const Scalars,
            noalias w_in: *const [len]*const Scalars,
            noalias amount_in: *const Scalars,
            noalias out: *const [len]*Scalars,
        ) void {
            for (v_in, w_in, out) |vec1_in, vec2_in, vec_out| {
                const opv1: OpVec = vec1_in.*;
                const opv2: OpVec = vec2_in.*;
                vec_out.* = (opv1 + ((opv2 - opv1) * amount_in));
            }
        }

        pub fn clamp(
            noalias v_in: *const [len]*const Scalars,
            noalias vmin_in: *const [len]*const Scalars,
            noalias vmax_in: *const [len]*const Scalars,
            noalias out: *const [len]*Scalars,
        ) void {
            for (v_in, vmin_in, vmax_in, out) |vec_in, vec_min, vec_max, vec_out| {
                const opvec: OpVec = vec_in.*;
                const opmin: OpVec = vec_min.*;
                const opmax: OpVec = vec_max.*;
                vec_out.* = @min(opmax, @max(opmin, opvec));
            }
        }

        // Using `@select` here to effectively perform bitwise operations is a rather
        // unfortunate result of https://github.com/ziglang/zig/issues/14306, but
        // codegen seems to be fine anyways.

        pub fn moveTowards(
            noalias v_in: *const [len]*const Scalars,
            noalias target_in: *const [len]*const Scalars,
            noalias max_dist_in: *const Scalars,
            noalias out: *const [len]*Scalars,
        ) void {
            var dists_sqrd = zeroes_vec;
            for (v_in, target_in) |vec, target| {
                const opvec: OpVec = vec.*;
                const optrgt: OpVec = target.*;
                const d = (optrgt - opvec);
                dists_sqrd += (d * d);
            }
            const dists = @sqrt(dists_sqrd);

            const max_dists: OpVec = max_dist_in.*;

            const pred_and1 = (max_dists >= zeroes_vec);
            const pred_and2 = (dists <= max_dists);

            const pred_or1 = (dists == zeroes_vec);
            const pred_or2 = @select(bool, pred_and1, pred_and2, pred_and1); // and

            const is_at_target = @select(bool, pred_or1, pred_or1, pred_or2); // or

            for (v_in, target_in, out) |vec_in, target, vec_out| {
                const opvec: OpVec = vec_in.*;
                const optrgt: OpVec = target.*;
                const d = (optrgt - opvec);
                const res = (opvec + ((d / dists) * max_dists));
                vec_out.* = @select(Element, is_at_target, optrgt, res);
            }
        }

        pub fn eql(
            v_in: *const [len]*const Scalars,
            w_in: *const [len]*const Scalars,
            noalias out: *Bools,
        ) void {
            var res: @Vector(vectors_per_op, bool) = @splat(true);
            for (v_in, w_in) |vec1, vec2| {
                const opv1: OpVec = vec1.*;
                const opv2: OpVec = vec2.*;

                const pred = (opv1 == opv2);
                res = @select(bool, pred, res, pred);
            }
            out.* = res;
        }

        pub const approxEqAbs = switch (@typeInfo(Element)) {
            .float => approxEqAbsFloat,
            else => approxEqUnimpl,
        };
        pub const approxEqRel = switch (@typeInfo(Element)) {
            .float => approxEqRelFloat,
            else => approxEqUnimpl,
        };

        fn approxEqAbsFloat(
            v_in: *const [len]*const Scalars,
            w_in: *const [len]*const Scalars,
            tolerances_in: *const Scalars,
            noalias out: *Bools,
        ) void {
            assert(@reduce(.Min, tolerances_in) >= 0.0);

            var res: @Vector((vectors_per_op / 8), u8) = @splat(@intFromBool(true));
            for (v_in, w_in, tolerances_in) |vec1, vec2, tolerance| {
                const opv1: OpVec = vec1.*;
                const opv2: OpVec = vec2.*;
                const optol: OpVec = tolerance.*;

                const pred = (@abs(opv1 - opv2) <= optol);
                res = @select(bool, pred, res, pred);
            }
            out.* = res;
        }
        fn approxEqRelFloat(
            v_in: *const [len]*const Scalars,
            w_in: *const [len]*const Scalars,
            tolerances_in: *const Scalars,
            noalias out: *Bools,
        ) void {
            assert(@reduce(.Min, tolerances_in) > 0.0);

            var res: @Vector(vectors_per_op, bool) = @splat(true);
            for (v_in, w_in, tolerances_in) |vec1, vec2, tolerance| {
                const opv1: OpVec = vec1.*;
                const opv2: OpVec = vec2.*;
                const optol: OpVec = tolerance.*;

                const rel_tolerance = (@max(@abs(opv1), @abs(opv2)) * optol);
                const pred = (@abs(opv1 - opv2) <= rel_tolerance);
                res = @select(bool, pred, res, pred);
            }
            out.* = res;
        }
        fn approxEqUnimpl(
            _: *const [len]*const Scalars,
            _: *const [len]*const Scalars,
            _: *const Scalars,
            _: *Bools,
        ) void {
            @panic("approxEq functions are only available for float vectors");
        }

        pub fn casted(comptime T: type) type {
            return namespaceWithConfig(len, T, .{ .batch_size = vectors_per_op });
        }
        pub fn cast(comptime T: type, in: *const [len]*const Scalars, out: *const [len]*casted(T).Scalars) void {
            for (in, out) |vec_in, vec_out| {
                const opvec: OpVec = vec_in.*;
                vec_out.* = @as(casted(T).OpVec, switch (@typeInfo(Element)) {
                    .float => switch (@typeInfo(T)) {
                        .float => @floatCast(opvec),
                        .int => @intFromFloat(opvec),
                        else => |tag| @compileError("cast from float to " ++ @tagName(tag) ++ " not possible"),
                    },
                    .int => switch (@typeInfo(T)) {
                        .float => @floatFromInt(opvec),
                        .int => @intCast(opvec),
                        .pointer, .optional => @ptrFromInt(opvec),
                        else => |tag| @compileError("cast from int to " ++ @tagName(tag) ++ " not possible"),
                    },
                    .pointer, .optional => switch (@typeInfo(T)) {
                        .int => @intFromPtr(opvec),
                        .pointer, .optional => @ptrCast(opvec),
                        else => |tag| @compileError("cast from pointer to " ++ @tagName(tag) ++ " not possible"),
                    },
                    else => unreachable,
                });
            }
        }

        pub const Slicable = [len]Scalars;

        fn SliceType(comptime Pointer: type) type {
            return switch (@typeInfo(Pointer)) {
                .pointer => |ptr_info| switch (ptr_info.size) {
                    .one => switch (@typeInfo(ptr_info.child)) {
                        .array => |arr_info| if (arr_info.len != len) {
                            @compileError(std.fmt.comptimePrint("need array of 'Scalars' with length {d}, got {d}", .{ len, arr_info.len }));
                        } else if (arr_info.child != Scalars) {
                            @compileError("need array of 'Scalars', got '" ++ @typeName(arr_info.child) ++ "'");
                        } else {
                            return @Type(.{ .array = .{
                                .len = len,
                                .child = if (ptr_info.is_const) *const Scalars else *Scalars,
                                .sentinel_ptr = null,
                            } });
                        },
                        else => @compileError("need pointer to 'Slicable'"),
                    },
                    else => @compileError("need pointer to a single 'Slicable'"),
                },
                else => |info| @compileError("need a 'pointer', got '" ++ @tagName(info) ++ "'"),
            };
        }

        /// Returns an array of pointers to appropriately sized arrays/slices inside of the provided `Slicable`.
        pub inline fn slices(ptr_to_slicable: anytype) SliceType(@TypeOf(ptr_to_slicable)) {
            var arr: SliceType(@TypeOf(ptr_to_slicable)) = undefined;
            for (&arr, ptr_to_slicable) |*a, *s| {
                a.* = s;
            }
            return arr;
        }
    };
}

inline fn iota(comptime len: usize, comptime T: type, comptime offset: usize, comptime stride: usize) [len]T {
    var arr: [len]T = undefined;
    inline for (0..len) |i| {
        const raw = @as(comptime_int, ((i * stride) + offset));
        arr[i] = raw;
    }
    return arr;
}

// TESTS

test "lenghts" {
    inline for (.{ 2, 3, 4 }) |len| {
        const ns = namespace(len, f32);
        const buf: ns.Slicable = @splat(iota(ns.vectors_per_op, f32, 0, 1));
        const in = ns.slices(&buf);
        var out: ns.Scalars = undefined;
        ns.lengths(&in, &out);

        for (out, 0..) |val, i| {
            var len_sqrd: f32 = 0;
            for (buf) |vec| {
                len_sqrd += (vec[i] * vec[i]);
            }
            const length = @sqrt(len_sqrd);
            try testing.expectEqual(length, val);
        }
    }
}

test "normalize + scale roundtrip" {
    inline for (.{ 2, 3, 4 }) |len| {
        const ns = namespace(len, f32);
        const buf: [len]ns.Scalars = @splat(iota(ns.vectors_per_op, f32, 0, 3));
        const in = ns.slices(&buf);
        var normed_buf: ns.Slicable = undefined;
        var scaled_buf: ns.Slicable = undefined;
        const normed = ns.slices(&normed_buf);
        const scaled = ns.slices(&scaled_buf);
        var lengths: ns.Scalars = undefined;

        ns.normalize(&in, &normed);
        ns.lengths(&in, &lengths);
        ns.scale(&normed, &lengths, &scaled);

        for (buf, scaled_buf) |orig, result| {
            for (orig, result) |a, b| {
                try testing.expectApproxEqAbs(a, b, 0.01);
            }
        }
    }
}

test "distances" {
    inline for (.{ 2, 3, 4 }) |len| {
        const ns = namespace(len, f32);
        var buf1: ns.Slicable = @splat(iota(ns.vectors_per_op, f32, 0, 3));
        var buf2: ns.Slicable = @splat(iota(ns.vectors_per_op, f32, 1000, 2));
        const in1 = ns.slices(&buf1);
        const in2 = ns.slices(&buf2);
        var out: ns.Scalars = undefined;
        ns.distances(&in1, &in2, &out);

        for (out, 0..) |val, i| {
            var dist_sqrd: f32 = 0;
            for (0..len) |j| {
                dist_sqrd += (buf1[j][i] - buf2[j][i]) * (buf1[j][i] - buf2[j][i]);
            }
            try testing.expectEqual(@sqrt(dist_sqrd), val);
        }
    }
}

test "dots" {
    inline for (.{ 2, 3, 4 }) |len| {
        const ns = namespace(len, f32);
        const buf1: ns.Slicable = @splat(iota(ns.vectors_per_op, f32, 0, 1));
        const in1 = ns.slices(&buf1);
        const buf2: ns.Slicable = @splat(iota(ns.vectors_per_op, f32, 1000, 3));
        const in2 = ns.slices(&buf2);
        var out: ns.Scalars = undefined;
        ns.dots(&in1, &in2, &out);

        for (out, 0..) |val, i| {
            var expected: f32 = 0;
            for (buf1, buf2) |vec1, vec2| {
                expected += (vec1[i] * vec2[i]);
            }
            try testing.expectEqual(expected, val);
        }
    }
}

test "cross 2D" {
    const len = 2;
    const ns = namespace(len, f32);
    var buf1: ns.Slicable = @splat(iota(ns.vectors_per_op, f32, 0, 1));
    var buf2: ns.Slicable = @splat(iota(ns.vectors_per_op, f32, 1000, 3));
    const in1 = ns.slices(&buf1);
    const in2 = ns.slices(&buf2);
    var out: ns.Scalars = undefined;
    ns.cross(&in1, &in2, &out);

    for (0..ns.vectors_per_op) |i| {
        const vx = buf1[0][i];
        const vy = buf1[1][i];
        const wx = buf2[0][i];
        const wy = buf2[1][i];
        const expected = (vx * wy - vy * wx);
        try testing.expectEqual(expected, out[i]);
    }
}

test "cross 3D" {
    const len = 3;
    const ns = namespace(len, f32);
    var buf1: ns.Slicable = @splat(iota(ns.vectors_per_op, f32, 0, 1));
    var buf2: ns.Slicable = @splat(iota(ns.vectors_per_op, f32, 1000, 3));
    const in1 = ns.slices(&buf1);
    const in2 = ns.slices(&buf2);
    var out_buf: ns.Slicable = undefined;
    const out = ns.slices(&out_buf);
    ns.cross(&in1, &in2, &out);

    for (0..len) |j| {
        for (0..ns.vectors_per_op) |i| {
            const vx = buf1[(j + 1) % 3][i];
            const vy = buf1[(j + 2) % 3][i];
            const wx = buf2[(j + 1) % 3][i];
            const wy = buf2[(j + 2) % 3][i];
            const expected = (vx * wy - vy * wx);
            try testing.expectEqual(expected, out_buf[j][i]);
        }
    }
}

test "moveTowards" {
    inline for (.{ 2, 3, 4 }) |len| {
        const ns = namespace(len, f32);
        var pos_buf: ns.Slicable = @splat(iota(ns.vectors_per_op, f32, 0, 1));
        var tgt_buf: ns.Slicable = @splat(iota(ns.vectors_per_op, f32, 1000, 300));
        const pos = ns.slices(&pos_buf);
        const tgt = ns.slices(&tgt_buf);
        const maxd: ns.Scalars = @splat(5.0);
        var out_buf: ns.Slicable = undefined;
        const out = ns.slices(&out_buf);
        ns.moveTowards(&pos, &tgt, &maxd, &out);

        for (0..ns.vectors_per_op) |j| {
            var dist_sqrd: f32 = 0;
            for (0..len) |i| {
                const start = pos_buf[i][j];
                const target = tgt_buf[i][j];
                const d = (target - start);
                dist_sqrd += (d * d);
            }
            const dist = @sqrt(dist_sqrd);
            if (dist == 0 or dist <= 5.0) {
                for (0..len) |i| {
                    const expected = tgt_buf[i][j];
                    try testing.expectEqual(expected, out_buf[i][j]);
                }
            }
            for (0..len) |i| {
                const start = pos_buf[i][j];
                const target = tgt_buf[i][j];
                const d = (target - start);
                const expected = (start + ((d / dist) * 5.0));
                try testing.expectEqual(expected, out_buf[i][j]);
            }
        }
    }
}

test "eql" {
    inline for (.{ 2, 3, 4 }) |len| {
        const ns = namespace(len, f32);
        const buf1: ns.Slicable = @splat(iota(ns.vectors_per_op, f32, 0, 1));
        const in1 = ns.slices(&buf1);
        const buf2: ns.Slicable = @splat(iota((ns.vectors_per_op / 2), f32, 0, 1) ++ iota((ns.vectors_per_op / 2), f32, 0, 1));
        const in2 = ns.slices(&buf2);
        var out: [ns.vectors_per_op]bool = undefined;
        ns.eql(&in1, &in2, &out);

        for (0..(ns.vectors_per_op / 2)) |i| {
            try testing.expectEqual(true, out[i]);
        }
        for ((ns.vectors_per_op / 2)..ns.vectors_per_op) |i| {
            try testing.expectEqual(false, out[i]);
        }
    }
}

test "cast" {
    inline for (.{ 2, 3, 4 }) |len| {
        const ns = namespace(len, f32);

        const buf_float: ns.Slicable = @splat(iota(ns.vectors_per_op, f32, 0, 1));
        const in_float = ns.slices(&buf_float);
        var buf_int: ns.casted(usize).Slicable = undefined;
        const out_int = ns.casted(usize).slices(&buf_int);
        ns.cast(usize, &in_float, &out_int);

        for (&buf_int) |vec| {
            for (vec, 0..) |v, i| {
                try testing.expectEqual(i, v);
            }
        }

        const in_int = ns.casted(usize).slices(&buf_int);
        var buf_pointer: ns.casted(?*anyopaque).Slicable = undefined;
        const out_pointer = ns.casted(?*anyopaque).slices(&buf_pointer);
        ns.casted(usize).cast(?*anyopaque, &in_int, &out_pointer);

        for (&buf_pointer) |vec| {
            for (vec, 0..) |v, i| {
                try testing.expectEqual(@as(?*anyopaque, @ptrFromInt(i)), v);
            }
        }
    }
}
