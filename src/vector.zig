const std = @import("std");
const testing = std.testing;
const assert = std.debug.assert;

/// Create a namespace for vector math functions specialized on `len` and `Element` to live in.
/// Supports floats, integers and nullable pointers.
/// Expects batches of `vectors_per_op` items (dependent on build target).
/// Note that all `out` parameters are marked `noalias` to facilitate load/store optimizations
/// when chaining vector operations.
pub fn namespace(comptime len: comptime_int, comptime Element: type) type {
    // bools are not supported, most ops dont make sense for them.
    switch (@typeInfo(Element)) {
        .int => |info| if (info.bits == 0) @compileError("needs to be able to store 1"),
        .optional => |info| if (@typeInfo(info.child) != .pointer) @compileError("only optional pointers are allowed"),
        .pointer => |info| if (!info.is_allowzero) @compileError("needs to be able to store null"),
        .float => {},
        else => @compileError("unsupported element type"),
    }
    return struct {
        pub const vectors_per_op = std.simd.suggestVectorLength(Element) orelse 1;
        const OpVec = @Vector(vectors_per_op, Element);

        pub const Scalars = *const [vectors_per_op]Element;
        pub const ScalarsMut = *[vectors_per_op]Element;
        pub const Vectors = *const [len]Scalars;
        pub const VectorsMut = *const [len]ScalarsMut;

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

        pub fn zeroesScalar(noalias out: ScalarsMut) void {
            splatScalar(zero_val, out);
        }
        pub fn onesScalar(noalias out: ScalarsMut) void {
            splatScalar(one_val, out);
        }

        pub fn zeroes(noalias out: VectorsMut) void {
            splat(zero_val, out);
        }
        pub fn ones(noalias out: VectorsMut) void {
            splat(one_val, out);
        }

        pub fn splatScalar(in: Element, noalias out: ScalarsMut) void {
            out.* = @splat(in);
        }
        pub fn splat(in: Element, noalias out: VectorsMut) void {
            for (out) |vec_out| {
                vec_out.* = @splat(in);
            }
        }

        pub fn asIn(slice: []const Element, offset: usize) Scalars {
            return @ptrCast(slice[offset..][0..vectors_per_op]);
        }
        pub fn asOut(slice: []Element, offset: usize) ScalarsMut {
            return @ptrCast(slice[offset..][0..vectors_per_op]);
        }

        pub fn add(v_in: Vectors, w_in: Vectors, noalias out: VectorsMut) void {
            for (v_in, w_in, out) |vec1_in, vec2_in, vec_out| {
                const opv1: OpVec = vec1_in.*;
                const opv2: OpVec = vec2_in.*;
                vec_out.* = (opv1 + opv2);
            }
        }
        pub fn sub(v_in: Vectors, w_in: Vectors, noalias out: VectorsMut) void {
            for (v_in, w_in, out) |vec1_in, vec2_in, vec_out| {
                const opv1: OpVec = vec1_in.*;
                const opv2: OpVec = vec2_in.*;
                vec_out.* = (opv1 - opv2);
            }
        }
        pub fn mul(v_in: Vectors, w_in: Vectors, noalias out: VectorsMut) void {
            for (v_in, w_in, out) |vec1_in, vec2_in, vec_out| {
                const opv1: OpVec = vec1_in.*;
                const opv2: OpVec = vec2_in.*;
                vec_out.* = (opv1 * opv2);
            }
        }
        pub fn div(v_in: Vectors, w_in: Vectors, noalias out: VectorsMut) void {
            for (v_in, w_in, out) |vec1_in, vec2_in, vec_out| {
                const opv1: OpVec = vec1_in.*;
                const opv2: OpVec = vec2_in.*;
                vec_out.* = (opv1 / opv2);
            }
        }
        /// Guarantees that division by zero results in zero.
        pub fn divAllowZero(v_in: Vectors, w_in: Vectors, noalias out: VectorsMut) void {
            for (v_in, w_in, out) |vec1_in, vec2_in, vec_out| {
                const opv1: OpVec = vec1_in.*;
                const opv2: OpVec = vec2_in.*;
                const res = (opv1 / opv2);
                vec_out.* = @select(Element, (opv2 == zeroes_vec), zeroes_vec, res);
            }
        }
        pub fn mod(v_in: Vectors, w_in: Vectors, noalias out: VectorsMut) void {
            for (v_in, w_in, out) |vec1_in, vec2_in, vec_out| {
                const opv1: OpVec = vec1_in.*;
                const opv2: OpVec = vec2_in.*;
                vec_out.* = (opv1 % opv2);
            }
        }

        pub fn lengths(in: Vectors, noalias out: ScalarsMut) void {
            const lens_sqrd = innerLengthsSqrd(in);
            const res = @sqrt(lens_sqrd);
            out.* = res;
        }
        pub fn lengthsSqrd(in: Vectors, noalias out: ScalarsMut) void {
            const res = innerLengthsSqrd(in);
            out.* = res;
        }
        inline fn innerLengthsSqrd(in: Vectors) OpVec {
            var res = zeroes_vec;
            for (in) |vec| {
                const opvec: OpVec = vec.*;
                const sqrd = (opvec * opvec);
                res += sqrd;
            }
            return res;
        }

        pub fn normalize(in: Vectors, noalias out: VectorsMut) void {
            const lens: OpVec = @sqrt(innerLengthsSqrd(in));
            const ilens = (ones_vec / lens);
            for (in, out) |vec_in, vec_out| {
                const opvec: OpVec = vec_in.*;
                vec_out.* = @select(Element, (lens == zeroes_vec), zeroes_vec, (opvec * ilens));
            }
        }

        pub fn scale(v_in: Vectors, s_in: Scalars, noalias out: VectorsMut) void {
            const factors: OpVec = s_in.*;
            for (v_in, out) |vec_in, vec_out| {
                var opvec: OpVec = vec_in.*;
                opvec *= factors;
                vec_out.* = opvec;
            }
        }

        pub fn dots(v_in: Vectors, w_in: Vectors, noalias out: ScalarsMut) void {
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
        fn cross2(v_in: Vectors, w_in: Vectors, noalias out: ScalarsMut) void {
            const prod1 = (@as(OpVec, v_in[0].*) * @as(OpVec, w_in[1].*));
            const prod2 = (@as(OpVec, v_in[1].*) * @as(OpVec, w_in[0].*));
            out.* = (prod1 - prod2);
        }

        fn cross3(v_in: Vectors, w_in: Vectors, noalias out: VectorsMut) void {
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

        fn crossUnimpl(_: Vectors, _: Vectors, _: anytype) noreturn {
            @panic("cross product is only implemented for dim=2-3");
        }

        pub fn distances(v_in: Vectors, w_in: Vectors, noalias out: ScalarsMut) void {
            const dist_sqrd = innerDistancesSqrd(v_in, w_in);
            const res = @sqrt(dist_sqrd);
            out.* = res;
        }
        pub fn distancesSqrd(v_in: Vectors, w_in: Vectors, noalias out: ScalarsMut) void {
            const res = innerDistancesSqrd(v_in, w_in);
            out.* = res;
        }
        inline fn innerDistancesSqrd(v_in: Vectors, w_in: Vectors) OpVec {
            var res = zeroes_vec;
            for (v_in, w_in) |vec1, vec2| {
                const opv1: OpVec = vec1.*;
                const opv2: OpVec = vec2.*;
                const tmp = (opv1 - opv2);
                res += (tmp * tmp);
            }
            return res;
        }

        pub fn invert(in: Vectors, noalias out: VectorsMut) void {
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
        fn negateImpl(in: Vectors, noalias out: VectorsMut) void {
            for (in, out) |vec_in, vec_out| {
                const opvec: OpVec = vec_in.*;
                vec_out.* = -opvec;
            }
        }
        fn negateUnimpl(_: Vectors, _: VectorsMut) void {
            @panic("cannot negate element type");
        }

        pub fn lerp(v_in: Vectors, w_in: Vectors, amount_in: Scalars, noalias out: VectorsMut) void {
            for (v_in, w_in, out) |vec1_in, vec2_in, vec_out| {
                const opv1: OpVec = vec1_in.*;
                const opv2: OpVec = vec2_in.*;
                vec_out.* = (opv1 + ((opv2 - opv1) * amount_in));
            }
        }

        pub fn clamp(v_in: Vectors, vmin_in: Vectors, vmax_in: Vectors, noalias out: VectorsMut) void {
            for (v_in, vmin_in, vmax_in, out) |vec_in, vec_min, vec_max, vec_out| {
                const opvec: OpVec = vec_in.*;
                const opmin: OpVec = vec_min.*;
                const opmax: OpVec = vec_max.*;
                vec_out.* = @min(opmax, @max(opmin, opvec));
            }
        }

        pub fn moveTowards(v_in: Vectors, target_in: Vectors, max_dist_in: Scalars, noalias out: VectorsMut) void {
            var dists_sqrd = zeroes_vec;
            for (v_in, target_in) |vec, target| {
                const opvec: OpVec = vec.*;
                const optrgt: OpVec = target.*;
                const d = (optrgt - opvec);
                dists_sqrd += (d * d);
            }
            const dists = @sqrt(dists_sqrd);

            const max_dists: OpVec = max_dist_in.*;
            const is_at_target = (dists == zeroes_vec or (max_dists >= zeroes_vec and dists <= max_dists));

            for (v_in, target_in, out) |vec_in, target, vec_out| {
                const opvec: OpVec = vec_in.*;
                const optrgt: OpVec = target.*;
                const d = (optrgt - opvec);
                const res = (opvec + ((d / dists) * max_dists));
                vec_out.* = @select(Element, is_at_target, optrgt, res);
            }
        }

        pub fn eql(v_in: Vectors, w_in: Vectors, noalias out: *[vectors_per_op]bool) void {
            var res: @Vector(vectors_per_op, bool) = @splat(true);
            for (v_in, w_in) |vec1, vec2| {
                const opv1: OpVec = vec1.*;
                const opv2: OpVec = vec2.*;
                res = (res and (opv1 == opv2));
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

        fn approxEqAbsFloat(v_in: Vectors, w_in: Vectors, tolerances_in: Scalars, noalias out: *[vectors_per_op]bool) void {
            assert(@reduce(.Min, tolerances_in) >= 0.0);

            var res: @Vector(vectors_per_op, bool) = @splat(true);
            for (v_in, w_in, tolerances_in) |vec1, vec2, tolerance| {
                const opv1: OpVec = vec1.*;
                const opv2: OpVec = vec2.*;
                const optol: OpVec = tolerance.*;
                res = (res and @abs(opv1 - opv2) <= optol);
            }
            out.* = res;
        }
        fn approxEqRelFloat(v_in: Vectors, w_in: Vectors, tolerances_in: Scalars, noalias out: *[vectors_per_op]bool) void {
            assert(@reduce(.Min, tolerances_in) > 0.0);

            var res: @Vector(vectors_per_op, bool) = @splat(true);
            for (v_in, w_in, tolerances_in) |vec1, vec2, tolerance| {
                const opv1: OpVec = vec1.*;
                const opv2: OpVec = vec2.*;
                const optol: OpVec = tolerance.*;

                const rel_tolerance = (@max(@abs(opv1), @abs(opv2)) * optol);
                res = (res and @abs(opv1 - opv2) <= rel_tolerance);
            }
            out.* = res;
        }
        fn approxEqUnimpl(_: Vectors, _: Vectors, _: Scalars, _: *[vectors_per_op]bool) void {
            @panic("approxEq functions are only available for float vectors");
        }
    };
}

inline fn iota(comptime len: usize, comptime T: type) [len]T {
    var arr: [len]T = undefined;
    inline for (0..len) |i| {
        const raw = @as(comptime_int, i);
        arr[i] = raw;
    }
    return arr;
}
inline fn slices(comptime len: usize, comptime T: type, buf: *const [len]T) *const [len]*const T {
    var arr: [len]*const T = undefined;
    for (0..len) |i| {
        arr[i] = &buf[i];
    }
    return &arr;
}

test "lenghts" {
    inline for (.{ 2, 3, 4 }) |len| {
        const ns = namespace(len, f32);
        const Batch = [ns.vectors_per_op]f32;
        const buf: [len]Batch = @splat(iota(ns.vectors_per_op, f32));
        const in = slices(len, Batch, &buf);
        var out: Batch = undefined;
        ns.lengths(in, &out);

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
