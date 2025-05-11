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
    batch_size: ?comptime_int = null,
};

/// Create a namespace for vector math functions specialized on `len` and `Element` to live in.
/// Supports floats, integers and nullable pointers.
/// Expects batches of `vectors_per_op` items (depends on build target/config).
/// Note that all parameters are marked `noalias` to facilitate load/store optimizations
/// when chaining vector operations.
pub fn namespaceWithConfig(comptime len: comptime_int, comptime Element: type, comptime config: Config) type {
    verifyElemType(Element);
    return struct {
        const self = @This();

        pub const vectors_per_op = if (config.batch_size) |batch_size| blk: {
            if (batch_size <= 0) {
                @compileError("batch size needs to be at least 1");
            }
            break :blk batch_size;
        } else calcBatchSize(Element);

        const OpVec = @Vector(vectors_per_op, Element);

        pub const Scalars = [vectors_per_op]Element;
        pub const Bools = [vectors_per_op]bool;

        pub const single_dim = namespaceWithConfig(1, Element, .{ .batch_size = vectors_per_op });

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

        /// Generates an array of pointers suitable to pass as a vector batch from a `std.MultiArrayList`/`.Slice`.
        pub fn fromMultiArrayList(list_or_slice: anytype, comptime field_names: [len]@Type(.enum_literal), offset: usize) [len]*Scalars {
            var arr: [len]*Scalars = undefined;
            inline for (&arr, field_names) |*a, tag| {
                a.* = @ptrCast(list_or_slice.items(tag)[offset..][0..vectors_per_op]);
            }
            return arr;
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

        pub inline fn cross(
            noalias v_in: *const [len]*const Scalars,
            noalias w_in: *const [len]*const Scalars,
            noalias out: switch (len) {
                2 => *Scalars,
                3 => *const [len]*Scalars,
                else => *anyopaque,
            },
        ) void {
            switch (len) {
                2 => cross2(v_in, w_in, out),
                3 => cross3(v_in, w_in, out),
                else => @compileError("cross product is only implemented for len=2 and len=3"),
            }
        }
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

        pub inline fn negate(noalias in: *const [len]*const Scalars, noalias out: *const [len]*Scalars) void {
            switch (@typeInfo(Element)) {
                .float, .int => negateImpl(in, out),
                else => @compileError("cannot negate element type"),
            }
        }
        fn negateImpl(noalias in: *const [len]*const Scalars, noalias out: *const [len]*Scalars) void {
            for (in, out) |vec_in, vec_out| {
                const opvec: OpVec = vec_in.*;
                vec_out.* = -opvec;
            }
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

        pub fn min(
            noalias v_in: *const [len]*const Scalars,
            noalias w_in: *const [len]*const Scalars,
            noalias out: *const [len]*Scalars,
        ) void {
            for (v_in, w_in, out) |vec1_in, vec2_in, vec_out| {
                const opv1: OpVec = vec1_in.*;
                const opv2: OpVec = vec2_in.*;
                vec_out.* = @min(opv1, opv2);
            }
        }

        pub fn max(
            noalias v_in: *const [len]*const Scalars,
            noalias w_in: *const [len]*const Scalars,
            noalias out: *const [len]*Scalars,
        ) void {
            for (v_in, w_in, out) |vec1_in, vec2_in, vec_out| {
                const opv1: OpVec = vec1_in.*;
                const opv2: OpVec = vec2_in.*;
                vec_out.* = @max(opv1, opv2);
            }
        }

        pub inline fn sin(noalias in: *const [len]*const Scalars, noalias out: *const [len]*Scalars) void {
            switch (@typeInfo(Element)) {
                .float => sinFloat(in, out),
                else => @compileError("sin is only implemented for floats"),
            }
        }
        fn sinFloat(noalias in: *const [len]*const Scalars, noalias out: *const [len]*Scalars) void {
            for (in, out) |vec_in, vec_out| {
                const opvec: OpVec = vec_in.*;
                vec_out.* = @sin(opvec);
            }
        }

        pub inline fn cos(noalias in: *const [len]*const Scalars, noalias out: *const [len]*Scalars) void {
            switch (@typeInfo(Element)) {
                .float => cosFloat(in, out),
                else => @compileError("cos is only implemented for floats"),
            }
        }
        fn cosFloat(noalias in: *const [len]*const Scalars, noalias out: *const [len]*Scalars) void {
            for (in, out) |vec_in, vec_out| {
                const opvec: OpVec = vec_in.*;
                vec_out.* = @cos(opvec);
            }
        }

        pub inline fn abs(noalias in: *const [len]*const Scalars, noalias out: *const [len]*Scalars) void {
            switch (@typeInfo(Element)) {
                .float, .int => absImpl(in, out),
                else => @compileError("cannot take absolute value of element type"),
            }
        }
        fn absImpl(noalias in: *const [len]*const Scalars, noalias out: *const [len]*Scalars) void {
            for (in, out) |vec_in, vec_out| {
                const opvec: OpVec = vec_in.*;
                vec_out.* = @abs(opvec);
            }
        }

        // Using `@select` here to effectively perform bitwise operations is a rather
        // unfortunate result of https://github.com/ziglang/zig/issues/14306, but
        // codegen seems to be fine anyways.

        const BoolVec = @Vector(vectors_per_op, bool);

        inline fn vAnd(v1: BoolVec, v2: BoolVec) BoolVec {
            return @select(bool, v1, v2, v1);
        }
        inline fn vOr(v1: BoolVec, v2: BoolVec) BoolVec {
            return @select(bool, v1, v1, v2);
        }

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

            const is_at_target = vOr((dists == zeroes_vec), vAnd((max_dists >= zeroes_vec), (dists <= max_dists)));

            for (v_in, target_in, out) |vec_in, target, vec_out| {
                const opvec: OpVec = vec_in.*;
                const optrgt: OpVec = target.*;
                const d = (optrgt - opvec);
                const res = (opvec + ((d / dists) * max_dists));
                vec_out.* = @select(Element, is_at_target, optrgt, res);
            }
        }

        pub fn eql(
            noalias v_in: *const [len]*const Scalars,
            noalias w_in: *const [len]*const Scalars,
            noalias out: *Bools,
        ) void {
            var res: BoolVec = @splat(true);
            for (v_in, w_in) |vec1, vec2| {
                const opv1: OpVec = vec1.*;
                const opv2: OpVec = vec2.*;

                res = vAnd((opv1 == opv2), res);
            }
            out.* = res;
        }

        pub inline fn approxEqAbs(
            noalias v_in: *const [len]*const Scalars,
            noalias w_in: *const [len]*const Scalars,
            noalias tolerances: *const Scalars,
            noalias out: *Bools,
        ) void {
            switch (@typeInfo(Element)) {
                .float => approxEqAbsFloat(v_in, w_in, tolerances, out),
                else => @compileError("approxEqAbs is only available for float vectors"),
            }
        }
        fn approxEqAbsFloat(
            noalias v_in: *const [len]*const Scalars,
            noalias w_in: *const [len]*const Scalars,
            noalias tolerances: *const Scalars,
            noalias out: *Bools,
        ) void {
            const optol: OpVec = tolerances.*;
            assert(@reduce(.Min, optol) >= 0.0);

            var res: BoolVec = @splat(true);
            for (v_in, w_in) |vec1, vec2| {
                const opv1: OpVec = vec1.*;
                const opv2: OpVec = vec2.*;

                res = vAnd((@abs(opv1 - opv2) <= optol), res);
            }
            out.* = res;
        }

        pub inline fn approxEqRel(
            noalias v_in: *const [len]*const Scalars,
            noalias w_in: *const [len]*const Scalars,
            noalias tolerances: *const Scalars,
            noalias out: *Bools,
        ) void {
            switch (@typeInfo(Element)) {
                .float => approxEqRelFloat(v_in, w_in, tolerances, out),
                else => @compileError("approxEqRel is only available for float vectors"),
            }
        }
        fn approxEqRelFloat(
            noalias v_in: *const [len]*const Scalars,
            noalias w_in: *const [len]*const Scalars,
            noalias tolerances: *const Scalars,
            noalias out: *Bools,
        ) void {
            const optol: OpVec = tolerances.*;
            assert(@reduce(.Min, optol) > 0.0);

            var res: BoolVec = @splat(true);
            for (v_in, w_in) |vec1, vec2| {
                const opv1: OpVec = vec1.*;
                const opv2: OpVec = vec2.*;

                const rel_tolerance = (@max(@abs(opv1), @abs(opv2)) * optol);
                res = vAnd((@abs(opv1 - opv2) <= rel_tolerance), res);
            }
            out.* = res;
        }

        pub fn casted(comptime T: type) type {
            return namespaceWithConfig(len, T, .{ .batch_size = vectors_per_op });
        }
        pub fn cast(
            comptime T: type,
            noalias in: *const [len]*const Scalars,
            noalias out: *const [len]*casted(T).Scalars,
        ) void {
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
        pub inline fn slices(noalias ptr_to_slicable: anytype) SliceType(@TypeOf(ptr_to_slicable)) {
            var arr: SliceType(@TypeOf(ptr_to_slicable)) = undefined;
            for (&arr, ptr_to_slicable) |*a, *s| {
                a.* = s;
            }
            return arr;
        }

        fn ScalarsPtrType(comptime Slices: type) type {
            return switch (@typeInfo(Slices)) {
                .pointer => |ptr_info| switch (ptr_info.size) {
                    .one => switch (@typeInfo(ptr_info.child)) {
                        .array => |arr_info| if (arr_info.len != len) {
                            @compileError(std.fmt.comptimePrint("need array of 'Scalars' with length {d}, got {d}", .{ len, arr_info.len }));
                        } else switch (@typeInfo(arr_info.child)) {
                            .pointer => |inner_ptr_info| {
                                switch (inner_ptr_info.size) {
                                    .one => if (inner_ptr_info.child != Scalars) {
                                        @compileError("need array of 'Scalars', got '" ++ @typeName(arr_info.child) ++ "'");
                                    } else {
                                        return if (inner_ptr_info.is_const) *const Scalars else *Scalars;
                                    },
                                    else => @compileError("need pointers to single 'Scalars'"),
                                }
                            },
                            else => @compileError("need pointers to 'Scalars'"),
                        },
                        else => @compileError("need pointer to array of pointers to 'Scalars'"),
                    },
                    else => @compileError("need pointer to a single array of pointers to 'Scalars'"),
                },
                else => |info| @compileError("need a 'pointer', got '" ++ @tagName(info) ++ "'"),
            };
        }

        pub const Dimension = switch (len) {
            1 => enum(usize) { x = 0 },
            2 => enum(usize) { x = 0, y = 1 },
            3 => enum(usize) { x = 0, y = 1, z = 2 },
            4 => enum(usize) { x = 0, y = 1, z = 2, w = 3 },
            else => usize,
        };
        inline fn dimAsInt(dimension: Dimension) usize {
            if (len <= 4) {
                return @intFromEnum(dimension);
            } else {
                assert(dimension < len);
                return dimension;
            }
        }

        /// Extract a single dimension from `in`.
        pub inline fn extractDim(noalias in: anytype, dimension: Dimension) ScalarsPtrType(@TypeOf(in)) {
            const index = dimAsInt(dimension);
            return in.*[index];
        }

        /// Extract a single element from `in`.
        pub inline fn extractElem(noalias in: *const [len]*const Scalars, index: usize) [len]Element {
            assert(index < vectors_per_op);
            var elems: [len]Element = undefined;
            for (in, &elems) |vec, *elem| {
                elem.* = vec.*[index];
            }
            return elems;
        }

        pub fn swizzle(comptime layout: []const Dimension, noalias in: *const [len]*const Scalars, noalias out: *const [layout.len]*Scalars) void {
            for (out, layout) |vec_out, dim| {
                const index = dimAsInt(dim);
                vec_out.* = in.*[index].*;
            }
        }

        fn noop(noalias in: *const [len]*const Scalars, noalias out: *const [len]*Scalars) void {
            for (in, out) |vec_in, vec_out| {
                vec_out.* = vec_in.*;
            }
        }

        /// This works like an ALU with an accumulator register.
        /// The results always gets saved in an intermediary buffer and
        /// the user can either apply a transformation that works solely
        /// based off of this register or supply additional data for an
        /// operation that involves multiple inputs.
        pub const Accumulator = struct {
            buffer: Slicable,

            const Output = union(enum) {
                scalars: *Scalars,
                vectors: [len]*Scalars,
            };

            pub fn begin(
                comptime op: Op,
                noalias in: op.kind.in,
                noalias extra_args: op.kind.extra,
            ) Accumulator {
                comptime assert(op.kind.out != *Scalars);
                var accu = Accumulator{ .buffer = undefined };
                const out_raw: Output = switch (op.kind.out) {
                    *Scalars => .{ .scalars = &accu.buffer[0] },
                    *const [len]*Scalars => .{ .vectors = slices(&accu.buffer) },
                    else => comptime unreachable,
                };
                const out = switch (op.kind.out) {
                    *Scalars => out_raw.scalars,
                    *const [len]*Scalars => &out_raw.vectors,
                    else => comptime unreachable,
                };
                op.call(in, extra_args, out);
                return accu;
            }

            pub fn cont(
                noalias accu: *Accumulator,
                comptime op: Op,
                noalias extra_args: op.kind.extra,
            ) void {
                comptime assert(op.kind.out != *Scalars);
                const in = slices(&accu.buffer);
                // TODO: find prettier solution
                var tmp: Slicable = undefined;
                const out_raw: Output = switch (op.kind.out) {
                    *Scalars => .{ .scalars = &tmp[0] },
                    *const [len]*Scalars => .{ .vectors = slices(&tmp) },
                    else => comptime unreachable,
                };
                const out = switch (op.kind.out) {
                    *Scalars => out_raw.scalars,
                    *const [len]*Scalars => &out_raw.vectors,
                    else => comptime unreachable,
                };
                op.call(&in, extra_args, out);
                accu.buffer = tmp;
            }

            pub fn end(
                noalias accu: *Accumulator,
                comptime op: Op,
                noalias extra_args: op.kind.extra,
                noalias out: op.kind.out,
            ) void {
                const in = slices(&accu.buffer);
                op.call(&in, extra_args, out);
            }

            pub fn cast(noalias accu: *Accumulator, comptime T: type) casted(T).Accumulator {
                var casted_accu: casted(T).Accumulator = .{ .buffer = undefined };
                const in = slices(&accu.buffer);
                const out = casted(T).slices(&casted_accu.buffer);
                self.cast(T, &in, &out);
                return casted_accu;
            }

            pub const Op = struct {
                kind: Op.Kind,
                fn_name: [:0]const u8,

                pub const add = Op{
                    .kind = .vw_to_v,
                    .fn_name = "add",
                };
                pub const sub = Op{
                    .kind = .vw_to_v,
                    .fn_name = "sub",
                };
                pub const mul = Op{
                    .kind = .vw_to_v,
                    .fn_name = "mul",
                };
                pub const div = Op{
                    .kind = .vw_to_v,
                    .fn_name = "div",
                };
                pub const div_allow_zero = Op{
                    .kind = .vw_to_v,
                    .fn_name = "divAllowZero",
                };
                pub const mod = Op{
                    .kind = .vw_to_v,
                    .fn_name = "mod",
                };
                pub const length = Op{
                    .kind = .v_to_s,
                    .fn_name = "lengths",
                };
                pub const length_sqrd = Op{
                    .kind = .v_to_s,
                    .fn_name = "lengthsSqrd",
                };
                pub const normalize = Op{
                    .kind = .v_to_v,
                    .fn_name = "normalize",
                };
                pub const scale = Op{
                    .kind = .vs_to_v,
                    .fn_name = "scale",
                };
                pub const dot = Op{
                    .kind = .vw_to_s,
                    .fn_name = "dots",
                };
                pub const cross = Op{
                    .kind = switch (len) {
                        2 => .vw_to_s,
                        3 => .vw_to_v,
                        else => undefined,
                    },
                    .fn_name = "cross",
                };
                pub const distance = Op{
                    .kind = .vw_to_s,
                    .fn_name = "distances",
                };
                pub const distance_sqrd = Op{
                    .kind = .vw_to_s,
                    .fn_name = "distancesSqrd",
                };
                pub const invert = Op{
                    .kind = .v_to_v,
                    .fn_name = "invert",
                };
                pub const negate = Op{
                    .kind = .v_to_v,
                    .fn_name = "negate",
                };
                pub const min = Op{
                    .kind = .vw_to_v,
                    .fn_name = "min",
                };
                pub const max = Op{
                    .kind = .vw_to_v,
                    .fn_name = "max",
                };
                pub const clamp = Op{
                    .kind = .uvw_to_v,
                    .fn_name = "clamp",
                };
                pub const sin = Op{
                    .kind = .v_to_v,
                    .fn_name = "sin",
                };
                pub const cos = Op{
                    .kind = .v_to_v,
                    .fn_name = "cos",
                };
                pub const move_towards = Op{
                    .kind = .vws_to_v,
                    .fn_name = "moveTowards",
                };

                /// This just loads the input into the `Accumulator` buffer,
                /// if you feel the need to use this you probably don't need
                ///  an `Accumulator` in the first place.
                /// Mainly for testing purposes.
                pub const noop = Op{
                    .kind = .v_to_v,
                    .fn_name = "noop",
                };

                const Kind = struct {
                    in: type,
                    extra: type,
                    out: type,

                    pub const v_to_v = Kind{
                        .in = *const [len]*const Scalars,
                        .extra = std.meta.Tuple(&.{}),
                        .out = *const [len]*Scalars,
                    };
                    pub const v_to_s = Kind{
                        .in = *const [len]*const Scalars,
                        .extra = std.meta.Tuple(&.{}),
                        .out = *Scalars,
                    };
                    pub const vw_to_v = Kind{
                        .in = *const [len]*const Scalars,
                        .extra = std.meta.Tuple(&.{*const [len]*const Scalars}),
                        .out = *const [len]*Scalars,
                    };
                    pub const vw_to_s = Kind{
                        .in = *const [len]*const Scalars,
                        .extra = std.meta.Tuple(&.{*const [len]*const Scalars}),
                        .out = *Scalars,
                    };
                    pub const vs_to_v = Kind{
                        .in = *const [len]*const Scalars,
                        .extra = std.meta.Tuple(&.{*const Scalars}),
                        .out = *const [len]*Scalars,
                    };
                    pub const vws_to_v = Kind{
                        .in = *const [len]*const Scalars,
                        .extra = std.meta.Tuple(&.{ *const [len]*const Scalars, *const Scalars }),
                        .out = *const [len]*Scalars,
                    };
                    pub const uvw_to_v = Kind{
                        .in = *const [len]*const Scalars,
                        .extra = std.meta.Tuple(&.{ *const [len]*const Scalars, *const [len]*const Scalars }),
                        .out = *const [len]*Scalars,
                    };
                };

                fn call(comptime op: Op, in: op.kind.in, extra_args: op.kind.extra, out: op.kind.out) void {
                    @call(.auto, @field(self, op.fn_name), .{in} ++ extra_args ++ .{out});
                }
            };
        };
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
                try testing.expectApproxEqRel(a, b, std.math.floatEps(f32));
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

test "min/max" {
    inline for (.{ 2, 3, 4 }) |len| {
        const ns = namespace(len, f32);
        const buf1: ns.Slicable = @splat(iota(ns.vectors_per_op, f32, 0, 2));
        const in1 = ns.slices(&buf1);
        const buf2: ns.Slicable = @splat(iota(ns.vectors_per_op, f32, 2, 1));
        const in2 = ns.slices(&buf2);
        var out_buf_min: ns.Slicable = undefined;
        const out_min = ns.slices(&out_buf_min);
        var out_buf_max: ns.Slicable = undefined;
        const out_max = ns.slices(&out_buf_max);

        ns.min(&in1, &in2, &out_min);
        ns.max(&in1, &in2, &out_max);

        for (in1, in2, out_min, out_max) |vec1, vec2, vecmin, vecmax| {
            for (vec1, vec2, vecmin, vecmax) |a, b, rmin, rmax| {
                try testing.expectEqual(@min(a, b), rmin);
                try testing.expectEqual(@max(a, b), rmax);
            }
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
        var out: ns.Bools = undefined;
        ns.eql(&in1, &in2, &out);

        for (0..(ns.vectors_per_op / 2)) |i| {
            try testing.expectEqual(true, out[i]);
        }
        for ((ns.vectors_per_op / 2)..ns.vectors_per_op) |i| {
            try testing.expectEqual(false, out[i]);
        }
    }
}

test "approx eql" {
    // Checks for the infamous 0.1 + 0.2 != 0.3
    inline for (.{ 2, 3, 4 }) |len| {
        const ns = namespace(len, f64);
        const buf1: ns.Slicable = @splat(ns.splatScalar(0.1));
        const in1 = ns.slices(&buf1);
        const buf2: ns.Slicable = @splat(ns.splatScalar(0.2));
        const in2 = ns.slices(&buf2);
        const expected_buf: ns.Slicable = @splat(ns.splatScalar(0.3));
        const expected = ns.slices(&expected_buf);
        var buf_out: ns.Slicable = undefined;
        const out = ns.slices(&buf_out);
        const tols: ns.Scalars = @splat(std.math.floatEps(f64));
        var exact: ns.Bools = undefined;
        var approx_abs: ns.Bools = undefined;
        var approx_rel: ns.Bools = undefined;

        ns.add(&in1, &in2, &out);

        ns.eql(&out, &expected, &exact);
        ns.approxEqAbs(&out, &expected, &tols, &approx_abs);
        ns.approxEqRel(&out, &expected, &tols, &approx_rel);

        for (exact, approx_abs, approx_rel) |ex, ab, rl| {
            try testing.expectEqual(false, ex);
            try testing.expectEqual(true, ab);
            try testing.expectEqual(true, rl);
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

test "extract" {
    inline for (.{ 2, 3, 4 }) |len| {
        const ns = namespace(len, f32);

        const buf1: ns.Slicable = @splat(iota(ns.vectors_per_op, f32, 0, 1));
        const in1 = ns.slices(&buf1);

        const dim2 = ns.extractDim(&in1, .y);
        for (dim2, &buf1[1]) |*extr, *orig| {
            try testing.expectEqual(orig, extr);
        }

        var buf2: ns.Slicable = @splat(iota(ns.vectors_per_op, f32, 0, 1));
        const in2 = ns.slices(&buf2);

        const dim1 = ns.extractDim(&in2, .x);
        for (dim1, &buf2[0]) |*extr, *orig| {
            try testing.expectEqual(orig, extr);
        }

        const elem1 = ns.extractElem(&in1, (ns.vectors_per_op - 1));
        for (elem1) |e| {
            try testing.expectEqual((ns.vectors_per_op - 1), e);
        }
    }
}

test "swizzle" {
    const ns = namespace(4, usize);

    const in_buf: ns.Slicable = .{
        ns.splatScalar(0),
        ns.splatScalar(1),
        ns.splatScalar(2),
        ns.splatScalar(3),
    };
    const in = ns.slices(&in_buf);

    inline for (.{
        [2]ns.Dimension{ .z, .x },
        [4]ns.Dimension{ .z, .z, .y, .w },
        [6]ns.Dimension{ .w, .y, .x, .z, .w, .x },
    }) |layout| {
        const tgt_ns = namespace(layout.len, usize);
        var out_buf: tgt_ns.Slicable = undefined;
        const out = tgt_ns.slices(&out_buf);

        ns.swizzle(&layout, &in, &out);

        for (out, layout) |vec, expected| {
            for (vec) |elem| {
                try testing.expectEqual(ns.dimAsInt(expected), elem);
            }
        }
    }
}

test "slices from MultiArrayList/Slice" {
    inline for (.{ 2, 3, 4 }) |len| {
        const ns = namespace(len, usize);
        var list = std.MultiArrayList(struct { a: usize, b: usize, c: usize, d: usize }).empty;
        defer list.deinit(testing.allocator);

        const elem_count = (10 * ns.vectors_per_op);

        try list.setCapacity(testing.allocator, elem_count);
        for (0..elem_count) |i| {
            list.appendAssumeCapacity(.{
                .a = i,
                .b = (2 * i),
                .c = (3 * i),
                .d = (4 * i),
            });
        }

        var offset: usize = 0;
        while (offset < elem_count) : (offset += ns.vectors_per_op) {
            const in = ns.fromMultiArrayList(list.slice(), ([_]@Type(.enum_literal){ .a, .b, .c, .d })[0..len].*, offset);
            for (in, 1..) |vec, c| {
                for (vec, offset..) |elem, i| {
                    try testing.expectEqual((c * i), elem);
                }
            }
        }
    }
}

test "Accumulator: basic usage" {
    inline for (.{ 2, 3, 4 }) |len| {
        const ns = namespace(len, i32);

        const buf1: ns.Slicable = @splat(iota(ns.vectors_per_op, i32, 0, 1));
        const in1 = ns.slices(&buf1);
        const buf2: ns.Slicable = @splat(iota(ns.vectors_per_op, i32, 10, 1));
        const in2 = ns.slices(&buf2);
        const buf3: ns.Slicable = @splat(iota(ns.vectors_per_op, i32, 5, 2));
        const in3 = ns.slices(&buf3);
        const factors: ns.Scalars = @splat(2);
        var out_buf: ns.Slicable = undefined;
        const out = ns.slices(&out_buf);

        var accu = ns.Accumulator.begin(.add, &in1, .{&in2});
        accu.cont(.sub, .{&in3});
        accu.cont(.noop, .{});
        accu.end(.scale, .{&factors}, &out);

        for (out) |vec| {
            for (vec, 0..) |elem, idx| {
                const i: i32 = @intCast(idx);
                var expected = i;
                expected += (10 + i);
                expected -= (5 + (2 * i));
                expected *= 2;
                try testing.expectEqual(expected, elem);
            }
        }
    }
}

test "Accumulator: casting" {
    inline for (.{ 2, 3, 4 }) |len| {
        const ns = namespace(len, f32);

        const buf: ns.Slicable = @splat(iota(ns.vectors_per_op, f32, 0, 1));
        const in = ns.slices(&buf);

        var accu = ns.Accumulator.begin(.noop, &in, .{});
        const accu_casted = accu.cast(u32);
        for (accu_casted.buffer) |vec| {
            for (vec, 0..) |elem, i| {
                try testing.expectEqual(elem, @as(u32, @intCast(i)));
            }
        }
    }
}

test "Accumulator: multiple extra args" {
    inline for (.{ 2, 3, 4 }) |len| {
        const ns = namespace(len, f32);

        const buf1: ns.Slicable = @splat(iota(ns.vectors_per_op, f32, 10, 2));
        const in1 = ns.slices(&buf1);
        const buf2: ns.Slicable = @splat(iota(ns.vectors_per_op, f32, 0, 1));
        const in2 = ns.slices(&buf2);

        const cmin_val = 12.0;
        const cmin_buf: ns.Slicable = @splat(@splat(cmin_val));
        const cmin = ns.slices(&cmin_buf);

        const cmax_val = 30.0;
        const cmax_buf: ns.Slicable = @splat(@splat(cmax_val));
        const cmax = ns.slices(&cmax_buf);

        const tgt_buf: ns.Slicable = @splat(iota(ns.vectors_per_op, f32, 1000, 300));
        const tgt = ns.slices(&tgt_buf);
        const maxd: ns.Scalars = @splat(5.0);
        var out_buf: ns.Slicable = undefined;
        const out = ns.slices(&out_buf);

        var accu = ns.Accumulator.begin(.add, &in1, .{&in2});
        accu.cont(.clamp, .{ &cmin, &cmax });
        accu.end(.move_towards, .{ &tgt, &maxd }, &out);

        for (0..ns.vectors_per_op) |j| {
            var dist_sqrd: f32 = 0;
            for (0..len) |i| {
                const start = std.math.clamp((buf1[i][j] + buf2[i][j]), cmin_val, cmax_val);
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
                const start = std.math.clamp((buf1[i][j] + buf2[i][j]), cmin_val, cmax_val);
                const target = tgt_buf[i][j];
                const d = (target - start);
                const expected = (start + ((d / dist) * 5.0));
                try testing.expectEqual(expected, out_buf[i][j]);
            }
        }
    }
}
