const std = @import("std");
const root = @import("root");
const domath = @import("domath");

pub fn normalize4(allocator: std.mem.Allocator) void {
    _ = allocator;
    root.setMaxListCount(2);

    const v4 = domath.v4f32;
    const V = struct { x: f32, y: f32, z: f32, w: f32 };

    var list_in = root.getRandomMultiArrayList(V, 0);
    var list_out = root.getUndefMultiArrayList(V);

    const slice_in = list_in.slice();
    const slice_out = list_out.slice();

    var offset: usize = 0;
    while (offset < root.getListLen()) : (offset += v4.vectors_per_op) {
        const in = v4.fromMultiArrayList(slice_in, .{ .x, .y, .z, .w }, offset);
        const out = v4.fromMultiArrayList(slice_out, .{ .x, .y, .z, .w }, offset);
        v4.normalize(&in, &out);
    }
}

pub fn scaleByLen3(allocator: std.mem.Allocator) void {
    _ = allocator;
    root.setMaxListCount(2);

    const v3 = domath.v3f32;
    const V = struct { x: f32, y: f32, z: f32 };

    var list_in = root.getRandomMultiArrayList(V, 0);
    var list_out = root.getUndefMultiArrayList(V);

    const slice_in = list_in.slice();
    const slice_out = list_out.slice();
    var lens: v3.Scalars = undefined;

    var offset: usize = 0;
    while (offset < root.getListLen()) : (offset += v3.vectors_per_op) {
        const in = v3.fromMultiArrayList(slice_in, .{ .x, .y, .z }, offset);
        const out = v3.fromMultiArrayList(slice_out, .{ .x, .y, .z }, offset);

        v3.lengths(&in, &lens);
        v3.scale(&in, &lens, &out);
    }
}

pub fn multiplyAddNegate2(allocator: std.mem.Allocator) void {
    _ = allocator;
    root.setMaxListCount(4);

    const v2 = domath.v2f32;
    const V = struct { x: f32, y: f32 };

    var list_in1 = root.getRandomMultiArrayList(V, 0);
    var list_in2 = root.getRandomMultiArrayList(V, 100);
    var list_in3 = root.getRandomMultiArrayList(V, 1000);
    var list_out = root.getUndefMultiArrayList(V);

    const slice_in1 = list_in1.slice();
    const slice_in2 = list_in2.slice();
    const slice_in3 = list_in3.slice();
    const slice_out = list_out.slice();

    var offset: usize = 0;
    while (offset < root.getListLen()) : (offset += v2.vectors_per_op) {
        const in1 = v2.fromMultiArrayList(slice_in1, .{ .x, .y }, offset);
        const in2 = v2.fromMultiArrayList(slice_in2, .{ .x, .y }, offset);
        const in3 = v2.fromMultiArrayList(slice_in3, .{ .x, .y }, offset);
        const out = v2.fromMultiArrayList(slice_out, .{ .x, .y }, offset);

        var accu = v2.Accumulator.begin(.mul, &in1, .{&in2});
        accu.cont(.add, .{&in3});
        accu.end(.negate, .{}, &out);
    }
}

pub fn cross3(allocator: std.mem.Allocator) void {
    _ = allocator;
    root.setMaxListCount(3);

    const v3 = domath.v3f32;
    const V = struct { x: f32, y: f32, z: f32 };

    var list_in1 = root.getRandomMultiArrayList(V, 0);
    var list_in2 = root.getRandomMultiArrayList(V, 100);
    var list_out = root.getUndefMultiArrayList(V);

    const slice_in1 = list_in1.slice();
    const slice_in2 = list_in2.slice();
    const slice_out = list_out.slice();

    var offset: usize = 0;
    while (offset < root.getListLen()) : (offset += v3.vectors_per_op) {
        const in1 = v3.fromMultiArrayList(slice_in1, .{ .x, .y, .z }, offset);
        const in2 = v3.fromMultiArrayList(slice_in2, .{ .x, .y, .z }, offset);
        const out = v3.fromMultiArrayList(slice_out, .{ .x, .y, .z }, offset);

        v3.cross(&in1, &in2, &out);
    }
}
