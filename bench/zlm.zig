const std = @import("std");
const root = @import("root");
const zlm = @import("zlm");

const vec_count = root.vec_count;

pub fn normalize4(allocator: std.mem.Allocator) void {
    var list_in = root.getRandomArrayList(zlm.Vec4, allocator, 0);
    defer list_in.deinit(allocator);

    var list_out = root.getUndefArrayList(zlm.Vec4, allocator);
    defer list_out.deinit(allocator);

    for (list_in.items, list_out.items) |vec_in, *vec_out| {
        vec_out.* = vec_in.normalize();
    }
}

pub fn scaleByLen3(allocator: std.mem.Allocator) void {
    var list_in = root.getRandomArrayList(zlm.Vec3, allocator, 0);
    defer list_in.deinit(allocator);

    var list_out = root.getUndefArrayList(zlm.Vec3, allocator);
    defer list_out.deinit(allocator);

    for (list_in.items, list_out.items) |vec_in, *vec_out| {
        const len = vec_in.length();
        vec_out.* = vec_in.scale(len);
    }
}

pub fn multiplyAddNegate2(allocator: std.mem.Allocator) void {
    var list_in1 = root.getRandomArrayList(zlm.Vec2, allocator, 0);
    defer list_in1.deinit(allocator);

    var list_in2 = root.getRandomArrayList(zlm.Vec2, allocator, 100);
    defer list_in2.deinit(allocator);

    var list_in3 = root.getRandomArrayList(zlm.Vec2, allocator, 1000);
    defer list_in3.deinit(allocator);

    var list_out = root.getUndefArrayList(zlm.Vec2, allocator);
    defer list_out.deinit(allocator);

    for (list_in1.items, list_in2.items, list_in3.items, list_out.items) |vec_in1, vec_in2, vec_in3, *vec_out| {
        vec_out.* = vec_in1.mul(vec_in2).add(vec_in3).neg();
    }
}

pub fn crossLerp3(allocator: std.mem.Allocator) void {
    var list_in1 = root.getRandomArrayList(zlm.Vec3, allocator, 0);
    defer list_in1.deinit(allocator);

    var list_in2 = root.getRandomArrayList(zlm.Vec3, allocator, 100);
    defer list_in2.deinit(allocator);

    var list_in3 = root.getRandomArrayList(zlm.Vec3, allocator, 1000);
    defer list_in3.deinit(allocator);

    var list_out = root.getUndefArrayList(zlm.Vec3, allocator);
    defer list_out.deinit(allocator);

    for (list_in1.items, list_in2.items, list_in3.items, list_out.items) |vec_in1, vec_in2, vec_in3, *vec_out| {
        vec_out.* = vec_in1.cross(vec_in2).lerp(vec_in3, 2.0);
    }
}
