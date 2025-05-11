const std = @import("std");
const root = @import("root");
const zm = @import("zm");

const vec_count = root.vec_count;

pub fn normalize4(allocator: std.mem.Allocator) void {
    var list_in = root.getRandomArrayList(zm.Vec4f, allocator, 0);
    defer list_in.deinit(allocator);

    var list_out = root.getUndefArrayList(zm.Vec4f, allocator);
    defer list_out.deinit(allocator);

    for (list_in.items, list_out.items) |vec_in, *vec_out| {
        vec_out.* = zm.vec.normalize(vec_in);
    }
}

pub fn scaleByLen3(allocator: std.mem.Allocator) void {
    var list_in = root.getRandomArrayList(zm.Vec3f, allocator, 0);
    defer list_in.deinit(allocator);

    var list_out = root.getUndefArrayList(zm.Vec3f, allocator);
    defer list_out.deinit(allocator);

    for (list_in.items, list_out.items) |vec_in, *vec_out| {
        const len = zm.vec.len(vec_in);
        vec_out.* = zm.vec.scale(vec_in, len);
    }
}

pub fn multiplyAddNegate2(allocator: std.mem.Allocator) void {
    var list_in1 = root.getRandomArrayList(zm.Vec2f, allocator, 0);
    defer list_in1.deinit(allocator);

    var list_in2 = root.getRandomArrayList(zm.Vec2f, allocator, 100);
    defer list_in2.deinit(allocator);

    var list_in3 = root.getRandomArrayList(zm.Vec2f, allocator, 1000);
    defer list_in3.deinit(allocator);

    var list_out = root.getUndefArrayList(zm.Vec2f, allocator);
    defer list_out.deinit(allocator);

    for (list_in1.items, list_in2.items, list_in3.items, list_out.items) |vec_in1, vec_in2, vec_in3, *vec_out| {
        vec_out.* = -((vec_in1 * vec_in2) + vec_in3);
    }
}

pub fn cross3(allocator: std.mem.Allocator) void {
    var list_in1 = root.getRandomArrayList(zm.Vec3f, allocator, 0);
    defer list_in1.deinit(allocator);

    var list_in2 = root.getRandomArrayList(zm.Vec3f, allocator, 100);
    defer list_in2.deinit(allocator);

    var list_out = root.getUndefArrayList(zm.Vec3f, allocator);
    defer list_out.deinit(allocator);

    for (list_in1.items, list_in2.items, list_out.items) |vec_in1, vec_in2, *vec_out| {
        vec_out.* = zm.vec.cross(vec_in1, vec_in2);
    }
}
