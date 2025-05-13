const std = @import("std");
const root = @import("root");
const zm = @import("zm");

pub fn normalize4(allocator: std.mem.Allocator) void {
    _ = allocator;
    root.setMaxListCount(2);

    const list_in = root.getRandomArrayList(zm.Vec4f, 0);
    const list_out = root.getUndefArrayList(zm.Vec4f);

    for (list_in.items, list_out.items) |vec_in, *vec_out| {
        vec_out.* = zm.vec.normalize(vec_in);
    }
}

pub fn scaleByLen3(allocator: std.mem.Allocator) void {
    _ = allocator;
    root.setMaxListCount(2);

    const list_in = root.getRandomArrayList(zm.Vec3f, 0);
    const list_out = root.getUndefArrayList(zm.Vec3f);

    for (list_in.items, list_out.items) |vec_in, *vec_out| {
        const len = zm.vec.len(vec_in);
        vec_out.* = zm.vec.scale(vec_in, len);
    }
}

pub fn multiplyAddNegate2(allocator: std.mem.Allocator) void {
    _ = allocator;
    root.setMaxListCount(4);

    const list_in1 = root.getRandomArrayList(zm.Vec2f, 0);
    const list_in2 = root.getRandomArrayList(zm.Vec2f, 100);
    const list_in3 = root.getRandomArrayList(zm.Vec2f, 1000);
    const list_out = root.getUndefArrayList(zm.Vec2f);

    for (list_in1.items, list_in2.items, list_in3.items, list_out.items) |vec_in1, vec_in2, vec_in3, *vec_out| {
        vec_out.* = -((vec_in1 * vec_in2) + vec_in3);
    }
}

pub fn cross3(allocator: std.mem.Allocator) void {
    _ = allocator;
    root.setMaxListCount(3);

    const list_in1 = root.getRandomArrayList(zm.Vec3f, 0);
    const list_in2 = root.getRandomArrayList(zm.Vec3f, 100);
    const list_out = root.getUndefArrayList(zm.Vec3f);

    for (list_in1.items, list_in2.items, list_out.items) |vec_in1, vec_in2, *vec_out| {
        vec_out.* = zm.vec.cross(vec_in1, vec_in2);
    }
}
