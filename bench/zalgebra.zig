const std = @import("std");
const root = @import("root");
const zalgebra = @import("zalgebra");

pub fn normalize4(allocator: std.mem.Allocator) void {
    _ = allocator;
    root.setMaxListCount(2);

    const list_in = root.getRandomArrayList(zalgebra.Vec4, 0);
    const list_out = root.getUndefArrayList(zalgebra.Vec4);

    for (list_in.items, list_out.items) |vec_in, *vec_out| {
        vec_out.* = vec_in.norm();
    }
}

pub fn scaleByLen3(allocator: std.mem.Allocator) void {
    _ = allocator;
    root.setMaxListCount(2);

    const list_in = root.getRandomArrayList(zalgebra.Vec3, 0);
    const list_out = root.getUndefArrayList(zalgebra.Vec3);

    for (list_in.items, list_out.items) |vec_in, *vec_out| {
        const length = vec_in.length();
        vec_out.* = vec_in.scale(length);
    }
}

pub fn multiplyAddNegate2(allocator: std.mem.Allocator) void {
    _ = allocator;
    root.setMaxListCount(4);

    const list_in1 = root.getRandomArrayList(zalgebra.Vec2, 0);
    const list_in2 = root.getRandomArrayList(zalgebra.Vec2, 100);
    const list_in3 = root.getRandomArrayList(zalgebra.Vec2, 1000);
    const list_out = root.getUndefArrayList(zalgebra.Vec2);

    for (list_in1.items, list_in2.items, list_in3.items, list_out.items) |vec_in1, vec_in2, vec_in3, *vec_out| {
        vec_out.* = vec_in1.mul(vec_in2).add(vec_in3).negate();
    }
}

pub fn cross3(allocator: std.mem.Allocator) void {
    _ = allocator;
    root.setMaxListCount(3);

    const list_in1 = root.getRandomArrayList(zalgebra.Vec3, 0);
    const list_in2 = root.getRandomArrayList(zalgebra.Vec3, 100);
    const list_out = root.getUndefArrayList(zalgebra.Vec3);

    for (list_in1.items, list_in2.items, list_out.items) |vec_in1, vec_in2, *vec_out| {
        vec_out.* = vec_in1.cross(vec_in2);
    }
}
