const std = @import("std");
const domath = @import("domath");
const zbench = @import("zbench");
const zalgebra = @import("zalgebra");
const zm = @import("zm");
const zlm = @import("zlm");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    var bench = zbench.Benchmark.init(allocator, .{ .time_budget_ns = 5e9 });
    defer bench.deinit();

    try bench.add("domath: basic ops", benchDomathBasic, .{});
    try bench.add("zalgebra: basic ops", benchZalgebraBasic, .{});
    try bench.add("zm: basic ops", benchZmBasic, .{});
    try bench.add("zlm: basic ops", benchZlmBasic, .{});

    try bench.run(std.io.getStdOut().writer());
}

const basic_vec_count = (1 << 20);

fn benchDomathBasic(allocator: std.mem.Allocator) void {
    const v4 = domath.v4f32;
    const List = std.MultiArrayList(struct { x: f32, y: f32, z: f32, w: f32 });

    var list_in = List.empty;
    defer list_in.deinit(allocator);
    list_in.setCapacity(allocator, basic_vec_count) catch @panic("OOM");
    list_in.len = basic_vec_count;

    var list_out = List.empty;
    defer list_out.deinit(allocator);
    list_out.setCapacity(allocator, basic_vec_count) catch @panic("OOM");
    list_out.len = basic_vec_count;

    var rand = std.Random.DefaultPrng.init(0);
    rand.fill(list_in.bytes[0..List.capacityInBytes(basic_vec_count)]);

    const slice_in = list_in.slice();
    const slice_out = list_out.slice();

    var offset: usize = 0;
    while (offset < basic_vec_count) : (offset += v4.vectors_per_op) {
        const in = v4.fromMultiArrayList(slice_in, .{ .x, .y, .z, .w }, offset);
        const out = v4.fromMultiArrayList(slice_out, .{ .x, .y, .z, .w }, offset);
        v4.normalize(&in, &out);
    }
}

fn benchZalgebraBasic(allocator: std.mem.Allocator) void {
    const List = std.ArrayListUnmanaged(zalgebra.Vec4);

    var list_in = List.initCapacity(allocator, basic_vec_count) catch @panic("OOM");
    defer list_in.deinit(allocator);
    list_in.expandToCapacity();

    var list_out = List.initCapacity(allocator, basic_vec_count) catch @panic("OOM");
    defer list_out.deinit(allocator);
    list_out.expandToCapacity();

    var rand = std.Random.DefaultPrng.init(0);
    rand.fill(std.mem.sliceAsBytes(list_in.items));

    for (list_in.items, list_out.items) |vec_in, *vec_out| {
        vec_out.* = vec_in.norm();
    }
}

fn benchZmBasic(allocator: std.mem.Allocator) void {
    const List = std.ArrayListUnmanaged(zm.Vec4);

    var list_in = List.initCapacity(allocator, basic_vec_count) catch @panic("OOM");
    defer list_in.deinit(allocator);
    list_in.expandToCapacity();

    var list_out = List.initCapacity(allocator, basic_vec_count) catch @panic("OOM");
    defer list_out.deinit(allocator);
    list_out.expandToCapacity();

    var rand = std.Random.DefaultPrng.init(0);
    rand.fill(std.mem.sliceAsBytes(list_in.items));

    for (list_in.items, list_out.items) |vec_in, *vec_out| {
        vec_out.* = zm.vec.normalize(vec_in);
    }
}

fn benchZlmBasic(allocator: std.mem.Allocator) void {
    const List = std.ArrayListUnmanaged(zlm.Vec4);

    var list_in = List.initCapacity(allocator, basic_vec_count) catch @panic("OOM");
    defer list_in.deinit(allocator);
    list_in.expandToCapacity();

    var list_out = List.initCapacity(allocator, basic_vec_count) catch @panic("OOM");
    defer list_out.deinit(allocator);
    list_out.expandToCapacity();

    var rand = std.Random.DefaultPrng.init(0);
    rand.fill(std.mem.sliceAsBytes(list_in.items));

    for (list_in.items, list_out.items) |vec_in, *vec_out| {
        vec_out.* = vec_in.normalize();
    }
}
