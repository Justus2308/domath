const std = @import("std");
const zbench = @import("zbench");

pub const vec_count = (1 << 20);

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const writer = std.io.getStdOut().writer();

    try runBenchmarks("normalize4", allocator, writer);
    try runBenchmarks("scaleByLen3", allocator, writer);
    try runBenchmarks("multiplyAddNegate2", allocator, writer);
    try runBenchmarks("cross3", allocator, writer);
}

const candidates = struct {
    pub const domath = @import("domath.zig");
    pub const zalgebra = @import("zalgebra.zig");
    pub const zm = @import("zm.zig");
    pub const zlm = @import("zlm.zig");
};

fn runBenchmarks(comptime name: [:0]const u8, allocator: std.mem.Allocator, writer: anytype) !void {
    var bench = zbench.Benchmark.init(allocator, .{ .time_budget_ns = 5e9 });
    defer bench.deinit();

    inline for (comptime std.meta.declarations(candidates)) |decl| {
        const candidate = @field(candidates, decl.name);
        try bench.add(decl.name ++ ": " ++ name ++ "", @field(candidate, name), .{});
    }

    try bench.run(writer);
}

pub noinline fn getRandomArrayList(comptime T: type, allocator: std.mem.Allocator, seed: u64) std.ArrayListUnmanaged(T) {
    const list = getUndefArrayList(T, allocator);
    var rand = std.Random.DefaultPrng.init(seed);
    rand.fill(std.mem.sliceAsBytes(list.items));
    return list;
}

pub noinline fn getUndefArrayList(comptime T: type, allocator: std.mem.Allocator) std.ArrayListUnmanaged(T) {
    var list = std.ArrayListUnmanaged(T).initCapacity(allocator, vec_count) catch @panic("OOM");
    list.expandToCapacity();
    return list;
}

pub noinline fn getRandomMultiArrayList(comptime T: type, allocator: std.mem.Allocator, seed: u64) std.MultiArrayList(T) {
    const list = getUndefMultiArrayList(T, allocator);
    var rand = std.Random.DefaultPrng.init(seed);
    rand.fill(list.bytes[0..@TypeOf(list).capacityInBytes(vec_count)]);
    return list;
}

pub noinline fn getUndefMultiArrayList(comptime T: type, allocator: std.mem.Allocator) std.MultiArrayList(T) {
    var list = std.MultiArrayList(T).empty;
    list.setCapacity(allocator, vec_count) catch @panic("OOM");
    list.len = vec_count;
    return list;
}
