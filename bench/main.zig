const std = @import("std");
const zbench = @import("zbench");

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
    memory = .init(allocator);
    defer memory.deinit(allocator);

    var bench = zbench.Benchmark.init(allocator, .{
        .time_budget_ns = 5e9,
        .hooks = .{
            .after_all = &hookResetMemory,
            .after_each = &hookResetMemory,
        },
    });
    defer bench.deinit();

    inline for (comptime std.meta.declarations(candidates)) |decl| {
        const candidate = @field(candidates, decl.name);
        try bench.add(decl.name ++ ": " ++ name ++ "", @field(candidate, name), .{});
    }

    try bench.run(writer);
}

fn hookResetMemory() void {
    memory.reset();
}

const vec_count = (1 << 20);

pub inline fn setMaxListCount(count: usize) void {
    memory.list_len = (@as(usize, vec_count) >> @intCast(std.math.log2_int_ceil(usize, count)));
}

pub inline fn getListLen() usize {
    return memory.list_len;
}

var memory: struct {
    bytes: []u8,
    fba: std.heap.FixedBufferAllocator,
    list_len: usize,

    pub fn init(allocator_: std.mem.Allocator) @This() {
        // very hacky but whatever, this ensures that there's always enough memory.
        const TMax = extern struct { x: f32, y: f32, z: f32, w: f32 };
        const size = (@max(std.MultiArrayList(TMax).capacityInBytes(vec_count), (@sizeOf(TMax) * vec_count)) + std.heap.page_size_max);
        const alignment = comptime std.mem.Alignment.max(.of(TMax), .fromByteUnits(std.atomic.cache_line));
        const bytes = allocator_.alignedAlloc(u8, alignment, size) catch @panic("OOM");

        return .{
            .bytes = bytes,
            .fba = .init(bytes),
            .list_len = vec_count,
        };
    }
    pub fn reset(self: *@This()) void {
        self.fba.reset();
    }
    pub fn deinit(self: *@This(), allocator_: std.mem.Allocator) void {
        const alignment = comptime std.mem.Alignment.max(.of(@Vector(4, f32)), .fromByteUnits(std.atomic.cache_line));
        allocator_.free(@as([]align(alignment.toByteUnits()) u8, @alignCast(self.bytes)));
        self.* = undefined;
    }

    pub fn allocator(self: *@This()) std.mem.Allocator {
        return self.fba.allocator();
    }
} = undefined;

pub noinline fn getRandomArrayList(comptime T: type, seed: u64) std.ArrayListUnmanaged(T) {
    const list = getUndefArrayList(T);
    var rand = std.Random.DefaultPrng.init(seed);
    rand.fill(std.mem.sliceAsBytes(list.items));
    return list;
}

pub noinline fn getUndefArrayList(comptime T: type) std.ArrayListUnmanaged(T) {
    var list = std.ArrayListUnmanaged(T).initCapacity(memory.allocator(), memory.list_len) catch unreachable;
    list.expandToCapacity();
    return list;
}

pub noinline fn getRandomMultiArrayList(comptime T: type, seed: u64) std.MultiArrayList(T) {
    const list = getUndefMultiArrayList(T);
    var rand = std.Random.DefaultPrng.init(seed);
    rand.fill(list.bytes[0..@TypeOf(list).capacityInBytes(memory.list_len)]);
    return list;
}

pub noinline fn getUndefMultiArrayList(comptime T: type) std.MultiArrayList(T) {
    var list = std.MultiArrayList(T).empty;
    list.setCapacity(memory.allocator(), memory.list_len) catch unreachable;
    list.len = memory.list_len;
    return list;
}
