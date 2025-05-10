const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // MAIN MODULE

    const domath = b.addModule("domath", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const unit_tests = b.addTest(.{
        .root_module = domath,
    });

    const run_unit_tests = b.addRunArtifact(unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);

    // BENCHMARKS

    const bench = b.createModule(.{
        .root_source_file = b.path("bench/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    bench.addImport("domath", domath);

    if (b.lazyDependency("zbench", .{
        .target = target,
        .optimize = optimize,
    })) |dep| {
        bench.addImport("zbench", dep.module("zbench"));
    }
    if (b.lazyDependency("zalgebra", .{
        .target = target,
        .optimize = optimize,
    })) |dep| {
        bench.addImport("zalgebra", dep.module("zalgebra"));
    }

    const bench_exe = b.addExecutable(.{
        .name = "bench",
        .root_module = bench,
    });

    const run_bench = b.addRunArtifact(bench_exe);
    if (b.args) |args| {
        run_bench.addArgs(args);
    }

    const bench_step = b.step("bench", "Run benchmark");
    bench_step.dependOn(&run_bench.step);
}
