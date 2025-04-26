const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const module = b.addModule("domath", .{
        .root_source_file = "src/root.zig",
        .target = target,
        .optimize = optimize,
    });

    const unit_tests = b.addTest(.{
        .root_module = module,
    });

    const run_unit_tests = b.addRunArtifact(unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);
}
