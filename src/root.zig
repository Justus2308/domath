const std = @import("std");
const testing = std.testing;

const vector_import = @import("vector.zig");
pub const vector = vector_import.namespace;
pub const VectorConfig = vector_import.Config;
pub const vectorEx = vector_import.namespaceWithConfig;

pub const v2f32 = vector(2, f32);
pub const v3f32 = vector(3, f32);
pub const v4f32 = vector(4, f32);

pub const v2f64 = vector(2, f64);
pub const v3f64 = vector(3, f64);
pub const v4f64 = vector(4, f64);

pub const v2i32 = vector(2, i32);
pub const v3i32 = vector(3, i32);
pub const v4i32 = vector(4, i32);
