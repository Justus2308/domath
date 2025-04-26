const std = @import("std");
const testing = std.testing;

pub const vector = @import("vector.zig").namespace;

pub const v2f32 = vector(2, f32);
pub const v3f32 = vector(3, f32);
pub const v4f32 = vector(4, f32);

pub const v2f64 = vector(2, f64);
pub const v3f64 = vector(3, f64);
pub const v4f64 = vector(4, f64);

pub const v2i32 = vector(2, i32);
pub const v3i32 = vector(3, i32);
pub const v4i32 = vector(4, i32);
