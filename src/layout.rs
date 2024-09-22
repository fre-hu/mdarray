use crate::mapping::{DenseMapping, Mapping, StridedMapping};
use crate::shape::Shape;

/// Array memory layout trait.
pub trait Layout {
    /// Array layout mapping type.
    type Mapping<S: Shape>: Mapping<Shape = S, Layout = Self>;

    /// True if the layout type is dense.
    const IS_DENSE: bool;
}

/// Dense array layout type.
pub struct Dense;

/// Strided array layout type.
pub struct Strided;

impl Layout for Dense {
    type Mapping<S: Shape> = DenseMapping<S>;

    const IS_DENSE: bool = true;
}

impl Layout for Strided {
    type Mapping<S: Shape> = StridedMapping<S>;

    const IS_DENSE: bool = false;
}
