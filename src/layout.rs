use crate::mapping::{DenseMapping, FlatMapping, GeneralMapping, Mapping, StridedMapping};
use crate::shape::Shape;

/// Array memory layout trait.
pub trait Layout {
    /// Corresponding layout which may have non-uniform stride.
    type NonUniform: Layout;

    /// Corresponding layout which may have non-unit inner stride.
    type NonUnitStrided: Layout;

    /// Corresponding layout with uniform stride.
    type Uniform: Uniform;

    /// Corresponding layout with unit inner stride.
    type UnitStrided: UnitStrided;

    /// Combined layout, which has uniform strides if both inputs have uniform
    /// strides, and is unit strided if both inputs are unit strided.
    type Merge<L: Layout>: Layout;

    /// Array layout mapping type.
    type Mapping<S: Shape>: Mapping<Shape = S, Layout = Self>;

    /// True if the layout type has uniform stride.
    const IS_UNIFORM: bool;

    /// True if the layout type has unit inner stride.
    const IS_UNIT_STRIDED: bool;
}

/// Trait for layout types with uniform stride.
pub trait Uniform: Layout<Uniform = Self, UnitStrided = Dense, NonUnitStrided = Flat> {}

/// Trait for layout types with unit inner stride.
pub trait UnitStrided: Layout<Uniform = Dense, UnitStrided = Self, NonUniform = General> {}

/// Dense array layout type.
pub struct Dense;

/// Flat array layout type.
pub struct Flat;

/// General array layout type.
pub struct General;

/// Strided array layout type.
pub struct Strided;

impl Layout for Dense {
    type NonUniform = General;
    type NonUnitStrided = Flat;
    type Uniform = Self;
    type UnitStrided = Self;

    type Merge<L: Layout> = L;
    type Mapping<S: Shape> = DenseMapping<S>;

    const IS_UNIFORM: bool = true;
    const IS_UNIT_STRIDED: bool = true;
}

impl Layout for Flat {
    type NonUniform = Strided;
    type NonUnitStrided = Self;
    type Uniform = Self;
    type UnitStrided = Dense;

    type Merge<L: Layout> = L::NonUnitStrided;
    type Mapping<S: Shape> = FlatMapping<S>;

    const IS_UNIFORM: bool = true;
    const IS_UNIT_STRIDED: bool = false;
}

impl Layout for General {
    type NonUniform = Self;
    type NonUnitStrided = Strided;
    type Uniform = Dense;
    type UnitStrided = Self;

    type Merge<L: Layout> = L::NonUniform;
    type Mapping<S: Shape> = GeneralMapping<S>;

    const IS_UNIFORM: bool = false;
    const IS_UNIT_STRIDED: bool = true;
}

impl Layout for Strided {
    type NonUniform = Self;
    type NonUnitStrided = Self;
    type Uniform = Flat;
    type UnitStrided = General;

    type Merge<L: Layout> = Self;
    type Mapping<S: Shape> = StridedMapping<S>;

    const IS_UNIFORM: bool = false;
    const IS_UNIT_STRIDED: bool = false;
}

impl Uniform for Dense {}
impl Uniform for Flat {}

impl UnitStrided for Dense {}
impl UnitStrided for General {}
