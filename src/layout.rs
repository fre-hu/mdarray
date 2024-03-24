use crate::dim::Dim;
use crate::mapping::{DenseMapping, FlatMapping, GeneralMapping, Mapping, StridedMapping};

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

    /// Combined layout based on the dimension.
    type Layout<D: Dim, L: Layout>: Layout;

    /// Array layout mapping type.
    type Mapping<D: Dim>: Mapping<Dim = D, Layout = Self>;

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

    type Layout<D: Dim, L: Layout> = D::Layout<L>;
    type Mapping<D: Dim> = DenseMapping<D>;

    const IS_UNIFORM: bool = true;
    const IS_UNIT_STRIDED: bool = true;
}

impl Layout for Flat {
    type NonUniform = Strided;
    type NonUnitStrided = Self;
    type Uniform = Self;
    type UnitStrided = Dense;

    type Layout<D: Dim, L: Layout> = D::Layout<L::NonUnitStrided>;
    type Mapping<D: Dim> = FlatMapping<D>;

    const IS_UNIFORM: bool = true;
    const IS_UNIT_STRIDED: bool = false;
}

impl Layout for General {
    type NonUniform = Self;
    type NonUnitStrided = Strided;
    type Uniform = Dense;
    type UnitStrided = Self;

    type Layout<D: Dim, L: Layout> = D::Layout<L::NonUniform>;
    type Mapping<D: Dim> = GeneralMapping<D>;

    const IS_UNIFORM: bool = false;
    const IS_UNIT_STRIDED: bool = true;
}

impl Layout for Strided {
    type NonUniform = Self;
    type NonUnitStrided = Self;
    type Uniform = Flat;
    type UnitStrided = General;

    type Layout<D: Dim, L: Layout> = D::Layout<Self>;
    type Mapping<D: Dim> = StridedMapping<D>;

    const IS_UNIFORM: bool = false;
    const IS_UNIT_STRIDED: bool = false;
}

impl Uniform for Dense {}
impl Uniform for Flat {}

impl UnitStrided for Dense {}
impl UnitStrided for General {}
