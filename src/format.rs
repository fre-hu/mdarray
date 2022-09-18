use std::iter::FusedIterator;
use std::slice::{Iter, IterMut};

use crate::dim::Dim;
use crate::iter::{LinearIter, LinearIterMut};
use crate::mapping::{DenseMapping, FlatMapping, GeneralMapping, Mapping, StridedMapping};

/// Array format trait for memory layout.
pub trait Format {
    /// Corresponding format which may have non-uniform stride.
    type NonUniform: Format;

    /// Corresponding format which may have non-unit inner stride.
    type NonUnitStrided: Format;

    /// Corresponding format with uniform stride.
    type Uniform: Uniform;

    /// Corresponding format with unit inner stride.
    type UnitStrided: UnitStrided;

    /// Combined format based on the dimension.
    type Format<D: Dim, F: Format>: Format;

    /// Array iterator type.
    type Iter<'a, T: 'a>: Clone
        + DoubleEndedIterator
        + ExactSizeIterator
        + FusedIterator
        + Iterator<Item = &'a T>;

    /// Mutable array iterator type.
    type IterMut<'a, T: 'a>: DoubleEndedIterator
        + ExactSizeIterator
        + FusedIterator
        + Iterator<Item = &'a mut T>;

    #[doc(hidden)]
    type Mapping<D: Dim>: Mapping<Dim = D, Format = Self>;

    /// True if the format type has uniform stride.
    const IS_UNIFORM: bool;

    /// True if the format type has unit inner stride.
    const IS_UNIT_STRIDED: bool;
}

/// Trait for format types with uniform stride.
pub trait Uniform: Format {}

/// Trait for format types with unit inner stride.
pub trait UnitStrided: Format {}

/// Dense array format type.
pub struct Dense;

/// Flat array format type.
pub struct Flat;

/// General array format type.
pub struct General;

/// Strided array format type.
pub struct Strided;

impl Format for Dense {
    type NonUniform = General;
    type NonUnitStrided = Flat;
    type Uniform = Self;
    type UnitStrided = Self;

    type Format<D: Dim, F: Format> = D::Format<F>;

    type Iter<'a, T: 'a> = Iter<'a, T>;
    type IterMut<'a, T: 'a> = IterMut<'a, T>;

    type Mapping<D: Dim> = DenseMapping<D>;

    const IS_UNIFORM: bool = true;
    const IS_UNIT_STRIDED: bool = true;
}

impl Format for Flat {
    type NonUniform = Strided;
    type NonUnitStrided = Self;
    type Uniform = Self;
    type UnitStrided = Dense;

    type Format<D: Dim, F: Format> = D::Format<F::NonUnitStrided>;

    type Iter<'a, T: 'a> = LinearIter<'a, T>;
    type IterMut<'a, T: 'a> = LinearIterMut<'a, T>;

    type Mapping<D: Dim> = FlatMapping<D>;

    const IS_UNIFORM: bool = true;
    const IS_UNIT_STRIDED: bool = false;
}

impl Format for General {
    type NonUniform = Self;
    type NonUnitStrided = Strided;
    type Uniform = Dense;
    type UnitStrided = Self;

    type Format<D: Dim, F: Format> = D::Format<F::NonUniform>;

    type Iter<'a, T: 'a> = Iter<'a, T>;
    type IterMut<'a, T: 'a> = IterMut<'a, T>;

    type Mapping<D: Dim> = GeneralMapping<D>;

    const IS_UNIFORM: bool = false;
    const IS_UNIT_STRIDED: bool = true;
}

impl Format for Strided {
    type NonUniform = Self;
    type NonUnitStrided = Self;
    type Uniform = Flat;
    type UnitStrided = General;

    type Format<D: Dim, F: Format> = D::Format<Self>;

    type Iter<'a, T: 'a> = LinearIter<'a, T>;
    type IterMut<'a, T: 'a> = LinearIterMut<'a, T>;

    type Mapping<D: Dim> = StridedMapping<D>;

    const IS_UNIFORM: bool = false;
    const IS_UNIT_STRIDED: bool = false;
}

impl Uniform for Dense {}
impl Uniform for Flat {}

impl UnitStrided for Dense {}
impl UnitStrided for General {}
